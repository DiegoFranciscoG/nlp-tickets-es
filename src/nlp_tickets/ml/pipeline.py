"""Pipeline reproducible de entrenamiento y evaluación (``nlp-tickets train``).

Garantías de evaluación honesta:
- Misma partición y semilla para todos los modelos (hash ``split_fingerprint``).
- Hiperparámetros, temperatura y umbral se eligen en validación; la prueba se usa una vez.
- Las métricas del modelo final son las del artefacto ONNX INT8 que se despliega.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import statistics
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nlp_tickets.domain.catalog import Catalog, load_catalog
from nlp_tickets.ml import data as data_mod
from nlp_tickets.ml.baseline import fit_embedding_baseline, fit_tfidf_baseline
from nlp_tickets.ml.calibration import choose_threshold, fit_temperature, probs_to_logits, softmax
from nlp_tickets.ml.inference import META_FILE, OnnxTicketModel, sha256_file
from nlp_tickets.ml.metrics import (
    confusion,
    expected_calibration_error,
    paired_bootstrap_macro_f1,
    reliability_curve,
    selective_metrics,
    task_report,
)
from nlp_tickets.ml.routing import FloatArray, IntArray, RoutingMatrices
from nlp_tickets.ml.sources import (
    BANKING77_EN_COMMIT,
    BANKING77_ES_REVISION,
    BASE_MODEL,
    BASE_MODEL_REVISION,
    GOEMOTIONS_ES_REVISION,
    fetch_all,
)

log = logging.getLogger(__name__)
MODEL_NAME = "minilm-multitask"
DEFAULT_TARGET_ROUTING_ACCURACY = 0.99
TASKS = ("intent", "category", "team", "urgency", "sentiment")


@dataclass(frozen=True)
class PipelineOptions:
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("reports")
    manifest_path: Path = Path("models/manifest.json")
    feedback_path: Path | None = None
    epochs: int | None = None
    target_routing_accuracy: float = DEFAULT_TARGET_ROUTING_ACCURACY
    smoke: bool = False  # prueba de humo: submuestras pequeñas para validar el pipeline completo


@dataclass
class PreparedData:
    catalog: Catalog
    routing: RoutingMatrices
    intent: data_mod.Splits
    sentiment: data_mod.Splits
    ood: list[str]
    reports: dict[str, Any]

    @property
    def y_val(self) -> IntArray:
        return _labels(self.intent.validation)

    @property
    def y_test(self) -> IntArray:
        return _labels(self.intent.test)

    @property
    def ys_val(self) -> IntArray:
        return _labels(self.sentiment.validation)

    @property
    def ys_test(self) -> IntArray:
        return _labels(self.sentiment.test)


@dataclass
class IntentEvaluation:
    report: dict[str, Any]
    test_probs: FloatArray
    temperature: float
    threshold: float


@dataclass
class ModelOutcome:
    evaluation: IntentEvaluation
    sentiment_probs: FloatArray | None = None
    sentiment_temperature: float = 1.0
    test_logits: FloatArray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _labels(frame: pd.DataFrame) -> IntArray:
    return np.asarray(frame["label"].to_numpy(), dtype=np.int64)


def _argmax(probs: FloatArray, matrix: FloatArray | None = None) -> IntArray:
    projected = probs if matrix is None else probs @ matrix
    return np.asarray(projected.argmax(1), dtype=np.int64)


def evaluate_intent(
    *,
    val_logits: FloatArray,
    test_logits: FloatArray,
    ood_logits: FloatArray,
    data: PreparedData,
    target: float,
) -> IntentEvaluation:
    routing, y_val, y_test = data.routing, data.y_val, data.y_test
    temperature = fit_temperature(val_logits, y_val)
    p_val = softmax(val_logits, temperature)
    p_test = softmax(test_logits, temperature)
    p_ood = softmax(ood_logits, temperature)
    threshold = choose_threshold(
        routing.team_probs(p_val), routing.map_labels(y_val, routing.to_team), target
    )
    selective = selective_metrics(
        routing.team_probs(p_test), routing.map_labels(y_test, routing.to_team), threshold
    )
    report: dict[str, Any] = {
        "intent": task_report(p_test, y_test).as_dict(),
        "intent_ece_uncalibrated": round(
            expected_calibration_error(softmax(test_logits), y_test), 4
        ),
        "temperature": round(temperature, 4),
    }
    for task, matrix in (
        ("category", routing.to_category),
        ("team", routing.to_team),
        ("urgency", routing.to_urgency),
    ):
        report[task] = task_report(p_test @ matrix, routing.map_labels(y_test, matrix)).as_dict()
    report["routing"] = {
        "target_accuracy_validation": target,
        "threshold": threshold,
        "coverage_test": round(selective.coverage, 4),
        "auto_routed_accuracy_test": round(selective.selective_accuracy, 4),
        "human_review_rate_test": round(selective.human_review_rate, 4),
        "human_review_rate_out_of_domain": round(
            float((routing.team_probs(p_ood).max(axis=1) < threshold).mean()), 4
        ),
    }
    return IntentEvaluation(report, p_test, temperature, threshold)


def evaluate_sentiment(
    val_logits: FloatArray, test_logits: FloatArray, data: PreparedData
) -> tuple[dict[str, Any], FloatArray, float]:
    temperature = fit_temperature(val_logits, data.ys_val)
    probs = softmax(test_logits, temperature)
    report = {**task_report(probs, data.ys_test).as_dict(), "temperature": round(temperature, 4)}
    return report, probs, temperature


def _subsample(splits: data_mod.Splits, size: int = 400) -> data_mod.Splits:
    def take(frame: pd.DataFrame) -> pd.DataFrame:
        n = min(size, len(frame))
        return frame.sample(n, random_state=data_mod.SEED).reset_index(drop=True)

    return data_mod.Splits(take(splits.train), take(splits.validation), take(splits.test))


def _load_feedback(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_parquet(path)[["text", "label"]]


def prepare_data(options: PipelineOptions) -> PreparedData:
    paths = fetch_all()
    catalog = load_catalog()
    spanish, original = data_mod.load_raw_banking(paths.banking_es, paths.banking_en)
    intent, intent_report = data_mod.build_intent_splits(
        spanish, original, catalog, extra_train=_load_feedback(options.feedback_path)
    )
    sentiment, sentiment_report = data_mod.build_sentiment_splits(
        data_mod.load_raw_goemotions(paths.goemotions_es)
    )
    ood = data_mod.load_massive_ood(paths.massive_es)
    if options.smoke:
        intent, sentiment, ood = _subsample(intent), _subsample(sentiment), ood[:200]
    return PreparedData(
        catalog=catalog,
        routing=RoutingMatrices.from_catalog(catalog),
        intent=intent,
        sentiment=sentiment,
        ood=ood,
        reports={
            "intent": {**intent_report, **intent.sizes()},
            "sentiment": {**sentiment_report, **sentiment.sizes()},
            "out_of_domain_samples": len(ood),
        },
    )


def run_tfidf(data: PreparedData, target: float) -> ModelOutcome:
    """Línea base: TF-IDF + regresión logística (intención y sentimiento)."""
    intent_model = fit_tfidf_baseline(data.intent, len(data.catalog.intents))
    evaluation = evaluate_intent(
        val_logits=probs_to_logits(intent_model.predict_proba(data.intent.validation["text"])),
        test_logits=probs_to_logits(intent_model.predict_proba(data.intent.test["text"])),
        ood_logits=probs_to_logits(intent_model.predict_proba(data.ood)),
        data=data,
        target=target,
    )
    sentiment_model = fit_tfidf_baseline(data.sentiment, len(data_mod.SENTIMENT_LABELS))
    report, probs, temperature = evaluate_sentiment(
        probs_to_logits(sentiment_model.predict_proba(data.sentiment.validation["text"])),
        probs_to_logits(sentiment_model.predict_proba(data.sentiment.test["text"])),
        data,
    )
    evaluation.report.update(
        sentiment=report, c_intent=intent_model.c, c_sentiment=sentiment_model.c
    )
    return ModelOutcome(evaluation, probs, temperature)


def run_embedding_baseline(data: PreparedData, target: float) -> ModelOutcome:
    """Línea base intermedia: embeddings congelados del modelo base + regresión logística."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    encoder = SentenceTransformer(BASE_MODEL, revision=BASE_MODEL_REVISION, device="cpu")

    def encode(texts: list[str]) -> FloatArray:
        return np.asarray(encoder.encode(texts, batch_size=64, normalize_embeddings=True))

    classifier = fit_embedding_baseline(
        encode(data.intent.train["text"].tolist()),
        _labels(data.intent.train),
        len(data.catalog.intents),
    )
    evaluation = evaluate_intent(
        val_logits=probs_to_logits(
            classifier.predict_proba(encode(data.intent.validation["text"].tolist()))
        ),
        test_logits=probs_to_logits(
            classifier.predict_proba(encode(data.intent.test["text"].tolist()))
        ),
        ood_logits=probs_to_logits(classifier.predict_proba(encode(data.ood))),
        data=data,
        target=target,
    )
    return ModelOutcome(evaluation)


def _write_meta(model_dir: Path, payload: dict[str, Any]) -> None:
    (model_dir / META_FILE).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _latency_ms(model: OnnxTicketModel, texts: list[str]) -> dict[str, float]:
    timings = []
    for text in texts:
        started = time.perf_counter()
        model.predict(text)
        timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return {
        "p50": round(statistics.median(timings), 1),
        "p95": round(timings[int(0.95 * (len(timings) - 1))], 1),
    }


def run_transformer(
    data: PreparedData, options: PipelineOptions, version: str, fingerprint: str
) -> ModelOutcome:
    """Fine-tuning multitarea, exportación a ONNX INT8 y evaluación del artefacto exportado."""
    from nlp_tickets.ml.export_onnx import export_onnx  # noqa: PLC0415
    from nlp_tickets.ml.train_transformer import (  # noqa: PLC0415
        TrainConfig,
        predict_logits,
        train_multitask,
    )

    config = TrainConfig() if options.epochs is None else TrainConfig(epochs=options.epochs)
    trained = train_multitask(
        data.intent,
        data.sentiment,
        len(data.catalog.intents),
        len(data_mod.SENTIMENT_LABELS),
        config,
    )
    model_dir = options.artifacts_dir / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    fp32_path, int8_path = export_onnx(
        trained.model, trained.tokenizer, model_dir, max_length=config.max_length
    )
    meta: dict[str, Any] = {
        "name": MODEL_NAME,
        "version": version,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "intent_labels": data.catalog.intent_codes,
        "sentiment_labels": list(data_mod.SENTIMENT_LABELS),
        "temperature_intent": 1.0,
        "temperature_sentiment": 1.0,
        "confidence_threshold": 1.0,
        "max_length": config.max_length,
        "pad_id": int(trained.tokenizer.pad_token_id),
        "pad_token": str(trained.tokenizer.pad_token),
        "embedding_dim": int(trained.model.encoder.config.hidden_size),
        "onnx_sha256": sha256_file(int8_path),
        "dataset_revision": BANKING77_ES_REVISION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split_fingerprint": fingerprint,
    }
    _write_meta(model_dir, meta)

    # Calibración, umbral y métricas con el artefacto INT8 que realmente se despliega.
    onnx_model = OnnxTicketModel(model_dir)
    val_logits, _, _ = onnx_model.raw_outputs(data.intent.validation["text"].tolist())
    test_logits, _, _ = onnx_model.raw_outputs(data.intent.test["text"].tolist())
    ood_logits, _, _ = onnx_model.raw_outputs(data.ood)
    _, sent_val_logits, _ = onnx_model.raw_outputs(data.sentiment.validation["text"].tolist())
    _, sent_test_logits, _ = onnx_model.raw_outputs(data.sentiment.test["text"].tolist())
    target = options.target_routing_accuracy
    evaluation = evaluate_intent(
        val_logits=val_logits,
        test_logits=test_logits,
        ood_logits=ood_logits,
        data=data,
        target=target,
    )
    sentiment_report, sentiment_probs, sentiment_temperature = evaluate_sentiment(
        sent_val_logits, sent_test_logits, data
    )
    evaluation.report["sentiment"] = sentiment_report

    torch_test_logits, _, _ = predict_logits(
        trained.model,
        trained.tokenizer,
        data.intent.test["text"].tolist(),
        max_length=config.max_length,
    )
    parity = {
        "argmax_agreement": round(
            float((torch_test_logits.argmax(1) == test_logits.argmax(1)).mean()), 4
        ),
        "max_abs_logit_diff": round(float(np.abs(torch_test_logits - test_logits).max()), 4),
        "fp32_intent_macro_f1": task_report(softmax(torch_test_logits), data.y_test).macro_f1,
        "onnx_fp32_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "onnx_int8_mb": round(int8_path.stat().st_size / 1e6, 1),
    }
    fp32_path.unlink()
    shutil.rmtree(model_dir / "tokenizer", ignore_errors=True)
    _write_meta(
        model_dir,
        {
            **meta,
            "temperature_intent": evaluation.temperature,
            "temperature_sentiment": sentiment_temperature,
            "confidence_threshold": evaluation.threshold,
            "target_routing_accuracy": target,
        },
    )
    latency = _latency_ms(OnnxTicketModel(model_dir), data.intent.test["text"].tolist()[:200])
    return ModelOutcome(
        evaluation,
        sentiment_probs,
        sentiment_temperature,
        test_logits,
        extra={
            "parity": parity,
            "latency": latency,
            "training": {
                "config": asdict(config),
                "best_epoch": trained.best_epoch,
                "history": trained.history,
                "seconds": round(trained.seconds, 1),
            },
        },
    )


def compare(data: PreparedData, base: ModelOutcome, final: ModelOutcome) -> dict[str, Any]:
    """Bootstrap pareado sobre la misma partición de prueba (modelo final frente a TF-IDF)."""
    routing, y_test = data.routing, data.y_test
    if base.sentiment_probs is None or final.sentiment_probs is None:
        raise ValueError("faltan las probabilidades de sentimiento para comparar")
    return {
        "intent": asdict(
            paired_bootstrap_macro_f1(
                y_test, _argmax(base.evaluation.test_probs), _argmax(final.evaluation.test_probs)
            )
        ),
        "team": asdict(
            paired_bootstrap_macro_f1(
                routing.map_labels(y_test, routing.to_team),
                _argmax(base.evaluation.test_probs, routing.to_team),
                _argmax(final.evaluation.test_probs, routing.to_team),
            )
        ),
        "sentiment": asdict(
            paired_bootstrap_macro_f1(
                data.ys_test, _argmax(base.sentiment_probs), _argmax(final.sentiment_probs)
            )
        ),
    }


def write_figures(reports_dir: Path, data: PreparedData, models: dict[str, ModelOutcome]) -> None:
    from nlp_tickets.ml import reporting  # noqa: PLC0415

    catalog, routing, y_test = data.catalog, data.routing, data.y_test
    final, base = models["minilm_multitask_onnx_int8"], models["tfidf_logreg"]
    figures = reports_dir / "figures"
    labels = {
        "tfidf_logreg": "TF-IDF + reg. logística",
        "embeddings_logreg": "Embeddings congelados + RL",
        "minilm_multitask_onnx_int8": "MiniLM fine-tuning (ONNX INT8)",
    }
    reporting.plot_model_comparison(
        ["intención", "categoría", "equipo", "urgencia", "sentimiento"],
        {
            labels[name]: [
                outcome.evaluation.report[t]["macro_f1"]
                if t in outcome.evaluation.report
                else np.nan
                for t in TASKS
            ]
            for name, outcome in models.items()
        },
        figures / "comparison_macro_f1.png",
        "Macro-F1 en prueba: misma partición y semilla",
    )
    for task, matrix, codes, namer in (
        ("team", routing.to_team, routing.team_codes, catalog.team_name),
        ("category", routing.to_category, routing.category_codes, catalog.category_name),
    ):
        reporting.plot_confusion(
            confusion(
                routing.map_labels(y_test, matrix),
                _argmax(final.evaluation.test_probs, matrix),
                len(codes),
            ),
            [namer(c) for c in codes],
            figures / f"confusion_{task}.png",
            f"{'Equipo destino' if task == 'team' else 'Categoría'}: recall por fila",
        )
    if final.test_logits is None:
        raise ValueError("faltan los logits de prueba del modelo final")
    curves = {
        "transformer sin calibrar": reliability_curve(softmax(final.test_logits), y_test),
        "transformer calibrado": reliability_curve(final.evaluation.test_probs, y_test),
        "TF-IDF calibrado": reliability_curve(base.evaluation.test_probs, y_test),
    }
    reporting.plot_reliability(
        {name: (curve[0], curve[1]) for name, curve in curves.items()},
        figures / "reliability_intent.png",
        "Fiabilidad de la confianza (intención)",
    )
    reporting.plot_training_curve(
        final.extra["training"]["history"], figures / "training_curve.png", "Fine-tuning por época"
    )


def _bundle(model_dir: Path, artifacts_dir: Path, version: str) -> Path:
    bundle = artifacts_dir / f"nlp-tickets-model-{version}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("model.onnx", "tokenizer.json", META_FILE):
            archive.write(model_dir / name, arcname=name)
    return bundle


def write_manifest(
    options: PipelineOptions, version: str, fingerprint: str, report: dict[str, Any]
) -> None:
    model_dir = options.artifacts_dir / "model"
    bundle = _bundle(model_dir, options.artifacts_dir, version)
    final = report["models"]["minilm_multitask_onnx_int8"]
    manifest = {
        "name": MODEL_NAME,
        "version": version,
        "bundle": bundle.name,
        "bundle_sha256": sha256_file(bundle),
        "artifacts": [
            {
                "file": name,
                "sha256": sha256_file(model_dir / name),
                "bytes": (model_dir / name).stat().st_size,
            }
            for name in ("model.onnx", "tokenizer.json", META_FILE)
        ],
        "size_mb": report["parity_fp32_vs_int8"]["onnx_int8_mb"],
        "split_fingerprint": fingerprint,
        "test_macro_f1": {
            name: {t: m[t]["macro_f1"] for t in TASKS if t in m}
            for name, m in report["models"].items()
        },
        "confidence_threshold": final["routing"]["threshold"],
    }
    options.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    options.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _mlflow_setup() -> Any:
    import mlflow  # noqa: PLC0415

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("nlp-tickets-es")
    return mlflow


def run(options: PipelineOptions) -> dict[str, Any]:
    started = time.perf_counter()
    data = prepare_data(options)
    fingerprint = data.intent.fingerprint()
    version = f"{datetime.now(UTC):%Y.%m.%d}-{fingerprint[:8]}"
    target = options.target_routing_accuracy
    log.info("versión %s, particiones %s", version, data.intent.sizes())

    mlflow = _mlflow_setup()
    with mlflow.start_run(run_name=version) as run_info:
        mlflow.log_params(
            {
                "split_fingerprint": fingerprint,
                "banking77_es_revision": BANKING77_ES_REVISION,
                "banking77_en_commit": BANKING77_EN_COMMIT,
                "goemotions_es_revision": GOEMOTIONS_ES_REVISION,
                "base_model": BASE_MODEL,
                "base_model_revision": BASE_MODEL_REVISION,
                "seed": data_mod.SEED,
                "target_routing_accuracy": target,
                **{f"size_{k}": v for k, v in data.intent.sizes().items()},
            }
        )
        models = {
            "tfidf_logreg": run_tfidf(data, target),
            "embeddings_logreg": run_embedding_baseline(data, target),
            "minilm_multitask_onnx_int8": run_transformer(data, options, version, fingerprint),
        }
        final = models["minilm_multitask_onnx_int8"]
        write_figures(options.reports_dir, data, models)
        report: dict[str, Any] = {
            "version": version,
            "split_fingerprint": fingerprint,
            "sentiment_split_fingerprint": data.sentiment.fingerprint(),
            "data": data.reports,
            "models": {name: outcome.evaluation.report for name, outcome in models.items()},
            "comparisons_vs_tfidf": compare(data, models["tfidf_logreg"], final),
            "parity_fp32_vs_int8": final.extra["parity"],
            "latency_ms_single_text_local_cpu": final.extra["latency"],
            "training": final.extra["training"],
            "pipeline_seconds": round(time.perf_counter() - started, 1),
            "mlflow_run_id": run_info.info.run_id,
        }
        options.reports_dir.mkdir(parents=True, exist_ok=True)
        (options.reports_dir / "metrics.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
        )
        write_manifest(options, version, fingerprint, report)

        for name, model_report in report["models"].items():
            for task in TASKS:
                if task in model_report:
                    mlflow.log_metric(f"{name}.{task}.macro_f1", model_report[task]["macro_f1"])
            mlflow.log_metric(f"{name}.routing.coverage", model_report["routing"]["coverage_test"])
        mlflow.log_metrics({f"parity.{k}": float(v) for k, v in final.extra["parity"].items()})
        mlflow.log_artifacts(str(options.reports_dir), artifact_path="reports")
        mlflow.log_artifact(str(options.manifest_path))
    log.info("pipeline terminado en %.1f s", report["pipeline_seconds"])
    return report
