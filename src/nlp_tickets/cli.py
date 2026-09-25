"""Línea de comandos: ``nlp-tickets <comando>``.

Flujo típico: fetch-data -> train -> migrate -> register-model -> seed -> (API) ->
export-feedback -> train --feedback -> register-model --feedback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger("nlp_tickets")
SEED_CSV = Path("data/seed/tickets_seed.csv")
REPORT = Path("reports/metrics.json")


def _cmd_fetch_data(_: argparse.Namespace) -> None:
    from nlp_tickets.ml.sources import fetch_all  # noqa: PLC0415

    fetch_all()
    log.info("datos descargados en data/raw (ver data/raw/manifest.json)")


def _cmd_train(args: argparse.Namespace) -> None:
    from nlp_tickets.ml.pipeline import PipelineOptions, run  # noqa: PLC0415

    output = Path(args.output_dir)
    metrics = run(
        PipelineOptions(
            artifacts_dir=output / "artifacts",
            reports_dir=output / "reports",
            manifest_path=output / "models" / "manifest.json",
            feedback_path=Path(args.feedback) if args.feedback else None,
            epochs=args.epochs,
            target_routing_accuracy=args.target_accuracy,
            smoke=args.smoke,
        )
    )
    summary = {
        name: {task: report[task]["macro_f1"] for task in ("intent", "team") if task in report}
        for name, report in metrics["models"].items()
    }
    print(json.dumps({"version": metrics["version"], "test_macro_f1": summary}, indent=2))


def _cmd_build_seed(_: argparse.Namespace) -> None:
    from nlp_tickets.domain.catalog import load_catalog  # noqa: PLC0415
    from nlp_tickets.ml import data as data_mod  # noqa: PLC0415
    from nlp_tickets.ml.seed_data import build_seed_frame  # noqa: PLC0415
    from nlp_tickets.ml.sources import fetch_all  # noqa: PLC0415

    paths = fetch_all()
    catalog = load_catalog()
    spanish, original = data_mod.load_raw_banking(paths.banking_es, paths.banking_en)
    splits, _report = data_mod.build_intent_splits(spanish, original, catalog)
    SEED_CSV.parent.mkdir(parents=True, exist_ok=True)
    build_seed_frame(splits, catalog).to_csv(SEED_CSV, index=False)
    log.info("seed escrito en %s", SEED_CSV)


def _cmd_migrate(_: argparse.Namespace) -> None:
    from alembic import command  # noqa: PLC0415
    from alembic.config import Config  # noqa: PLC0415

    command.upgrade(Config("alembic.ini"), "head")


def _load_onnx_model() -> object:
    from nlp_tickets.config import get_settings  # noqa: PLC0415
    from nlp_tickets.ml.inference import OnnxTicketModel  # noqa: PLC0415
    from nlp_tickets.ml.model_store import ensure_model  # noqa: PLC0415

    settings = get_settings()
    if not ensure_model(settings.model_dir, settings.model_url, settings.model_sha256):
        return None
    return OnnxTicketModel(settings.model_dir, expected_sha256=settings.model_sha256)


def _cmd_fetch_model(_: argparse.Namespace) -> None:
    from nlp_tickets.config import get_settings  # noqa: PLC0415
    from nlp_tickets.ml.model_store import ensure_model  # noqa: PLC0415

    settings = get_settings()
    if not ensure_model(settings.model_dir, settings.model_url, settings.model_sha256):
        raise SystemExit("define MODEL_URL y MODEL_SHA256 (paquete publicado en GitHub Releases)")
    log.info("modelo disponible en %s", settings.model_dir)


def _cmd_register_model(args: argparse.Namespace) -> None:
    from nlp_tickets.db.session import get_session_factory  # noqa: PLC0415
    from nlp_tickets.ml.inference import OnnxTicketModel  # noqa: PLC0415
    from nlp_tickets.repositories.models import ModelRepository  # noqa: PLC0415
    from nlp_tickets.repositories.tickets import FeedbackRepository  # noqa: PLC0415
    from nlp_tickets.services.registry_service import ModelRegistryService  # noqa: PLC0415

    model = _load_onnx_model()
    if not isinstance(model, OnnxTicketModel):
        raise SystemExit("no hay modelo en MODEL_DIR (entrena o define MODEL_URL)")
    with get_session_factory()() as session:
        version = ModelRegistryService(ModelRepository(session)).register(model.meta, REPORT)
        if args.feedback:
            import pandas as pd  # noqa: PLC0415

            ids = pd.read_parquet(args.feedback)["feedback_id"].tolist()
            marked = FeedbackRepository(session).mark_used(ids, version.id)
            log.info("%d correcciones marcadas como usadas por %s", marked, version.version)
        session.commit()
    log.info("modelo %s:%s registrado y activo", model.meta.name, model.meta.version)


def _cmd_seed(args: argparse.Namespace) -> None:
    from nlp_tickets.db.session import get_session_factory  # noqa: PLC0415
    from nlp_tickets.ml.inference import OnnxTicketModel  # noqa: PLC0415
    from nlp_tickets.repositories.models import CatalogRepository  # noqa: PLC0415
    from nlp_tickets.repositories.tickets import TicketRepository  # noqa: PLC0415
    from nlp_tickets.services.seed_service import SeedService  # noqa: PLC0415

    model = _load_onnx_model()
    if not isinstance(model, OnnxTicketModel):
        log.warning("sin modelo no se pueden calcular embeddings: se omite el seed")
        return
    with get_session_factory()() as session:
        catalog = CatalogRepository(session).load()
        inserted = SeedService(catalog, TicketRepository(session), model).seed(Path(args.file))
        session.commit()
    log.info("%d tickets de ejemplo insertados", inserted)


def _cmd_export_feedback(args: argparse.Namespace) -> None:
    import pandas as pd  # noqa: PLC0415

    from nlp_tickets.db.session import get_session_factory  # noqa: PLC0415
    from nlp_tickets.repositories.models import CatalogRepository  # noqa: PLC0415
    from nlp_tickets.repositories.tickets import FeedbackRepository  # noqa: PLC0415
    from nlp_tickets.services.feedback_service import TrainingDataExporter  # noqa: PLC0415

    with get_session_factory()() as session:
        catalog = CatalogRepository(session).load()
        rows = TrainingDataExporter(catalog, FeedbackRepository(session)).export(args.max_rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "feedback_id": [str(r.feedback_id) for r in rows],
            "text": [r.text for r in rows],
            "label": [r.label for r in rows],
        }
    ).to_parquet(out, index=False)
    log.info("%d correcciones exportadas a %s", len(rows), out)


def _cmd_update_readme(_: argparse.Namespace) -> None:
    from nlp_tickets.ml.summary import update_readme  # noqa: PLC0415

    readme = Path("README.md")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    readme.write_text(update_readme(readme.read_text(encoding="utf-8"), report), encoding="utf-8")
    log.info("README actualizado con las métricas de %s", report["version"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nlp-tickets", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-data", help="descarga los datasets fijados por revisión").set_defaults(
        func=_cmd_fetch_data
    )
    train = sub.add_parser("train", help="entrena baseline + transformer y exporta ONNX")
    train.add_argument("--feedback", help="parquet con correcciones humanas (text, label)")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--target-accuracy", type=float, default=0.99)
    train.add_argument("--output-dir", default=".", help="raíz para artifacts/, reports/, models/")
    train.add_argument("--smoke", action="store_true", help="submuestras pequeñas (prueba rápida)")
    train.set_defaults(func=_cmd_train)

    sub.add_parser("build-seed", help="genera data/seed/tickets_seed.csv").set_defaults(
        func=_cmd_build_seed
    )
    sub.add_parser("migrate", help="aplica las migraciones de Alembic").set_defaults(
        func=_cmd_migrate
    )
    sub.add_parser("fetch-model", help="descarga y verifica el modelo (MODEL_URL)").set_defaults(
        func=_cmd_fetch_model
    )
    register = sub.add_parser("register-model", help="registra y activa el modelo de MODEL_DIR")
    register.add_argument(
        "--feedback", help="parquet exportado: marca esas correcciones como usadas"
    )
    register.set_defaults(func=_cmd_register_model)

    seed = sub.add_parser("seed", help="carga tickets de ejemplo con embeddings (idempotente)")
    seed.add_argument("--file", default=str(SEED_CSV))
    seed.set_defaults(func=_cmd_seed)

    export = sub.add_parser("export-feedback", help="exporta correcciones aceptadas a parquet")
    export.add_argument("--out", default="data/processed/feedback.parquet")
    export.add_argument("--max-rows", type=int, default=2000)
    export.set_defaults(func=_cmd_export_feedback)
    sub.add_parser(
        "update-readme", help="inserta las métricas de reports/metrics.json en el README"
    ).set_defaults(func=_cmd_update_readme)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
