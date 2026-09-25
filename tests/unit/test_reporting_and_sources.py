from pathlib import Path

import numpy as np
import pytest

from nlp_tickets.ml import sources


def test_sha256_de_archivo(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    assert sources.sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_descarga_solo_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        sources._download("http://example.org/data.csv", tmp_path / "x.csv")


def test_descarga_no_repite_si_existe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "x.csv"
    target.write_text("ya estaba")

    def fail(*_: object) -> None:
        raise AssertionError("no debía descargar")

    monkeypatch.setattr(sources.urllib.request, "urlretrieve", fail)
    assert sources._download("https://example.org/x.csv", target) == target


def test_figuras(tmp_path: Path) -> None:
    reporting = pytest.importorskip("nlp_tickets.ml.reporting")
    reporting.plot_model_comparison(
        ["a", "b"], {"m1": [0.8, 0.9], "m2": [0.85, np.nan]}, tmp_path / "c.png", "t"
    )
    reporting.plot_confusion(np.array([[3, 1], [0, 4]]), ["x", "y"], tmp_path / "m.png", "t")
    reporting.plot_reliability({"r": ([0.5, 0.9], [0.4, 0.95])}, tmp_path / "r.png", "t")
    reporting.plot_training_curve(
        [
            {"epoch": 1.0, "val_intent_macro_f1": 0.5, "val_sentiment_macro_f1": 0.4},
            {"epoch": 2.0, "val_intent_macro_f1": 0.8, "val_sentiment_macro_f1": 0.6},
        ],
        tmp_path / "t.png",
        "t",
    )
    assert all((tmp_path / f"{name}.png").stat().st_size > 1000 for name in "cmrt")
