import hashlib
import io
import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from nlp_tickets.ml import model_store
from nlp_tickets.ml.inference import ModelIntegrityError, ModelMeta, OnnxTicketModel, softmax
from nlp_tickets.security.api_key import ApiKeyAuthenticator, Role
from nlp_tickets.security.rate_limit import SlidingWindowRateLimiter

CLIENT, REVIEWER = "c" * 32, "r" * 32


class TestApiKeys:
    def test_roles(self) -> None:
        auth = ApiKeyAuthenticator([CLIENT], [REVIEWER])
        client = auth.authenticate(CLIENT)
        reviewer = auth.authenticate(REVIEWER)
        assert client is not None
        assert client.role is Role.CLIENT
        assert not client.can_review
        assert reviewer is not None
        assert reviewer.role is Role.REVIEWER
        assert reviewer.can_review

    @pytest.mark.parametrize("key", [None, "", "x" * 32, CLIENT[:-1]])
    def test_rechaza_claves_invalidas(self, key: str | None) -> None:
        assert ApiKeyAuthenticator([CLIENT], [REVIEWER]).authenticate(key) is None


class TestRateLimit:
    def test_ventana_deslizante(self) -> None:
        now = [0.0]
        limiter = SlidingWindowRateLimiter(2, window_seconds=60, clock=lambda: now[0])
        assert limiter.allow("a") == (True, 0)
        assert limiter.allow("a")[0]
        allowed, retry = limiter.allow("a")
        assert not allowed
        assert retry > 0
        assert limiter.allow("b")[0]  # otra identidad no se ve afectada
        now[0] = 61.0
        assert limiter.allow("a")[0]

    def test_expulsa_identidades_viejas(self) -> None:
        now = [0.0]
        limiter = SlidingWindowRateLimiter(
            1, window_seconds=10, clock=lambda: now[0], max_identities=2
        )
        limiter.allow("a")
        limiter.allow("b")
        now[0] = 20.0
        assert limiter.allow("c")[0]


def _bundle(files: Mapping[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


MODEL_FILES = MappingProxyType(
    {"model.onnx": b"onnx", "tokenizer.json": b"{}", "model_meta.json": b"{}"}
)


class TestModelStore:
    FILES = MODEL_FILES

    def test_modelo_presente_no_descarga(self, tmp_path: Path) -> None:
        for name, content in self.FILES.items():
            (tmp_path / name).write_bytes(content)
        assert model_store.ensure_model(tmp_path, "https://example.org/x.zip", "0" * 64)

    def test_sin_url_no_hay_modelo(self, tmp_path: Path) -> None:
        assert not model_store.ensure_model(tmp_path)

    def test_exige_https_y_hash(self, tmp_path: Path) -> None:
        with pytest.raises(model_store.ModelDownloadError, match="HTTPS"):
            model_store.ensure_model(tmp_path, "http://example.org/x.zip", "0" * 64)
        with pytest.raises(model_store.ModelDownloadError, match="SHA256"):
            model_store.ensure_model(tmp_path, "https://example.org/x.zip", "")

    def test_descarga_verificada(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _bundle(self.FILES)
        monkeypatch.setattr(
            model_store.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        sha = hashlib.sha256(payload).hexdigest()
        assert model_store.ensure_model(tmp_path / "m", "https://example.org/x.zip", sha)
        assert model_store.model_present(tmp_path / "m")

    def test_hash_incorrecto(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = _bundle(self.FILES)
        monkeypatch.setattr(
            model_store.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        with pytest.raises(model_store.ModelDownloadError, match="no coincide"):
            model_store.ensure_model(tmp_path / "m", "https://example.org/x.zip", "a" * 64)

    def test_zip_con_rutas_peligrosas(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _bundle({**self.FILES, "../evil.txt": b"x"})
        monkeypatch.setattr(
            model_store.urllib.request, "urlopen", lambda *a, **k: _FakeResponse(payload)
        )
        with pytest.raises(model_store.ModelDownloadError, match="estructura"):
            model_store.ensure_model(
                tmp_path / "m", "https://example.org/x.zip", hashlib.sha256(payload).hexdigest()
            )


class TestInferenciaSinModelo:
    def test_softmax_por_filas(self) -> None:
        probs = softmax(np.array([[0.0, 0.0], [10.0, 0.0]]), temperature=2.0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert probs[1, 0] > 0.99

    def test_meta_con_campos_extra(self, tmp_path: Path) -> None:
        meta = {
            "name": "m",
            "version": "v",
            "intent_labels": ["a"],
            "sentiment_labels": ["n"],
            "temperature_intent": 1.0,
            "temperature_sentiment": 1.0,
            "confidence_threshold": 0.5,
            "max_length": 64,
            "pad_id": 1,
            "pad_token": "<pad>",
            "embedding_dim": 384,
            "onnx_sha256": "0" * 64,
            "dataset_revision": "r",
            "trained_at": "2026-09-25T00:00:00+00:00",
            "split_fingerprint": "abc",
        }
        (tmp_path / "model_meta.json").write_text(json.dumps(meta))
        loaded = ModelMeta.load(tmp_path)
        assert loaded.extra == {"split_fingerprint": "abc"}
        (tmp_path / "model.onnx").write_bytes(b"alterado")
        with pytest.raises(ModelIntegrityError):
            OnnxTicketModel(tmp_path)
