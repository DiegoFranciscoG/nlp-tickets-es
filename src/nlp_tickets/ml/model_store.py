"""Obtiene el paquete del modelo (GitHub Releases) y lo verifica por SHA-256 antes de usarlo."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)
REQUIRED_FILES = ("model.onnx", "tokenizer.json", "model_meta.json")
_MAX_BUNDLE_BYTES = 400 * 1024 * 1024


class ModelDownloadError(RuntimeError):
    pass


def model_present(model_dir: Path) -> bool:
    return all((model_dir / name).is_file() for name in REQUIRED_FILES)


def ensure_model(model_dir: Path, url: str = "", sha256: str = "") -> bool:
    """True si el modelo quedó disponible en ``model_dir``."""
    if model_present(model_dir):
        return True
    if not url:
        return False
    if not url.startswith("https://"):
        raise ModelDownloadError("MODEL_URL debe usar HTTPS")
    if len(sha256) != 64:
        raise ModelDownloadError("MODEL_SHA256 es obligatorio cuando se define MODEL_URL")
    log.info("descargando el modelo desde %s", url)
    digest = hashlib.sha256()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle.zip"
        with urllib.request.urlopen(url, timeout=120) as response, bundle.open("wb") as out:  # noqa: S310
            total = 0
            while chunk := response.read(1 << 20):
                total += len(chunk)
                if total > _MAX_BUNDLE_BYTES:
                    raise ModelDownloadError("el paquete del modelo excede el tamaño permitido")
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256.lower():
            raise ModelDownloadError("el SHA-256 del paquete no coincide con MODEL_SHA256")
        with zipfile.ZipFile(bundle) as archive:
            names = set(archive.namelist())
            if not set(REQUIRED_FILES) <= names or any("/" in n or ".." in n for n in names):
                raise ModelDownloadError("el paquete no tiene la estructura esperada")
            model_dir.mkdir(parents=True, exist_ok=True)
            for name in REQUIRED_FILES:
                (model_dir / name).write_bytes(archive.read(name))
    return True
