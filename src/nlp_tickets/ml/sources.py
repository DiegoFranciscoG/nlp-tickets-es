"""Fuentes de datos y modelo base, fijadas por revisión para un entrenamiento reproducible.

Ver docs/investigacion.md (D1, D2, D8, D9, M1) para licencias y citas.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

RAW_DIR = Path("data/raw")

BANKING77_ES_REPO = "c2d-usp/banking77-es-la"
BANKING77_ES_REVISION = "11f2c4d028e7df4d5e8e6ad36977cf9bf10cb666"
BANKING77_EN_COMMIT = "57ec275d8078af65b7731c2a98be812d844a6d6b"
BANKING77_EN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    f"{BANKING77_EN_COMMIT}/banking_data/{{split}}.csv"
)
GOEMOTIONS_ES_REPO = "mrm8488/go_emotions-es-mt"
GOEMOTIONS_ES_REVISION = "f881ecdb455e1ef7b7e70164df594a98ddf3424e"
GOEMOTIONS_ES_FILES = {
    "train": "data/train-00000-of-00001-1a52cf3c7c7bc803.parquet",
    "validation": "data/validation-00000-of-00001-f19ff191d245b1cc.parquet",
    "test": "data/test-00000-of-00001-b851a1e51ef87315.parquet",
}
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)
MASSIVE_LOCALE_MEMBER = "1.1/data/es-ES.jsonl"

BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASE_MODEL_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"


@dataclass(frozen=True)
class RawPaths:
    banking_es: dict[str, Path]
    banking_en: dict[str, Path]
    goemotions_es: dict[str, Path]
    massive_es: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        if not url.startswith("https://"):
            raise ValueError("solo se permiten descargas HTTPS")
        urllib.request.urlretrieve(url, target)  # noqa: S310  (esquema validado arriba)
    return target


def _hf_file(repo: str, filename: str, revision: str, target_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download  # noqa: PLC0415  (dependencia de entrenamiento)

    return Path(
        hf_hub_download(
            repo, filename, repo_type="dataset", revision=revision, local_dir=target_dir
        )
    )


def _extract_massive(raw_dir: Path) -> Path:
    target = raw_dir / "massive" / "es-ES.jsonl"
    if target.exists():
        return target
    archive = _download(MASSIVE_URL, raw_dir / "massive" / "amazon-massive-dataset-1.1.tar.gz")
    with tarfile.open(archive, "r:gz") as tar:
        member = tar.getmember(MASSIVE_LOCALE_MEMBER)
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(MASSIVE_LOCALE_MEMBER)
        target.write_bytes(extracted.read())
    archive.unlink()
    return target


def fetch_all(raw_dir: Path = RAW_DIR) -> RawPaths:
    """Descarga (una sola vez) todas las fuentes y guarda un manifiesto con sus SHA-256."""
    banking_es = {
        split: _hf_file(
            BANKING77_ES_REPO,
            f"banking77-es-la-{split}.csv",
            BANKING77_ES_REVISION,
            raw_dir / "banking77_es_la",
        )
        for split in ("train", "test")
    }
    banking_en = {
        split: _download(
            BANKING77_EN_URL.format(split=split), raw_dir / "banking77_en" / f"{split}.csv"
        )
        for split in ("train", "test")
    }
    goemotions = {
        split: _hf_file(GOEMOTIONS_ES_REPO, name, GOEMOTIONS_ES_REVISION, raw_dir / "goemotions_es")
        for split, name in GOEMOTIONS_ES_FILES.items()
    }
    paths = RawPaths(banking_es, banking_en, goemotions, _extract_massive(raw_dir))
    manifest = {
        path.resolve().relative_to(raw_dir.resolve()).as_posix(): sha256_file(path)
        for group in (banking_es, banking_en, goemotions)
        for path in group.values()
    }
    manifest["massive/es-ES.jsonl"] = sha256_file(paths.massive_es)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return paths
