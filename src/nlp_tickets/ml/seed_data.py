"""Construye ``data/seed/tickets_seed.csv``: tickets históricos resueltos para la demo.

Solo usa la partición de entrenamiento (sin frases que estén en prueba) para que la búsqueda
de tickets parecidos no filtre información de evaluación. Canal y fecha son ficticios.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from nlp_tickets.domain.catalog import Catalog
from nlp_tickets.domain.enums import Channel
from nlp_tickets.ml.data import SEED, Splits

SEED_PATH = Path("data/seed/tickets_seed.csv")
CHANNEL_WEIGHTS = {
    Channel.CHAT: 0.35,
    Channel.WEB: 0.25,
    Channel.EMAIL: 0.2,
    Channel.PHONE: 0.1,
    Channel.SOCIAL: 0.05,
    Channel.BRANCH: 0.05,
}


def build_seed_frame(splits: Splits, catalog: Catalog, per_intent: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    sampled = (
        splits.train.sample(frac=1.0, random_state=SEED)
        .groupby("label")
        .head(per_intent)
        .reset_index(drop=True)
    )
    start = datetime(2026, 1, 5, 8, 0, tzinfo=UTC)
    minutes = rng.integers(0, 60 * 24 * 250, len(sampled))
    channels = rng.choice(
        [c.value for c in CHANNEL_WEIGHTS], size=len(sampled), p=list(CHANNEL_WEIGHTS.values())
    )
    frame = pd.DataFrame(
        {
            "text": sampled["text"],
            "intent_code": [catalog.intents[int(label)].code for label in sampled["label"]],
            "channel": channels,
            "created_at": [
                (start + timedelta(minutes=int(m))).isoformat(timespec="minutes") for m in minutes
            ],
        }
    )
    return frame.sort_values("created_at").reset_index(drop=True)
