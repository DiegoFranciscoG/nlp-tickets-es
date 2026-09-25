"""Fine-tuning multitarea de paraphrase-multilingual-MiniLM-L12-v2 (M1, M10).

Un encoder compartido con dos cabezas lineales (intención y sentimiento) entrenadas con lotes
alternados. El mismo *mean pooling* normalizado sirve de embedding para la búsqueda en pgvector.
"""

from __future__ import annotations

import copy
import logging
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerBase

from nlp_tickets.ml.data import SEED, Splits
from nlp_tickets.ml.metrics import macro_f1
from nlp_tickets.ml.sources import BASE_MODEL, BASE_MODEL_REVISION

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    sentiment_batch_size: int = 16
    max_length: int = 64
    encoder_lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    sentiment_loss_weight: float = 0.5
    dropout: float = 0.1
    freeze_word_embeddings: bool = True
    seed: int = SEED


class MultiTaskEncoder(nn.Module):
    def __init__(self, n_intents: int, n_sentiments: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
        hidden = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hidden, n_intents)
        self.sentiment_head = nn.Linear(hidden, n_sentiments)

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled: torch.Tensor = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pooled = self.pooled(input_ids, attention_mask)
        features = self.dropout(pooled)
        return (
            self.intent_head(features),
            self.sentiment_head(features),
            F.normalize(pooled, p=2, dim=-1),
        )


@dataclass
class TrainResult:
    model: MultiTaskEncoder
    tokenizer: PreTrainedTokenizerBase
    config: TrainConfig
    best_epoch: int
    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _batches(
    texts: list[str], labels: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[list[str], np.ndarray]]:
    order = rng.permutation(len(texts))
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield [texts[i] for i in idx], labels[idx]


def _cycle(
    texts: list[str], labels: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterator[tuple[list[str], np.ndarray]]:
    while True:
        yield from _batches(texts, labels, batch_size, rng)


@torch.no_grad()
def predict_logits(
    model: MultiTaskEncoder,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    *,
    max_length: int = 64,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    intents, sentiments, embeddings = [], [], []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        intent, sentiment, embedding = model(encoded["input_ids"], encoded["attention_mask"])
        intents.append(intent.numpy())
        sentiments.append(sentiment.numpy())
        embeddings.append(embedding.numpy())
    return np.concatenate(intents), np.concatenate(sentiments), np.concatenate(embeddings)


def train_multitask(
    intent_splits: Splits,
    sentiment_splits: Splits,
    n_intents: int,
    n_sentiments: int,
    config: TrainConfig | None = None,
) -> TrainResult:
    config = config or TrainConfig()
    set_seed(config.seed)
    # Se respeta el número de hilos por defecto (núcleos físicos): usar hilos lógicos
    # (hyperthreading) resultó 5 veces más lento en CPU.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_MODEL_REVISION)
    model = MultiTaskEncoder(n_intents, n_sentiments, config.dropout)
    if config.freeze_word_embeddings:
        # 96 M de los 118 M parámetros son el vocabulario multilingüe: congelarlo acelera el
        # entrenamiento en CPU y evita sobreajustar embeddings de tokens poco vistos.
        model.encoder.embeddings.word_embeddings.weight.requires_grad = False

    head_params = [*model.intent_head.parameters(), *model.sentiment_head.parameters()]
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": config.encoder_lr},
            {"params": head_params, "lr": config.head_lr},
        ],
        weight_decay=config.weight_decay,
    )
    intent_texts = intent_splits.train["text"].tolist()
    intent_labels = intent_splits.train["label"].to_numpy()
    steps_per_epoch = -(-len(intent_texts) // config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    warmup = max(1, int(total_steps * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (
            min(1.0, (step + 1) / warmup)
            * max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )
    rng = np.random.default_rng(config.seed)
    sentiment_stream = _cycle(
        sentiment_splits.train["text"].tolist(),
        sentiment_splits.train["label"].to_numpy(),
        config.sentiment_batch_size,
        rng,
    )

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return dict(
            tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
        )

    best_score, best_state, best_epoch = -1.0, copy.deepcopy(model.state_dict()), 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        for step, (texts, labels) in enumerate(
            _batches(intent_texts, intent_labels, config.batch_size, rng), start=1
        ):
            sent_texts, sent_labels = next(sentiment_stream)
            batch = encode(texts)
            intent_logits, _, _ = model(batch["input_ids"], batch["attention_mask"])
            loss = F.cross_entropy(intent_logits, torch.as_tensor(labels))
            sent_batch = encode(sent_texts)
            _, sentiment_logits, _ = model(sent_batch["input_ids"], sent_batch["attention_mask"])
            loss = loss + config.sentiment_loss_weight * F.cross_entropy(
                sentiment_logits, torch.as_tensor(sent_labels)
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                log.info(
                    "época %d paso %d/%d pérdida %.4f", epoch, step, steps_per_epoch, running / step
                )

        val_intent, _, _ = predict_logits(
            model,
            tokenizer,
            intent_splits.validation["text"].tolist(),
            max_length=config.max_length,
        )
        _, val_sentiment, _ = predict_logits(
            model,
            tokenizer,
            sentiment_splits.validation["text"].tolist(),
            max_length=config.max_length,
        )
        intent_f1 = macro_f1(intent_splits.validation["label"].to_numpy(), val_intent.argmax(1))
        sentiment_f1 = macro_f1(
            sentiment_splits.validation["label"].to_numpy(), val_sentiment.argmax(1)
        )
        record = {
            "epoch": float(epoch),
            "train_loss": running / steps_per_epoch,
            "val_intent_macro_f1": intent_f1,
            "val_sentiment_macro_f1": sentiment_f1,
            "elapsed_s": time.perf_counter() - started,
        }
        history.append(record)
        log.info("época %d: %s", epoch, record)
        if intent_f1 > best_score:  # selección por la tarea principal, siempre en validación
            best_score, best_state, best_epoch = intent_f1, copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model,
        tokenizer=tokenizer,
        config=config,
        best_epoch=best_epoch,
        history=history,
        seconds=time.perf_counter() - started,
    )
