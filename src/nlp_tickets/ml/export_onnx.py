"""Exportación a ONNX + cuantización dinámica INT8 (M8) para servir en 512 MB sin PyTorch."""

from __future__ import annotations

from pathlib import Path

import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import PreTrainedTokenizerBase

from nlp_tickets.ml.train_transformer import MultiTaskEncoder

OPSET = 17
INPUT_NAMES = ["input_ids", "attention_mask"]
OUTPUT_NAMES = ["intent_logits", "sentiment_logits", "embedding"]


def export_onnx(
    model: MultiTaskEncoder,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    *,
    max_length: int = 64,
) -> tuple[Path, Path]:
    """Devuelve (modelo FP32, modelo INT8). El tokenizador se guarda como ``tokenizer.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fp32_path = output_dir / "model.fp32.onnx"
    int8_path = output_dir / "model.onnx"
    model.eval()
    sample = tokenizer(
        ["ejemplo de ticket para trazar el grafo"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dynamic = {0: "batch", 1: "sequence"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32_path),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes={
                "input_ids": dynamic,
                "attention_mask": dynamic,
                "intent_logits": {0: "batch"},
                "sentiment_logits": {0: "batch"},
                "embedding": {0: "batch"},
            },
            opset_version=OPSET,
            dynamo=False,
        )
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    tokenizer.save_pretrained(str(output_dir / "tokenizer"))
    (output_dir / "tokenizer" / "tokenizer.json").replace(output_dir / "tokenizer.json")
    return fp32_path, int8_path
