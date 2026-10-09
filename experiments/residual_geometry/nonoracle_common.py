from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-9
FEATURE_DIM = VOCAB_SIZE + 7
ALPHA = 0.8


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def n_tokens_from_ppmd(ppmd_path: Path) -> int:
    row_bytes = VOCAB_SIZE * np.dtype(np.float16).itemsize
    size = ppmd_path.stat().st_size
    if size % row_bytes:
        raise ValueError(f"{ppmd_path} is not a whole number of probability rows")
    return size // row_bytes


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def build_ppmd_features(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    logq = np.log(np.clip(q, EPS, None))
    centered = logq - logq.mean(axis=1, keepdims=True)
    sorted_q = np.sort(q, axis=1)[:, ::-1]
    entropy = -np.sum(q * logq, axis=1, keepdims=True)
    top1 = sorted_q[:, 0:1]
    top2 = sorted_q[:, 1:2]
    margin = top1 - top2
    top4 = sorted_q[:, :4].sum(axis=1, keepdims=True)
    top8 = sorted_q[:, :8].sum(axis=1, keepdims=True)
    top16 = sorted_q[:, :16].sum(axis=1, keepdims=True)
    return np.concatenate([centered, entropy, top1, top2, margin, top4, top8, top16], axis=1)


def feature_stats(slice_dirs: list[Path], chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    total = 0
    sum_x = np.zeros(FEATURE_DIM, dtype=np.float64)
    sum_x2 = np.zeros(FEATURE_DIM, dtype=np.float64)
    for slice_dir in slice_dirs:
        n = n_tokens_from_ppmd(slice_dir / "ppmd.float16")
        q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n, VOCAB_SIZE))
        for start, end in iter_slices(n, chunk_size):
            x = build_ppmd_features(q[start:end]).astype(np.float64)
            total += x.shape[0]
            sum_x += x.sum(axis=0)
            sum_x2 += (x * x).sum(axis=0)
    mean = sum_x / total
    var = np.maximum(sum_x2 / total - mean * mean, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
