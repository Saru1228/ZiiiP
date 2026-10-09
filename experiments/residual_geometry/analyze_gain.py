#!/usr/bin/env python3
"""Analyze where saved Transformer probabilities beat PPMD probabilities."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--ppmd", required=True, type=Path)
    parser.add_argument("--transformer", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def bucket_stats(values: np.ndarray, gain: np.ndarray, buckets: int = 4):
    edges = np.quantile(values, np.linspace(0, 1, buckets + 1))
    rows = []
    for i in range(buckets):
        if i == buckets - 1:
            mask = (values >= edges[i]) & (values <= edges[i + 1])
        else:
            mask = (values >= edges[i]) & (values < edges[i + 1])
        g = gain[mask]
        rows.append(
            {
                "bucket": i + 1,
                "low": float(edges[i]),
                "high": float(edges[i + 1]),
                "count": int(mask.sum()),
                "mean_gain_bits": float(g.mean()),
                "median_gain_bits": float(np.median(g)),
                "positive_gain_fraction": float((g > 0).mean()),
                "gain_sum_bits": float(g.sum()),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.fromfile(args.tokens, dtype=np.uint8)
    n_tokens = int(tokens.shape[0])
    row_bytes = VOCAB_SIZE * np.dtype(np.float16).itemsize
    for path in [args.ppmd, args.transformer]:
        if path.stat().st_size != n_tokens * row_bytes:
            raise ValueError(f"{path} size does not match tokens x vocab x float16")

    q = np.memmap(args.ppmd, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(
        args.transformer, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE)
    )

    n_eval = n_tokens - 1
    targets = tokens[1:].astype(np.int64)
    gain = np.empty(n_eval, dtype=np.float32)
    entropy = np.empty(n_eval, dtype=np.float32)
    top1 = np.empty(n_eval, dtype=np.float32)
    margin = np.empty(n_eval, dtype=np.float32)

    for start, end in iter_slices(n_eval, args.chunk_size):
        pp = np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None)
        tt = np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None)
        target = targets[start:end]

        gain[start:end] = np.log2(tt[np.arange(end - start), target] / pp[np.arange(end - start), target])
        entropy[start:end] = -(pp * np.log2(pp)).sum(axis=1)
        part = np.partition(pp, -2, axis=1)
        first = part[:, -1]
        second = part[:, -2]
        top1[start:end] = first
        margin[start:end] = first - second

    summary = {
        "count": int(n_eval),
        "gain_mean_bits": float(gain.mean()),
        "gain_median_bits": float(np.median(gain)),
        "gain_sum_bits": float(gain.sum()),
        "positive_gain_fraction": float((gain > 0).mean()),
        "negative_gain_fraction": float((gain < 0).mean()),
        "gain_p01": float(np.quantile(gain, 0.01)),
        "gain_p05": float(np.quantile(gain, 0.05)),
        "gain_p25": float(np.quantile(gain, 0.25)),
        "gain_p75": float(np.quantile(gain, 0.75)),
        "gain_p95": float(np.quantile(gain, 0.95)),
        "gain_p99": float(np.quantile(gain, 0.99)),
        "entropy_corr": float(np.corrcoef(gain, entropy)[0, 1]),
        "top1_corr": float(np.corrcoef(gain, top1)[0, 1]),
        "margin_corr": float(np.corrcoef(gain, margin)[0, 1]),
    }
    (args.out_dir / "gain_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    write_csv(args.out_dir / "gain_by_entropy_bucket.csv", bucket_stats(entropy, gain))
    write_csv(args.out_dir / "gain_by_top1_bucket.csv", bucket_stats(top1, gain))
    write_csv(args.out_dir / "gain_by_margin_bucket.csv", bucket_stats(margin, gain))

    sample_rows = []
    for idx in np.argsort(gain)[-20:][::-1]:
        sample_rows.append(
            {
                "row": int(idx),
                "target_token": int(targets[idx]),
                "gain_bits": float(gain[idx]),
                "ppmd_entropy": float(entropy[idx]),
                "ppmd_top1": float(top1[idx]),
                "ppmd_margin": float(margin[idx]),
            }
        )
    write_csv(args.out_dir / "top_gain_examples.csv", sample_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
