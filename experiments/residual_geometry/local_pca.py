#!/usr/bin/env python3
"""Entropy-bucketed local log-ratio PCA probability generation."""

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
    parser.add_argument("--bucket-count", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--alphas", type=float, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def assign_buckets(values: np.ndarray, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(values, np.linspace(0, 1, bucket_count + 1))
    buckets = np.empty(values.shape[0], dtype=np.int16)
    for i in range(bucket_count):
        if i == bucket_count - 1:
            mask = (values >= edges[i]) & (values <= edges[i + 1])
        else:
            mask = (values >= edges[i]) & (values < edges[i + 1])
        buckets[mask] = i
    return buckets, edges


def nll_bits(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.log2(np.clip(probs[np.arange(len(targets)), targets], EPS, None)).sum())


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

    entropy = np.empty(n_tokens, dtype=np.float32)
    for start, end in iter_slices(n_tokens, args.chunk_size):
        pp = np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None)
        entropy[start:end] = -(pp * np.log2(pp)).sum(axis=1)
    bucket_ids, bucket_edges = assign_buckets(entropy, args.bucket_count)

    covs = [
        np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
        for _ in range(args.bucket_count)
    ]
    counts = np.zeros(args.bucket_count, dtype=np.int64)
    for start, end in iter_slices(n_tokens, args.chunk_size):
        logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
        logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
        residual = logp - logq
        residual -= residual.mean(axis=1, keepdims=True)
        local_buckets = bucket_ids[start:end]
        for b in range(args.bucket_count):
            mask = local_buckets == b
            if not np.any(mask):
                continue
            rb = residual[mask]
            covs[b] += rb.T.astype(np.float64) @ rb.astype(np.float64)
            counts[b] += int(mask.sum())

    bases = []
    cumulative_rows = []
    max_rank = max(args.ranks)
    for b, cov in enumerate(covs):
        cov /= counts[b]
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = np.maximum(eigvals[order], 0.0)
        eigvecs = eigvecs[:, order].astype(np.float32)
        explained = eigvals / eigvals.sum()
        cumulative = np.cumsum(explained)
        bases.append(eigvecs[:, :max_rank])
        for rank in args.ranks:
            cumulative_rows.append(
                {
                    "bucket": b,
                    "rank": rank,
                    "count": int(counts[b]),
                    "entropy_low": float(bucket_edges[b]),
                    "entropy_high": float(bucket_edges[b + 1]),
                    "explained_variance": float(cumulative[rank - 1]),
                }
            )

    rows = []
    targets = tokens[1:].astype(np.int64)
    for rank in args.ranks:
        for alpha in args.alphas:
            out_path = (
                args.out_dir
                / f"local_entropy{args.bucket_count}_rank{rank}_alpha{alpha_label(alpha)}.float16"
            )
            total_nll_bits = 0.0
            eval_count = 0
            with out_path.open("wb") as f:
                for start, end in iter_slices(n_tokens, args.chunk_size):
                    logq = np.log(
                        np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None)
                    )
                    logp = np.log(
                        np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None)
                    )
                    residual = logp - logq
                    residual -= residual.mean(axis=1, keepdims=True)
                    projected = np.empty_like(residual)
                    local_buckets = bucket_ids[start:end]
                    for b in range(args.bucket_count):
                        mask = local_buckets == b
                        if not np.any(mask):
                            continue
                        basis = bases[b][:, :rank]
                        rb = residual[mask]
                        projected[mask] = (rb @ basis) @ basis.T
                    probs = softmax_rows(logq + alpha * projected)

                    nll_end = min(end, n_tokens - 1)
                    if start < nll_end:
                        target = targets[start:nll_end]
                        total_nll_bits += nll_bits(probs[: nll_end - start], target)
                        eval_count += int(nll_end - start)
                    probs.astype(np.float16).tofile(f)

            mean_explained = float(
                np.mean(
                    [
                        row["explained_variance"]
                        for row in cumulative_rows
                        if row["rank"] == rank
                    ]
                )
            )
            rows.append(
                {
                    "experiment": "local_entropy_pca",
                    "representation": "logratio",
                    "rank": rank,
                    "alpha": alpha,
                    "bucket_count": args.bucket_count,
                    "top_k": "",
                    "prob_file": str(out_path),
                    "prob_file_bytes": out_path.stat().st_size,
                    "explained_variance": mean_explained,
                    "nll": total_nll_bits / eval_count,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    manifest = args.out_dir / "local_pca_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (args.out_dir / "local_pca_bucket_explained.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cumulative_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cumulative_rows)

    (args.out_dir / "local_pca_buckets.json").write_text(
        json.dumps(
            {
                "bucket_count": args.bucket_count,
                "entropy_edges": bucket_edges.tolist(),
                "counts": counts.tolist(),
                "eps": EPS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
