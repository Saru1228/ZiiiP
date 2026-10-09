#!/usr/bin/env python3
"""Sparse top-k log-ratio PCA corrections."""

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
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--top-k", type=int, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


def normalize_rows(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs, EPS, None)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


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

    cov = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    for start, end in iter_slices(n_tokens, args.chunk_size):
        logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
        logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
        residual = logp - logq
        residual -= residual.mean(axis=1, keepdims=True)
        cov += residual.T.astype(np.float64) @ residual.astype(np.float64)
    cov /= n_tokens
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order].astype(np.float32)
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)
    basis = eigvecs[:, : args.rank]

    rows = []
    targets = tokens[1:].astype(np.int64)
    for top_k in args.top_k:
        out_path = (
            args.out_dir
            / f"sparse_top{top_k}_rank{args.rank}_alpha{alpha_label(args.alpha)}.float16"
        )
        total_nll_bits = 0.0
        eval_count = 0
        with out_path.open("wb") as f:
            for start, end in iter_slices(n_tokens, args.chunk_size):
                pp = np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None)
                logq = np.log(pp)
                logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
                residual = logp - logq
                residual -= residual.mean(axis=1, keepdims=True)
                projected = (residual @ basis) @ basis.T

                probs = pp.copy()
                top_indices = np.argpartition(pp, -top_k, axis=1)[:, -top_k:]
                row_idx = np.arange(end - start)[:, None]
                probs[row_idx, top_indices] *= np.exp(args.alpha * projected[row_idx, top_indices])
                probs = normalize_rows(probs)

                nll_end = min(end, n_tokens - 1)
                if start < nll_end:
                    target = targets[start:nll_end]
                    total_nll_bits += nll_bits(probs[: nll_end - start], target)
                    eval_count += int(nll_end - start)
                probs.astype(np.float16).tofile(f)

        rows.append(
            {
                "experiment": "sparse_topk",
                "representation": "logratio",
                "rank": args.rank,
                "alpha": args.alpha,
                "bucket_count": "",
                "top_k": top_k,
                "prob_file": str(out_path),
                "prob_file_bytes": out_path.stat().st_size,
                "explained_variance": float(cumulative[args.rank - 1]),
                "nll": total_nll_bits / eval_count,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    manifest = args.out_dir / "sparse_topk_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
