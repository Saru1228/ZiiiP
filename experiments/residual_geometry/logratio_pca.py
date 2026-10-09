#!/usr/bin/env python3
"""Generate log-ratio PCA probability files for cmix injection tests."""

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


def nll_bits(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.log2(np.clip(probs[np.arange(len(targets)), targets], EPS, None)).sum())


def alpha_label(alpha: float) -> str:
    return f"{alpha:.2f}".replace(".", "p")


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

    rows = []
    max_rank = max(args.ranks)
    if max_rank > VOCAB_SIZE:
        raise ValueError(f"max rank {max_rank} > vocab size {VOCAB_SIZE}")

    for rank in args.ranks:
        basis = eigvecs[:, :rank]
        for alpha in args.alphas:
            out_path = args.out_dir / f"logratio_rank{rank}_alpha{alpha_label(alpha)}.float16"
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
                    projected = (residual @ basis) @ basis.T
                    probs = softmax_rows(logq + alpha * projected)

                    nll_end = min(end, n_tokens - 1)
                    if start < nll_end:
                        targets = tokens[start + 1 : nll_end + 1].astype(np.int64)
                        total_nll_bits += nll_bits(probs[: nll_end - start], targets)
                        eval_count += int(nll_end - start)

                    probs.astype(np.float16).tofile(f)

            rows.append(
                {
                    "experiment": "logratio_pca",
                    "representation": "logratio",
                    "rank": rank,
                    "alpha": alpha,
                    "prob_file": str(out_path),
                    "prob_file_bytes": out_path.stat().st_size,
                    "explained_variance": float(cumulative[rank - 1]),
                    "nll": total_nll_bits / eval_count,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)

    manifest = args.out_dir / "logratio_pca_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (args.out_dir / "logratio_pca_spectrum.json").write_text(
        json.dumps(
            {
                "eigenvalues": eigvals.tolist(),
                "explained_variance": explained.tolist(),
                "cumulative_explained_variance": cumulative.tolist(),
                "eps": EPS,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
