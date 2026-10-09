#!/usr/bin/env python3
"""Write low-rank residual probability files for cmix round-trip tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppmd", required=True, type=Path)
    parser.add_argument("--transformer", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
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


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ppmd_bytes = args.ppmd.stat().st_size
    transformer_bytes = args.transformer.stat().st_size
    if ppmd_bytes != transformer_bytes:
        raise ValueError("PPMD and Transformer files have different sizes")
    row_bytes = VOCAB_SIZE * np.dtype(np.float16).itemsize
    if ppmd_bytes % row_bytes != 0:
        raise ValueError("probability file size is not divisible by one row")
    n_rows = ppmd_bytes // row_bytes

    q = np.memmap(args.ppmd, dtype=np.float16, mode="r", shape=(n_rows, VOCAB_SIZE))
    p = np.memmap(
        args.transformer, dtype=np.float16, mode="r", shape=(n_rows, VOCAB_SIZE)
    )

    cov = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    for start, end in iter_slices(n_rows, args.chunk_size):
        logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
        logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
        residual = logp - logq
        residual -= residual.mean(axis=1, keepdims=True)
        cov += residual.T.astype(np.float64) @ residual.astype(np.float64)

    cov /= n_rows
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order].astype(np.float32)
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)

    manifest = {
        "rows": int(n_rows),
        "vocab_size": VOCAB_SIZE,
        "outputs": [],
    }
    for rank in args.ranks:
        if rank < 1 or rank > VOCAB_SIZE:
            raise ValueError(f"invalid rank {rank}")
        basis = eigvecs[:, :rank]
        out_path = args.out_dir / f"rank{rank}.float16"
        with out_path.open("wb") as f:
            for start, end in iter_slices(n_rows, args.chunk_size):
                logq = np.log(
                    np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None)
                )
                logp = np.log(
                    np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None)
                )
                residual = logp - logq
                residual -= residual.mean(axis=1, keepdims=True)
                projected = (residual @ basis) @ basis.T
                probs = softmax_rows(logq + projected).astype(np.float16)
                probs.tofile(f)
        manifest["outputs"].append(
            {
                "rank": rank,
                "path": str(out_path),
                "bytes": out_path.stat().st_size,
                "cumulative_explained_variance": float(cumulative[rank - 1]),
            }
        )

    (args.out_dir / "rank_prob_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
