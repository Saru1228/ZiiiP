#!/usr/bin/env python3
"""Gate 3 coding-aware low-rank ORACLE probability generation.

This is explicitly not deployable. It uses the true next token to optimize a
per-position latent vector in a fixed low-rank basis. The latent norm is capped
by the norm of the ordinary PCA coefficient for that row, so the oracle can
rotate the correction toward coding loss but cannot use unbounded scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-9
ALPHA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--ranks", type=int, nargs="+", required=True)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1.0)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def fit_full_basis(slice_dir: Path, max_rank: int, chunk_size: int) -> np.ndarray:
    tokens = np.fromfile(slice_dir / "tokens.uint8", dtype=np.uint8)
    n_tokens = int(tokens.shape[0])
    q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(slice_dir / "transformer.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    cov = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    for start, end in iter_slices(n_tokens, chunk_size):
        logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
        logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
        residual = logp - logq
        residual -= residual.mean(axis=1, keepdims=True)
        cov += residual.T.astype(np.float64) @ residual.astype(np.float64)
    cov /= n_tokens
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    return eigvecs[:, order[:max_rank]].astype(np.float32)


def project_to_caps(z: np.ndarray, caps: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(z, axis=1)
    scale = np.ones_like(norms, dtype=np.float32)
    mask = norms > caps
    scale[mask] = caps[mask] / np.maximum(norms[mask], 1e-12)
    return z * scale[:, None]


def nll_bits(probs: np.ndarray, targets: np.ndarray) -> float:
    return float(-np.log2(np.clip(probs[np.arange(len(targets)), targets], EPS, None)).sum())


def main() -> None:
    args = parse_args()
    args.slice_dir = args.slice_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.fromfile(args.slice_dir / "tokens.uint8", dtype=np.uint8)
    n_tokens = int(tokens.shape[0])
    targets = tokens[1:].astype(np.int64)
    q = np.memmap(args.slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(args.slice_dir / "transformer.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))

    basis_path = args.slice_dir / f"basis_rank{max(args.ranks)}.npy"
    if basis_path.exists():
        full_basis = np.load(basis_path).astype(np.float32)
        if full_basis.shape[1] < max(args.ranks):
            full_basis = fit_full_basis(args.slice_dir, max(args.ranks), args.chunk_size)
            np.save(basis_path, full_basis)
    else:
        full_basis = fit_full_basis(args.slice_dir, max(args.ranks), args.chunk_size)
        np.save(basis_path, full_basis)

    rows = []
    for rank in args.ranks:
        basis = full_basis[:, :rank]
        out_path = args.out_dir / f"oracle_coding_rank{rank}.float16"
        total_nll = 0.0
        eval_count = 0
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            for start, end in iter_slices(n_tokens, args.chunk_size):
                logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
                logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
                residual = logp - logq
                residual -= residual.mean(axis=1, keepdims=True)
                z0 = residual @ basis
                caps = np.linalg.norm(z0, axis=1).astype(np.float32)
                z = z0.astype(np.float32).copy()

                nll_end = min(end, n_tokens - 1)
                local_targets = None
                if start < nll_end:
                    local_targets = targets[start:nll_end]

                # Optimize only rows that have a next-token target. The final
                # row has no target and keeps the ordinary PCA coefficient.
                opt_n = 0 if local_targets is None else len(local_targets)
                for _ in range(args.steps):
                    if opt_n == 0:
                        break
                    logits = logq[:opt_n] + ALPHA * (z[:opt_n] @ basis.T)
                    probs = softmax_rows(logits)
                    expected = probs @ basis
                    target_basis = basis[local_targets]
                    grad = expected - target_basis
                    z[:opt_n] -= args.lr * grad
                    z[:opt_n] = project_to_caps(z[:opt_n], caps[:opt_n])

                probs = softmax_rows(logq + ALPHA * (z @ basis.T))
                if opt_n:
                    total_nll += nll_bits(probs[:opt_n], local_targets)
                    eval_count += opt_n
                probs.astype(np.float16).tofile(f)
        tmp_path.replace(out_path)
        row = {
            "experiment": "mixer_aware_lowrank_oracle",
            "representation": "logratio",
            "rank": rank,
            "alpha": ALPHA,
            "bucket_count": "",
            "top_k": "",
            "oracle_flag": "ORACLE_TRUE_TARGET_LATENT_NORM_CAPPED",
            "objective": "target_nll_projected_gradient_norm_capped_actual_bytes_eval",
            "prob_file": str(out_path),
            "prob_file_bytes": out_path.stat().st_size,
            "nll": total_nll / eval_count,
            "steps": args.steps,
            "lr": args.lr,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    manifest = args.out_dir / "killtest_mixer_aware_oracle_manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
