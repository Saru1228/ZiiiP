#!/usr/bin/env python3
"""Gate 2 cross-slice PCA basis generalization."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-9
RANK = 64
ALPHA = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate1-csv", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--cmix-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--dictionary", default="dictionary/english.dic")
    parser.add_argument("--weights", default="models/6m-q4-fp32.tfwc2")
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start, end in ((s, min(n, s + chunk_size)) for s in range(0, n, chunk_size)):
        yield start, end


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def fit_basis(slice_dir: Path, chunk_size: int) -> np.ndarray:
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
    return eigvecs[:, order[:RANK]].astype(np.float32)


def generate_cross_probs(slice_dir: Path, basis: np.ndarray, out_path: Path, chunk_size: int) -> None:
    tokens = np.fromfile(slice_dir / "tokens.uint8", dtype=np.uint8)
    n_tokens = int(tokens.shape[0])
    expected = n_tokens * VOCAB_SIZE * np.dtype(np.float16).itemsize
    if out_path.exists() and out_path.stat().st_size == expected:
        return
    q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(slice_dir / "transformer.float16", dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        for start, end in iter_slices(n_tokens, chunk_size):
            logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
            logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
            residual = logp - logq
            residual -= residual.mean(axis=1, keepdims=True)
            projected = (residual @ basis) @ basis.T
            probs = softmax_rows(logq + ALPHA * projected)
            probs.astype(np.float16).tofile(f)
    tmp.replace(out_path)


def run_cmix(cmix_dir: Path, input_path: Path, prob_path: Path, comp_path: Path, log_path: Path, dictionary: str, weights: str) -> float:
    if comp_path.exists():
        return 0.0
    start = time.monotonic()
    with log_path.open("wb") as log:
        subprocess.run(
            [
                "./cmix",
                "-c",
                dictionary,
                str(input_path),
                str(comp_path),
                "--transformer",
                weights,
                "--load-transformer-probs",
                str(prob_path),
            ],
            cwd=cmix_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - start


def main() -> None:
    args = parse_args()
    args.work_dir = args.work_dir.resolve()
    rows = list(csv.DictReader(args.gate1_csv.open(newline="")))
    by_slice = {row["slice"]: row for row in rows}

    train_slice = "slice_0"
    basis_path = args.work_dir / train_slice / "basis_rank64.npy"
    if basis_path.exists():
        basis = np.load(basis_path)
    else:
        basis = fit_basis(args.work_dir / train_slice, args.chunk_size)
        np.save(basis_path, basis)

    out_rows = []
    for row in rows:
        test_slice = row["slice"]
        if test_slice == train_slice:
            continue
        slice_dir = args.work_dir / test_slice
        cross_probs = slice_dir / "cross_from_slice0_rank64_alpha0p8.float16"
        generate_cross_probs(slice_dir, basis, cross_probs, args.chunk_size)
        cross_comp = slice_dir / "cross_from_slice0_rank64_alpha0p8.comp"
        run_cmix(
            args.cmix_dir,
            slice_dir / "input.bin",
            cross_probs,
            cross_comp,
            slice_dir / "cross_from_slice0_rank64_alpha0p8.log",
            args.dictionary,
            args.weights,
        )
        ppmd_bytes = int(row["ppmd_bytes"])
        gap = int(row["gap_bytes"])
        in_slice_bytes = int(row["rank64_bytes"])
        cross_slice_bytes = cross_comp.stat().st_size
        in_slice_recovered = (ppmd_bytes - in_slice_bytes) / gap * 100.0
        cross_slice_recovered = (ppmd_bytes - cross_slice_bytes) / gap * 100.0
        retention = cross_slice_recovered / in_slice_recovered if in_slice_recovered else float("nan")
        out = {
            "train_slice": train_slice,
            "test_slice": test_slice,
            "in_slice_bytes": in_slice_bytes,
            "cross_slice_bytes": cross_slice_bytes,
            "in_slice_recovered": in_slice_recovered,
            "cross_slice_recovered": cross_slice_recovered,
            "retention_ratio": retention,
        }
        out_rows.append(out)
        print(json.dumps(out, sort_keys=True), flush=True)

    fields = [
        "train_slice",
        "test_slice",
        "in_slice_bytes",
        "cross_slice_bytes",
        "in_slice_recovered",
        "cross_slice_recovered",
        "retention_ratio",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
