#!/usr/bin/env python3
"""Gate 1 independent-slice replication for the residual geometry kill test."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


VOCAB_SIZE = 205
EPS = 1e-9
SLICE_BYTES = 1_000_000
RANK = 64
ALPHA = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enwik9", required=True, type=Path)
    parser.add_argument("--cmix-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    parser.add_argument("--dictionary", default="dictionary/english.dic")
    parser.add_argument("--weights", default="models/6m-q4-fp32.tfwc2")
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def run_command(cmd: list[str], cwd: Path, log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log_path.open("wb") as log:
        subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    return time.monotonic() - start


def copy_slice(enwik9: Path, out_path: Path, offset: int) -> None:
    if out_path.exists() and out_path.stat().st_size == SLICE_BYTES:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with enwik9.open("rb") as src, out_path.open("wb") as dst:
        src.seek(offset)
        data = src.read(SLICE_BYTES)
        if len(data) != SLICE_BYTES:
            raise ValueError(f"offset {offset} produced only {len(data)} bytes")
        dst.write(data)


def file_ready(path: Path, size: int | None = None) -> bool:
    if not path.exists():
        return False
    return size is None or path.stat().st_size == size


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def generate_rank64_probs(tokens_path: Path, ppmd_path: Path, transformer_path: Path, out_path: Path, chunk_size: int) -> None:
    tokens = np.fromfile(tokens_path, dtype=np.uint8)
    n_tokens = int(tokens.shape[0])
    expected = n_tokens * VOCAB_SIZE * np.dtype(np.float16).itemsize
    if file_ready(out_path, expected):
        return

    q = np.memmap(ppmd_path, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(transformer_path, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))

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
    eigvecs = eigvecs[:, order].astype(np.float32)
    basis = eigvecs[:, :RANK]

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("wb") as f:
        for start, end in iter_slices(n_tokens, chunk_size):
            logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
            logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
            residual = logp - logq
            residual -= residual.mean(axis=1, keepdims=True)
            projected = (residual @ basis) @ basis.T
            probs = softmax_rows(logq + ALPHA * projected)
            probs.astype(np.float16).tofile(f)
    tmp_path.replace(out_path)


def read_existing(csv_path: Path) -> dict[str, dict[str, str]]:
    if not csv_path.exists():
        return {}
    with csv_path.open(newline="") as f:
        return {row["slice"]: row for row in csv.DictReader(f)}


def write_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "slice",
        "offset",
        "transformer_bytes",
        "ppmd_bytes",
        "gap_bytes",
        "rank64_bytes",
        "recovered_percent",
        "decision_notes",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_by_slice = read_existing(args.out_csv)

    for idx, offset in enumerate(args.offsets):
        name = f"slice_{idx}"
        if name in rows_by_slice:
            print(json.dumps(rows_by_slice[name], sort_keys=True), flush=True)
            continue

        work = args.out_dir / name
        work = work.resolve()
        work.mkdir(parents=True, exist_ok=True)
        input_path = work / "input.bin"
        copy_slice(args.enwik9, input_path, offset)

        tokens = work / "tokens.uint8"
        ppmd_probs = work / "ppmd.float16"
        transformer_probs = work / "transformer.float16"
        transformer_comp = work / "transformer.comp"
        transformer_log = work / "transformer.log"

        if not (transformer_comp.exists() and tokens.exists() and ppmd_probs.exists() and transformer_probs.exists()):
            run_command(
                [
                    "./cmix",
                    "-c",
                    args.dictionary,
                    str(input_path),
                    str(transformer_comp),
                    "--transformer",
                    args.weights,
                    "--save-ppmd-bytes",
                    str(tokens),
                    "--save-ppmd-probs",
                    str(ppmd_probs),
                    "--save-transformer-probs",
                    str(transformer_probs),
                ],
                args.cmix_dir,
                transformer_log,
            )

        expected_prob_size = tokens.stat().st_size * VOCAB_SIZE * np.dtype(np.float16).itemsize
        if ppmd_probs.stat().st_size != expected_prob_size or transformer_probs.stat().st_size != expected_prob_size:
            raise ValueError(f"{name}: probability file size mismatch")

        ppmd_comp = work / "ppmd_load.comp"
        if not ppmd_comp.exists():
            run_command(
                [
                    "./cmix",
                    "-c",
                    args.dictionary,
                    str(input_path),
                    str(ppmd_comp),
                    "--transformer",
                    args.weights,
                    "--load-transformer-probs",
                    str(ppmd_probs),
                ],
                args.cmix_dir,
                work / "ppmd_load.log",
            )

        rank_probs = work / "logratio_rank64_alpha0p8.float16"
        generate_rank64_probs(tokens, ppmd_probs, transformer_probs, rank_probs, args.chunk_size)

        rank_comp = work / "rank64_alpha0p8.comp"
        if not rank_comp.exists():
            run_command(
                [
                    "./cmix",
                    "-c",
                    args.dictionary,
                    str(input_path),
                    str(rank_comp),
                    "--transformer",
                    args.weights,
                    "--load-transformer-probs",
                    str(rank_probs),
                ],
                args.cmix_dir,
                work / "rank64_alpha0p8.log",
            )

        transformer_bytes = transformer_comp.stat().st_size
        ppmd_bytes = ppmd_comp.stat().st_size
        rank_bytes = rank_comp.stat().st_size
        gap = ppmd_bytes - transformer_bytes
        recovered = (ppmd_bytes - rank_bytes) / gap * 100.0 if gap else float("nan")
        row = {
            "slice": name,
            "offset": offset,
            "transformer_bytes": transformer_bytes,
            "ppmd_bytes": ppmd_bytes,
            "gap_bytes": gap,
            "rank64_bytes": rank_bytes,
            "recovered_percent": recovered,
            "decision_notes": "",
        }
        rows_by_slice[name] = {k: str(v) for k, v in row.items()}
        ordered_rows = [rows_by_slice[f"slice_{i}"] for i in range(len(args.offsets)) if f"slice_{i}" in rows_by_slice]
        write_rows(args.out_csv, ordered_rows)
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
