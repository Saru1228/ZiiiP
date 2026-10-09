#!/usr/bin/env python3
"""Analyze low-rank Transformer-vs-PPMD log-probability residuals.

This script is intentionally small and file-format oriented: it takes the
binary dumps produced by fx2-cmix and computes the first diagnostic needed for
the residual-low-rank hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VOCAB_SIZE = 205
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True, type=Path)
    parser.add_argument("--ppmd", required=True, type=Path)
    parser.add_argument("--transformer", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cmix-log", type=Path)
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
    )
    return parser.parse_args()


def logsumexp_rows(x: np.ndarray) -> np.ndarray:
    m = np.max(x, axis=1)
    return m + np.log(np.exp(x - m[:, None]).sum(axis=1))


def iter_slices(n: int, chunk_size: int):
    for start in range(0, n, chunk_size):
        yield start, min(n, start + chunk_size)


def parse_cmix_log(path: Path | None) -> dict[str, float | int] | None:
    if path is None or not path.exists():
        return None
    text = path.read_text(errors="replace")
    loss_matches = re.findall(
        r"transformer loss:\s+([0-9.]+)\s+nats/token over\s+([0-9]+)\s+tokens",
        text,
    )
    size_matches = re.findall(r"([0-9]+)\s+bytes ->\s+([0-9]+)\s+bytes", text)
    out: dict[str, float | int] = {}
    if loss_matches:
        nats, tokens = loss_matches[-1]
        out["cmix_reported_downstream_loss_nats_per_token"] = float(nats)
        out["cmix_reported_downstream_loss_bits_per_token"] = float(nats) / math.log(2.0)
        out["cmix_reported_loss_tokens"] = int(tokens)
    if size_matches:
        raw, compressed = size_matches[-1]
        out["raw_bytes"] = int(raw)
        out["compressed_bytes"] = int(compressed)
        out["compressed_bits_per_raw_byte"] = int(compressed) * 8.0 / int(raw)
    return out or None


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tokens = np.fromfile(args.tokens, dtype=np.uint8)
    n_tokens = int(tokens.shape[0])

    ppmd_bytes = args.ppmd.stat().st_size
    transformer_bytes = args.transformer.stat().st_size
    expected_bytes = n_tokens * VOCAB_SIZE * np.dtype(np.float16).itemsize
    if ppmd_bytes != expected_bytes:
        raise ValueError(f"PPMD file has {ppmd_bytes} bytes; expected {expected_bytes}")
    if transformer_bytes != expected_bytes:
        raise ValueError(
            f"Transformer file has {transformer_bytes} bytes; expected {expected_bytes}"
        )

    q = np.memmap(args.ppmd, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE))
    p = np.memmap(
        args.transformer, dtype=np.float16, mode="r", shape=(n_tokens, VOCAB_SIZE)
    )

    # Row i predicts token i + 1. The last row has no target in this standalone
    # prefix dataset, so evaluate rows [0, n_tokens - 1).
    n_eval = n_tokens - 1
    targets = tokens[1:].astype(np.int64)

    cov = np.zeros((VOCAB_SIZE, VOCAB_SIZE), dtype=np.float64)
    residual_ss = 0.0
    ppmd_loss_bits = 0.0
    transformer_loss_bits = 0.0

    for start, end in iter_slices(n_eval, args.chunk_size):
        target = targets[start:end]
        q_chunk = np.asarray(q[start:end], dtype=np.float32)
        p_chunk = np.asarray(p[start:end], dtype=np.float32)

        ppmd_loss_bits += float(
            -np.log2(np.clip(q_chunk[np.arange(end - start), target], EPS, None)).sum()
        )
        transformer_loss_bits += float(
            -np.log2(np.clip(p_chunk[np.arange(end - start), target], EPS, None)).sum()
        )

        logq = np.log(np.clip(q_chunk, EPS, None))
        logp = np.log(np.clip(p_chunk, EPS, None))
        residual = logp - logq
        residual -= residual.mean(axis=1, keepdims=True)

        cov += residual.T.astype(np.float64) @ residual.astype(np.float64)
        residual_ss += float(np.square(residual, dtype=np.float64).sum())

    cov /= n_eval
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)
    eigvecs = eigvecs[:, order]

    singular_values = np.sqrt(eigvals * n_eval)
    explained = eigvals / eigvals.sum()
    cumulative = np.cumsum(explained)

    ranks = [r for r in args.ranks if 1 <= r <= VOCAB_SIZE]
    rank_rows = []
    for rank in ranks:
        basis = eigvecs[:, :rank].astype(np.float32)
        loss_bits = 0.0
        for start, end in iter_slices(n_eval, args.chunk_size):
            target = targets[start:end]
            q_chunk = np.asarray(q[start:end], dtype=np.float32)
            p_chunk = np.asarray(p[start:end], dtype=np.float32)
            logq = np.log(np.clip(q_chunk, EPS, None))
            logp = np.log(np.clip(p_chunk, EPS, None))
            residual = logp - logq
            residual -= residual.mean(axis=1, keepdims=True)
            projected = (residual @ basis) @ basis.T
            logits = logq + projected
            loss_bits += float(
                ((logsumexp_rows(logits) - logits[np.arange(end - start), target]) / math.log(2.0)).sum()
            )

        bpb = loss_bits / n_eval
        rank_rows.append(
            {
                "rank": rank,
                "bpb": bpb,
                "cumulative_explained_variance": float(cumulative[rank - 1]),
            }
        )

    ppmd_bpb = ppmd_loss_bits / n_eval
    saved_transformer_bpb = transformer_loss_bits / n_eval
    cmix_log = parse_cmix_log(args.cmix_log)

    spectrum_csv = args.out_dir / "spectrum.csv"
    with spectrum_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "component",
                "eigenvalue",
                "singular_value",
                "explained_variance",
                "cumulative_explained_variance",
            ],
        )
        writer.writeheader()
        for i in range(VOCAB_SIZE):
            writer.writerow(
                {
                    "component": i + 1,
                    "eigenvalue": float(eigvals[i]),
                    "singular_value": float(singular_values[i]),
                    "explained_variance": float(explained[i]),
                    "cumulative_explained_variance": float(cumulative[i]),
                }
            )

    ranks_csv = args.out_dir / "rank_bpb.csv"
    with ranks_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "bpb", "cumulative_explained_variance"]
        )
        writer.writeheader()
        writer.writerows(rank_rows)

    metrics = {
        "tokens": n_tokens,
        "evaluated_predictions": n_eval,
        "vocab_size": VOCAB_SIZE,
        "ppmd_bpb": ppmd_bpb,
        "saved_downstream_transformer_bpb": saved_transformer_bpb,
        "saved_downstream_transformer_delta_vs_ppmd_bpb": (
            ppmd_bpb - saved_transformer_bpb
        ),
        "cmix_log": cmix_log,
        "ranks": rank_rows,
        "input_files": {
            "tokens": str(args.tokens),
            "ppmd": str(args.ppmd),
            "transformer": str(args.transformer),
        },
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )

    xs = np.arange(1, VOCAB_SIZE + 1)
    plt.figure(figsize=(8, 5))
    plt.semilogy(xs, singular_values, marker=".", linewidth=1)
    plt.xlabel("component")
    plt.ylabel("singular value")
    plt.title("Residual Singular Spectrum")
    plt.tight_layout()
    plt.savefig(args.out_dir / "singular_spectrum.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(xs, cumulative, linewidth=1.5)
    plt.xlabel("rank")
    plt.ylabel("cumulative explained variance")
    plt.ylim(0, 1.01)
    plt.title("Cumulative Explained Variance")
    plt.tight_layout()
    plt.savefig(args.out_dir / "cumulative_explained_variance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.axhline(ppmd_bpb, linestyle="--", color="tab:gray", label="PPMD")
    plt.axhline(
        saved_transformer_bpb,
        linestyle="--",
        color="tab:green",
        label="Saved downstream TF",
    )
    plt.plot(
        [row["rank"] for row in rank_rows],
        [row["bpb"] for row in rank_rows],
        marker="o",
        label="PCA residual rank",
    )
    plt.xscale("log", base=2)
    plt.xlabel("rank")
    plt.ylabel("bits per byte/token")
    plt.title("Rank Approximation Bpb")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "rank_bpb.png", dpi=160)
    plt.close()

    top_rows = "\n".join(
        f"| {row['rank']} | {row['cumulative_explained_variance']:.4f} | {row['bpb']:.6f} |"
        for row in rank_rows
    )
    report = f"""# Mini Residual SVD Probe

## Summary

This is a pipeline smoke test on a 1 MB prefix of `enwik9`, compressed through
the dictionary `-c` path with the repository's quantized Transformer weights.
It is not yet a representative Hutter Prize result, but it verifies that the
local WSL/C++ probability-dump path can produce aligned PPMD and Transformer
probability matrices and that the residual-rank analysis can run end to end.

## Inputs

- Raw prefix: 1,000,000 bytes from `data/enwik9`
- Preprocessed/token stream: {n_tokens:,} tokens
- Evaluated next-token predictions: {n_eval:,}
- Vocabulary size: {VOCAB_SIZE}
- PPMD probabilities: `{args.ppmd}`
- Saved downstream Transformer probabilities: `{args.transformer}`

## Main Metrics

- PPMD-only: `{ppmd_bpb:.6f}` bpb
- Saved downstream Transformer stream: `{saved_transformer_bpb:.6f}` bpb
- Saved downstream Transformer delta vs PPMD: `{ppmd_bpb - saved_transformer_bpb:.6f}` bpb

The saved Transformer stream is not the full final arithmetic-coder
distribution. It is the float16 distribution passed downstream in place of the
old LSTM output. In a normal `-c` run, later mixers can still change the final
bit predictions, so this number should be treated as a diagnostic for the
dumped rows, not as the final compressor's bpb.

## Rank Diagnostics

| rank | cumulative explained variance | reconstructed bpb |
|---:|---:|---:|
{top_rows}

## Decision

The local route is technically viable: C++ dumping, probability alignment,
covariance/SVD, rank reconstruction, and bpb evaluation all completed.

Scientific interpretation is still bounded. This sample is small and uses the
generic `-c dictionary` path on an enwik9 prefix rather than the exact full
`-e enwik9` article-reordered pipeline. Treat the rank curve as a screen for
whether the experiment machinery works, not as evidence about the final Hutter
Prize trade-off.

## Next Branch

The next useful gate is a 1M-token exact-style sample with article boundaries,
ideally by adding a normal early-stop option to the C++ runner so it can stop
cleanly after a target number of preprocessed enwik9 tokens while flushing all
three dump files.
"""
    (args.out_dir / "report.md").write_text(report)

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
