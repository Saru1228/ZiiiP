#!/usr/bin/env python3
"""Evaluate trained non-oracle predictors with legal inference inputs only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

from nonoracle_common import (
    ALPHA,
    VOCAB_SIZE,
    build_ppmd_features,
    iter_slices,
    n_tokens_from_ppmd,
    save_json,
    softmax_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--gate-csv", required=True, type=Path)
    parser.add_argument("--slices", nargs="+", required=True)
    parser.add_argument("--cmix-dir", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--prob-dir", required=True, type=Path)
    parser.add_argument("--cost-json", required=True, type=Path)
    parser.add_argument("--dictionary", default="dictionary/english.dic")
    parser.add_argument("--weights", default="models/6m-q4-fp32.tfwc2")
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--shuffle-slice")
    return parser.parse_args()


def parse_loss(log_text: str) -> float | None:
    matches = re.findall(r"transformer loss:\s+([0-9.]+)\s+nats/token", log_text)
    return float(matches[-1]) if matches else None


def predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    x = (x - model["feature_mean"]) / model["feature_std"]
    model_type = str(model["model_type"])
    if model_type == "linear":
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        return x_aug @ model["weights"]
    if model_type == "tiny_mlp":
        h1 = np.maximum(x @ model["w1"] + model["b1"], 0.0)
        h2 = np.maximum(h1 @ model["w2"] + model["b2"], 0.0)
        return h2 @ model["w3"] + model["b3"]
    raise ValueError(f"unknown model type {model_type}")


def generate_probs(model: dict[str, np.ndarray], basis: np.ndarray, slice_dir: Path, out_path: Path, chunk_size: int, shuffle: bool) -> float:
    n = n_tokens_from_ppmd(slice_dir / "ppmd.float16")
    q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n, VOCAB_SIZE))
    rank = int(model["rank"])
    basis = basis[:, :rank]
    expected = n * VOCAB_SIZE * np.dtype(np.float16).itemsize
    if out_path.exists() and out_path.stat().st_size == expected:
        return 0.0
    rng = np.random.default_rng(42)
    start_time = time.monotonic()
    if shuffle:
        z_all = np.empty((n, rank), dtype=np.float32)
        cursor = 0
        for start, end in iter_slices(n, chunk_size):
            z = predict(model, build_ppmd_features(q[start:end]).astype(np.float32))
            z_all[cursor : cursor + len(z)] = z
            cursor += len(z)
        rng.shuffle(z_all, axis=0)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        for start, end in iter_slices(n, chunk_size):
            q_chunk = np.asarray(q[start:end], dtype=np.float32)
            logq = np.log(np.clip(q_chunk, 1e-9, None))
            if shuffle:
                z = z_all[start:end]
            else:
                z = predict(model, build_ppmd_features(q_chunk).astype(np.float32))
            correction = z @ basis.T
            probs = softmax_rows(logq + ALPHA * correction)
            probs.astype(np.float16).tofile(f)
    tmp.replace(out_path)
    return time.monotonic() - start_time


def run_cmix(args: argparse.Namespace, slice_dir: Path, prob_path: Path, comp_path: Path, log_path: Path) -> float:
    if comp_path.exists() and comp_path.stat().st_size > 0:
        return 0.0
    start = time.monotonic()
    with log_path.open("wb") as log:
        subprocess.run(
            [
                "./cmix",
                "-c",
                args.dictionary,
                str(slice_dir / "input.bin"),
                str(comp_path),
                "--transformer",
                args.weights,
                "--load-transformer-probs",
                str(prob_path),
            ],
            cwd=args.cmix_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return time.monotonic() - start


def main() -> None:
    args = parse_args()
    args.model_file = args.model_file.resolve()
    args.basis = args.basis.resolve()
    args.work_dir = args.work_dir.resolve()
    args.gate_csv = args.gate_csv.resolve()
    args.cmix_dir = args.cmix_dir.resolve()
    args.out_csv = args.out_csv.resolve()
    args.prob_dir = args.prob_dir.resolve()
    args.cost_json = args.cost_json.resolve()
    args.prob_dir.mkdir(parents=True, exist_ok=True)
    model = dict(np.load(args.model_file, allow_pickle=False))
    model["model_type"] = str(model["model_type"])
    model["target"] = str(model["target"])
    model["rank"] = int(model["rank"])
    basis = np.load(args.basis).astype(np.float32)
    gate_rows = {row["slice"]: row for row in csv.DictReader(args.gate_csv.open(newline=""))}
    model_name = f"{model['model_type']}_{model['target']}_rank{model['rank']}"
    if args.shuffle_slice:
        model_name += "_shuffle"
    rows = []
    total_inference = 0.0
    for slice_name in args.slices:
        slice_dir = args.work_dir / slice_name
        prob_path = args.prob_dir / f"{model_name}_{slice_name}.float16"
        do_shuffle = args.shuffle_slice == slice_name
        total_inference += generate_probs(model, basis, slice_dir, prob_path, args.chunk_size, do_shuffle)
        comp_path = slice_dir / f"{model_name}.comp"
        log_path = slice_dir / f"{model_name}.log"
        runtime = run_cmix(args, slice_dir, prob_path, comp_path, log_path)
        ppmd = int(gate_rows[slice_name]["ppmd_bytes"])
        transformer = int(gate_rows[slice_name]["transformer_bytes"])
        gap = ppmd - transformer
        pred_bytes = comp_path.stat().st_size
        log_text = log_path.read_text(errors="replace")
        row = {
            "model": str(model["model_type"]),
            "target": str(model["target"]),
            "rank": model["rank"],
            "slice": slice_name,
            "ppmd_bytes": ppmd,
            "transformer_bytes": transformer,
            "predictor_bytes": pred_bytes,
            "gap_bytes": gap,
            "saved_vs_ppmd": ppmd - pred_bytes,
            "recovered_percent": (ppmd - pred_bytes) / gap * 100.0 if gap else float("nan"),
            "parameter_count": sum(v.size for k, v in model.items() if isinstance(v, np.ndarray) and k not in {"feature_mean", "feature_std"}),
            "alpha": ALPHA,
            "prob_file": str(prob_path),
            "comp_file": str(comp_path),
            "cmix_runtime_seconds": runtime,
            "inference_generation_seconds": total_inference,
            "reported_loss_nats_per_token": parse_loss(log_text) or "",
            "shuffle_sanity": "yes" if do_shuffle else "no",
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = args.out_csv.exists()
    with args.out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)

    costs = {}
    if args.cost_json.exists():
        costs = json.loads(args.cost_json.read_text())
    key = str(args.model_file)
    costs.setdefault(key, {})
    costs[key]["inference_time"] = total_inference
    save_json(args.cost_json, costs)


if __name__ == "__main__":
    main()
