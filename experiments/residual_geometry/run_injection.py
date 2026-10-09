#!/usr/bin/env python3
"""Run cmix injection tests for generated probability files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path


FIELDNAMES = [
    "experiment",
    "representation",
    "rank",
    "alpha",
    "bucket_count",
    "top_k",
    "compressed_bytes",
    "delta_vs_ppmd",
    "delta_vs_transformer",
    "gap_recovered_percent",
    "objective",
    "oracle_flag",
    "nll",
    "explained_variance",
    "runtime_seconds",
    "prob_file",
    "comp_file",
    "log_file",
    "reported_loss_nats_per_token",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--cmix-dir", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--dictionary", default="dictionary/english.dic")
    parser.add_argument("--weights", default="models/6m-q4-fp32.tfwc2")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_existing(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            done.add(
                (
                    row.get("experiment", ""),
                    row.get("rank", ""),
                    row.get("alpha", ""),
                    row.get("bucket_count", ""),
                    row.get("top_k", ""),
                )
            )
    return done


def append_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def parse_loss(log_text: str) -> float | None:
    matches = re.findall(r"transformer loss:\s+([0-9.]+)\s+nats/token", log_text)
    return float(matches[-1]) if matches else None


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    transformer_bytes = int(baseline["transformer_bytes"])
    ppmd_bytes = int(baseline["ppmd_bytes"])
    gap_bytes = int(baseline["gap_bytes"])

    with args.manifest.open(newline="") as f:
        manifest_rows = list(csv.DictReader(f))

    done = read_existing(args.out_csv)
    ran = 0
    for item in manifest_rows:
        key = (
            item.get("experiment", ""),
            item.get("rank", ""),
            item.get("alpha", ""),
            item.get("bucket_count", ""),
            item.get("top_k", ""),
        )
        if key in done:
            continue
        if args.limit is not None and ran >= args.limit:
            break

        rank = item["rank"]
        alpha = item["alpha"]
        bucket_count = item.get("bucket_count", "")
        top_k = item.get("top_k", "")
        label_parts = [item.get("experiment", "experiment"), f"rank{rank}"]
        if bucket_count:
            label_parts.append(f"buckets{bucket_count}")
        if top_k:
            label_parts.append(f"top{top_k}")
        label_parts.append(f"alpha{alpha}")
        label = "_".join(label_parts).replace(".", "p")
        comp_file = args.work_dir / f"{label}.comp"
        log_file = args.work_dir / f"{label}.log"
        prob_file = Path(item["prob_file"])
        if not prob_file.is_absolute():
            prob_file = Path("/mnt/d/hunterprize") / prob_file

        cmd = [
            "./cmix",
            "-c",
            args.dictionary,
            str(args.input),
            str(comp_file),
            "--transformer",
            args.weights,
            "--load-transformer-probs",
            str(prob_file),
        ]
        start = time.monotonic()
        with log_file.open("wb") as log:
            subprocess.run(cmd, cwd=args.cmix_dir, stdout=log, stderr=subprocess.STDOUT, check=True)
        runtime = time.monotonic() - start
        compressed_bytes = comp_file.stat().st_size
        log_text = log_file.read_text(errors="replace")
        loss = parse_loss(log_text)

        row = {
            "experiment": item.get("experiment", "logratio_pca"),
            "representation": item.get("representation", "logratio"),
            "rank": rank,
            "alpha": alpha,
            "bucket_count": bucket_count,
            "top_k": top_k,
            "compressed_bytes": compressed_bytes,
            "delta_vs_ppmd": ppmd_bytes - compressed_bytes,
            "delta_vs_transformer": compressed_bytes - transformer_bytes,
            "gap_recovered_percent": (ppmd_bytes - compressed_bytes) / gap_bytes * 100.0,
            "objective": item.get("objective", ""),
            "oracle_flag": item.get("oracle_flag", ""),
            "nll": item.get("nll", ""),
            "explained_variance": item.get("explained_variance", ""),
            "runtime_seconds": f"{runtime:.2f}",
            "prob_file": str(prob_file),
            "comp_file": str(comp_file),
            "log_file": str(log_file),
            "reported_loss_nats_per_token": loss if loss is not None else "",
            "notes": "",
        }
        append_row(args.out_csv, row)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
        ran += 1


if __name__ == "__main__":
    main()
