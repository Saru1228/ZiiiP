#!/usr/bin/env python3
"""Generate supervised PCA coefficient labels.

This script is allowed to read Transformer probabilities. Its outputs are
training labels only; the evaluation script does not import this module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from nonoracle_common import EPS, VOCAB_SIZE, iter_slices, n_tokens_from_ppmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path)
    parser.add_argument("--slices", nargs="+", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--chunk-size", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    basis = np.load(args.basis).astype(np.float32)
    max_rank = max(args.ranks)
    if basis.shape[1] < max_rank:
        raise ValueError(f"basis has only {basis.shape[1]} columns, need {max_rank}")

    manifest = []
    for slice_name in args.slices:
        slice_dir = args.work_dir / slice_name
        n = n_tokens_from_ppmd(slice_dir / "ppmd.float16")
        p_path = slice_dir / "transformer.float16"
        if n_tokens_from_ppmd(p_path) != n:
            raise ValueError(f"{slice_name}: Transformer rows do not match PPMD rows")
        q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n, VOCAB_SIZE))
        p = np.memmap(p_path, dtype=np.float16, mode="r", shape=(n, VOCAB_SIZE))

        z_path = args.out_dir / f"{slice_name}_pca_z_rank{max_rank}.float32"
        tmp = z_path.with_suffix(z_path.suffix + ".tmp")
        with tmp.open("wb") as f:
            for start, end in iter_slices(n, args.chunk_size):
                logq = np.log(np.clip(np.asarray(q[start:end], dtype=np.float32), EPS, None))
                logp = np.log(np.clip(np.asarray(p[start:end], dtype=np.float32), EPS, None))
                residual = logp - logq
                residual -= residual.mean(axis=1, keepdims=True)
                z = residual @ basis[:, :max_rank]
                z.astype(np.float32).tofile(f)
        tmp.replace(z_path)
        item = {
            "slice": slice_name,
            "target": "pca",
            "max_rank": max_rank,
            "rows": n,
            "label_file": str(z_path),
            "label_file_bytes": z_path.stat().st_size,
            "uses_transformer_probability": True,
            "for_training_only": True,
        }
        manifest.append(item)
        print(json.dumps(item, sort_keys=True), flush=True)

    manifest_path = args.out_dir / "pca_target_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

