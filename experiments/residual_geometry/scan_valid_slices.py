#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enwik9", required=True, type=Path)
    parser.add_argument("--cmix-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--offsets", type=int, nargs="+", required=True)
    args = parser.parse_args()

    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for off in args.offsets:
        inp = args.out_dir / f"input_{off}.bin"
        tok = args.out_dir / f"tokens_{off}.uint8"
        comp = args.out_dir / f"dummy_{off}.comp"
        with args.enwik9.open("rb") as f:
            f.seek(off)
            data = f.read(1_000_000)
        inp.write_bytes(data)
        subprocess.run(
            [
                "./cmix",
                "-c",
                "dictionary/english.dic",
                str(inp),
                str(comp),
                "--bytes-only",
                "--save-ppmd-bytes",
                str(tok),
            ],
            cwd=args.cmix_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        a = np.fromfile(tok, dtype=np.uint8)
        print(f"{off},{len(a)},{len(set(a.tolist()))}", flush=True)


if __name__ == "__main__":
    main()
