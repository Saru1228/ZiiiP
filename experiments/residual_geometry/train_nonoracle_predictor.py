#!/usr/bin/env python3
"""Train legal-context non-oracle coefficient predictors."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from nonoracle_common import (
    FEATURE_DIM,
    VOCAB_SIZE,
    build_ppmd_features,
    feature_stats,
    iter_slices,
    n_tokens_from_ppmd,
    save_json,
)


SEED = 42
LINEAR_LAMBDAS = [1e-4, 1e-3, 1e-2]
MLP_LRS = [1e-3, 3e-4]
MLP_HIDDEN1 = 256
MLP_HIDDEN2 = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--label-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-slices", nargs="+", required=True)
    parser.add_argument("--val-slice", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--chunk-size", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser.parse_args()


def load_xy(slice_dir: Path, label_path: Path, rank: int, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
    n = n_tokens_from_ppmd(slice_dir / "ppmd.float16")
    q = np.memmap(slice_dir / "ppmd.float16", dtype=np.float16, mode="r", shape=(n, VOCAB_SIZE))
    labels = np.memmap(label_path, dtype=np.float32, mode="r", shape=(n, 32))
    xs = []
    ys = []
    for start, end in iter_slices(n, chunk_size):
        xs.append(build_ppmd_features(q[start:end]).astype(np.float32))
        ys.append(np.asarray(labels[start:end, :rank], dtype=np.float32))
    return np.vstack(xs), np.vstack(ys)


def load_split(args: argparse.Namespace, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_xs = []
    train_ys = []
    for name in args.train_slices:
        x, y = load_xy(args.work_dir / name, args.label_dir / f"{name}_pca_z_rank32.float32", rank, args.chunk_size)
        train_xs.append(x)
        train_ys.append(y)
    x_train = np.vstack(train_xs)
    y_train = np.vstack(train_ys)
    x_val, y_val = load_xy(
        args.work_dir / args.val_slice,
        args.label_dir / f"{args.val_slice}_pca_z_rank32.float32",
        rank,
        args.chunk_size,
    )
    train_dirs = [args.work_dir / s for s in args.train_slices]
    mean, std = feature_stats(train_dirs, args.chunk_size)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return x_train, y_train, x_val, y_val, mean, std


def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((y_hat - y) ** 2))


def cosine(y_hat: np.ndarray, y: np.ndarray) -> float:
    num = np.sum(y_hat * y, axis=1)
    den = np.linalg.norm(y_hat, axis=1) * np.linalg.norm(y, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-12)))


def train_linear(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    xtx = x_aug.T @ x_aug
    reg = np.eye(xtx.shape[0], dtype=np.float32) * lam
    reg[-1, -1] = 0.0
    xty = x_aug.T @ y
    return np.linalg.solve(xtx + reg, xty).astype(np.float32)


def linear_predict(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    return x_aug @ weights


def init_mlp(rank: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    def w(fan_in: int, fan_out: int) -> np.ndarray:
        return (rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)).astype(np.float32)

    return {
        "w1": w(FEATURE_DIM, MLP_HIDDEN1),
        "b1": np.zeros(MLP_HIDDEN1, dtype=np.float32),
        "w2": w(MLP_HIDDEN1, MLP_HIDDEN2),
        "b2": np.zeros(MLP_HIDDEN2, dtype=np.float32),
        "w3": w(MLP_HIDDEN2, rank),
        "b3": np.zeros(rank, dtype=np.float32),
    }


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def mlp_predict(params: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    h1 = relu(x @ params["w1"] + params["b1"])
    h2 = relu(h1 @ params["w2"] + params["b2"])
    return h2 @ params["w3"] + params["b3"]


def train_mlp(x: np.ndarray, y: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, rank: int, lr: float, epochs: int, batch_size: int) -> tuple[dict[str, np.ndarray], float]:
    rng = np.random.default_rng(SEED + rank)
    params = init_mlp(rank, rng)
    best_params = {k: v.copy() for k, v in params.items()}
    best_mse = float("inf")
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    step = 0
    order = np.arange(x.shape[0])
    for _ in range(epochs):
        rng.shuffle(order)
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = y[idx]
            h1_pre = xb @ params["w1"] + params["b1"]
            h1 = relu(h1_pre)
            h2_pre = h1 @ params["w2"] + params["b2"]
            h2 = relu(h2_pre)
            pred = h2 @ params["w3"] + params["b3"]
            grad = (2.0 / np.prod(yb.shape)) * (pred - yb)
            grads = {}
            grads["w3"] = h2.T @ grad
            grads["b3"] = grad.sum(axis=0)
            dh2 = grad @ params["w3"].T
            dh2[h2_pre <= 0] = 0
            grads["w2"] = h1.T @ dh2
            grads["b2"] = dh2.sum(axis=0)
            dh1 = dh2 @ params["w2"].T
            dh1[h1_pre <= 0] = 0
            grads["w1"] = xb.T @ dh1
            grads["b1"] = dh1.sum(axis=0)
            step += 1
            for key in params:
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] * grads[key])
                mh = m[key] / (1 - beta1**step)
                vh = v[key] / (1 - beta2**step)
                params[key] -= lr * mh / (np.sqrt(vh) + eps)
        val_mse = mse(mlp_predict(params, x_val), y_val)
        if val_mse < best_mse:
            best_mse = val_mse
            best_params = {k: v.copy() for k, v in params.items()}
    return best_params, best_mse


def count_linear(rank: int) -> int:
    return (FEATURE_DIM + 1) * rank


def count_mlp(rank: int) -> int:
    return (
        FEATURE_DIM * MLP_HIDDEN1
        + MLP_HIDDEN1
        + MLP_HIDDEN1 * MLP_HIDDEN2
        + MLP_HIDDEN2
        + MLP_HIDDEN2 * rank
        + rank
    )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "nonoracle_validation_latent.csv"
    fieldnames = [
        "model",
        "target",
        "rank",
        "hyperparameter",
        "validation_mse",
        "validation_cosine",
        "parameter_count",
        "training_time",
        "model_file",
        "resumed_from_existing",
    ]
    if csv_path.exists():
        csv_path.unlink()

    def append_validation_row(row: dict[str, object]) -> None:
        exists = csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    costs = {}
    for rank in args.ranks:
        x_train, y_train, x_val, y_val, mean, std = load_split(args, rank)
        for lam in LINEAR_LAMBDAS:
            model_path = args.out_dir / f"linear_pca_rank{rank}_lambda{lam:g}.npz"
            if model_path.exists():
                saved = np.load(model_path, allow_pickle=False)
                weights = saved["weights"]
                training_time = 0.0
                resumed = True
            else:
                start = time.monotonic()
                weights = train_linear(x_train, y_train, lam)
                training_time = time.monotonic() - start
                np.savez(model_path, model_type="linear", target="pca", rank=rank, weights=weights, feature_mean=mean, feature_std=std)
                resumed = False
            pred_val = linear_predict(weights, x_val)
            row = {
                "model": "linear",
                "target": "pca",
                "rank": rank,
                "hyperparameter": f"lambda={lam:g}",
                "validation_mse": mse(pred_val, y_val),
                "validation_cosine": cosine(pred_val, y_val),
                "parameter_count": count_linear(rank),
                "training_time": training_time,
                "model_file": str(model_path),
                "resumed_from_existing": resumed,
            }
            costs[str(model_path)] = {
                "parameter_count": count_linear(rank),
                "fp32_weight_bytes": count_linear(rank) * 4,
                "int8_weight_bytes_estimate": count_linear(rank),
                "training_time": training_time,
                "inference_time": None,
            }
            append_validation_row(row)
            save_json(args.out_dir / "model_cost.json", costs)
            print(json.dumps(row, sort_keys=True), flush=True)

        for lr in MLP_LRS:
            model_path = args.out_dir / f"tiny_mlp_pca_rank{rank}_lr{lr:g}.npz"
            if model_path.exists():
                saved = np.load(model_path, allow_pickle=False)
                params = {k: saved[k] for k in ["w1", "b1", "w2", "b2", "w3", "b3"]}
                training_time = 0.0
                resumed = True
            else:
                start = time.monotonic()
                params, _ = train_mlp(x_train, y_train, x_val, y_val, rank, lr, args.epochs, args.batch_size)
                training_time = time.monotonic() - start
                np.savez(
                    model_path,
                    model_type="tiny_mlp",
                    target="pca",
                    rank=rank,
                    feature_mean=mean,
                    feature_std=std,
                    **params,
                )
                resumed = False
            pred_val = mlp_predict(params, x_val)
            row = {
                "model": "tiny_mlp",
                "target": "pca",
                "rank": rank,
                "hyperparameter": f"lr={lr:g}",
                "validation_mse": mse(pred_val, y_val),
                "validation_cosine": cosine(pred_val, y_val),
                "parameter_count": count_mlp(rank),
                "training_time": training_time,
                "model_file": str(model_path),
                "resumed_from_existing": resumed,
            }
            costs[str(model_path)] = {
                "parameter_count": count_mlp(rank),
                "fp32_weight_bytes": count_mlp(rank) * 4,
                "int8_weight_bytes_estimate": count_mlp(rank),
                "training_time": training_time,
                "inference_time": None,
            }
            append_validation_row(row)
            save_json(args.out_dir / "model_cost.json", costs)
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
