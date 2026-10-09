# Residual Geometry Kill Test Progress

Date: 2026-10-08

This file records the current state of `GPTreview/007.md`.

# Gate 1: Independent Slice Replication

## Status

PASS.

## Method

Four non-overlapping 1MB slices were tested with fixed parameters:

```text
representation = log-ratio
rank = 64
alpha = 0.8
```

No per-slice rank or alpha tuning was allowed.

## Results

| slice | offset | Transformer | PPMD | rank64 | recovered |
|---|---:|---:|---:|---:|---:|
| slice_0 | 0 | 175,130 | 176,338 | 176,064 | 22.68% |
| slice_1 | 10,000,000 | 184,842 | 185,956 | 185,702 | 22.80% |
| slice_2 | 6,000,000 | 189,118 | 190,291 | 190,044 | 21.06% |
| slice_3 | 15,000,000 | 188,166 | 189,423 | 189,121 | 24.03% |

Summary:

```text
mean recovered:   22.64%
median recovered: 22.74%
minimum:          21.06%
positive slices:  4 / 4
```

## Gate Decision

PASS.

The result is stable across independent slices and does not trigger any Gate 1 stop condition.

# Gate 2: Cross-Slice PCA Basis

## Status

PASS.

## Method

Fit the rank64 PCA basis only on:

```text
slice_0
```

Then apply the same basis to:

```text
slice_1
slice_2
slice_3
```

No per-test-slice PCA basis fitting was used for the cross-slice result.

## Results

| train | test | in-slice recovered | cross-slice recovered | retention |
|---|---|---:|---:|---:|
| slice_0 | slice_1 | 22.80% | 23.43% | 1.03 |
| slice_0 | slice_2 | 21.06% | 22.34% | 1.06 |
| slice_0 | slice_3 | 24.03% | 25.06% | 1.04 |

Summary:

```text
mean cross-slice recovered: 23.61%
mean retention ratio:       1.04
minimum cross recovered:    22.34%
minimum retention:          1.03
```

## Gate Decision

PASS.

The slice_0 basis generalizes to the other tested slices. This argues against the signal being only a file-local PCA artifact.

# Next Gate

Proceed to Gate 3:

```text
Mixer-Aware Low-Rank ORACLE
```

Gate 3 has not yet been executed. No final GO/STOP decision is allowed until Gate 3 is completed or fails.

# Artifacts

```text
experiments/residual_geometry/output/killtest_independent_slices.csv
experiments/residual_geometry/output/killtest_cross_slice.csv
experiments/residual_geometry/killtest_gate1.py
experiments/residual_geometry/killtest_gate2.py
```
