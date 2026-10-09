# Final Kill Test Report

Date: 2026-10-09

## Final Decision

**STRONG GO for the research direction, not for immediate deployment.**

The kill test asked whether the low-rank residual idea still looks worth pursuing after three gates:

1. Does the signal repeat across independent 1MB slices?
2. Does a basis learned on one slice still help other slices?
3. Under a very favorable oracle setting, can low rank recover enough coding value to justify the next stage?

All three gates passed. Gate 3 passed by a very large margin.

Important boundary: Gate 3 is an **oracle upper-bound** test. It uses the true next token to choose the low-rank correction direction at each position. That means it is not a deployable compressor. It only answers: "Is there enough useful information inside this low-rank space to make the direction worth continuing?"

The answer is yes.

## Baseline

The canonical test slice is `slice_0`, offset `0`, using the patched `cmix` binary that preserves the PPMD mixer path when loading external Transformer probabilities.

| baseline | compressed bytes |
|---|---:|
| Transformer | 175,130 |
| PPMD | 176,338 |
| PPMD - Transformer gap | 1,208 |

Artifacts:

- `experiments/residual_geometry/output/baseline.json`
- `patches/fx2-roundtrip-ppmd-fix.patch`

## Gate 1: Independent Slices

Question: Does fixed log-ratio PCA rank64 alpha0.8 recover a consistent part of the PPMD-to-Transformer gap on several independent slices?

Gate rule from `GPTreview/007.md`:

- Stop if mean recovered `< 15%`
- Stop if more than one slice is `<= 5%`
- Stop if results are systematically negative
- Pass if mean `>= 15%`, majority `>= 10%`, and no systematic negatives

Result:

| slice | offset | PPMD | Transformer | rank64 | recovered |
|---|---:|---:|---:|---:|---:|
| slice_0 | 0 | 176,338 | 175,130 | 176,064 | 22.68% |
| slice_1 | 10,000,000 | 185,956 | 184,842 | 185,702 | 22.80% |
| slice_2 | 6,000,000 | 190,291 | 189,118 | 190,044 | 21.06% |
| slice_3 | 15,000,000 | 189,423 | 188,166 | 189,121 | 24.03% |

Summary:

- Mean recovered: **22.64%**
- Median recovered: **22.74%**
- Minimum recovered: **21.06%**
- Negative slices: **0**

Decision: **PASS**

Artifact: `experiments/residual_geometry/output/killtest_independent_slices.csv`

## Gate 2: Cross-Slice Basis

Question: If the basis is learned only on `slice_0`, does it still work on other slices?

Gate rule from `GPTreview/007.md`:

- Stop if mean cross-slice recovered `< 10%`
- Stop if retention `< 0.60`
- Pass if mean cross recovered `>= 10%` and retention `>= 0.60`

Result:

| train | test | in-slice recovered | cross-slice recovered | retention |
|---|---|---:|---:|---:|
| slice_0 | slice_1 | 22.80% | 23.43% | 1.03 |
| slice_0 | slice_2 | 21.06% | 22.34% | 1.06 |
| slice_0 | slice_3 | 24.03% | 25.06% | 1.04 |

Summary:

- Mean cross-slice recovered: **23.61%**
- Mean retention: **1.04**
- Minimum cross-slice recovered: **22.34%**
- Minimum retention: **1.03**

Decision: **PASS**

Artifact: `experiments/residual_geometry/output/killtest_cross_slice.csv`

## Gate 3: Mixer-Aware Low-Rank Oracle

Question: If we give the low-rank space a very favorable oracle objective, can it recover enough coding value to justify learning a real non-oracle predictor later?

Method:

- Basis: fixed log-ratio basis from `slice_0`
- Ranks tested: 8, 16, 32, 64
- Objective: `target_nll_projected_gradient_norm_capped_actual_bytes_eval`
- Oracle flag: `ORACLE_TRUE_TARGET_LATENT_NORM_CAPPED`
- Alpha: 1.0
- Evaluation: actual `cmix` compressed bytes, not only probability loss

The latent correction is target-aware, but its norm is capped by the ordinary PCA coefficient norm for that row. This keeps the experiment from using unlimited scale, while still making it an oracle upper-bound.

Result:

| rank | compressed bytes | recovered vs PPMD | recovered vs original gap | reported loss |
|---:|---:|---:|---:|---:|
| 8 | 158,356 | 17,982 bytes | 1,488.58% | 3.751778 |
| 16 | 118,852 | 57,486 bytes | 4,758.77% | 2.965958 |
| 32 | 51,214 | 125,124 bytes | 10,357.95% | 1.722169 |
| 64 | 6,103 | 170,235 bytes | 14,092.30% | 0.500949 |

Gate rule from `GPTreview/007.md`:

- Stop if rank64 recovered `< 30%`
- Stop if rank32 `< 20%` and rank64 `< 40%`
- Minimum GO if rank32 `>= 40%` or rank64 `>= 50%`
- Strong GO if rank8 `>= 20%`, rank16 `>= 30%`, rank32 `>= 50%`
- If ambiguous, stop

Decision: **STRONG GO**

Artifacts:

- `experiments/residual_geometry/output/killtest_mixer_aware_oracle.csv`
- `experiments/residual_geometry/probs/killtest_mixer_aware_oracle_manifest.csv`
- `experiments/residual_geometry/killtest_gate3_oracle.py`
- `experiments/residual_geometry/run_injection.py`

## Interpretation

The earlier non-oracle PCA experiment recovered only about 22-23% of the small PPMD-to-Transformer gap. That was enough to show a stable residual geometry, but not enough to claim a practical compressor.

The oracle experiment shows a different thing: inside the same kind of low-rank space, there is much more coding value than the ordinary PCA projection extracts. In plain terms, the space is not empty. The missing part is not the basis itself; the missing part is choosing the right correction coefficients without looking at the answer.

This routes the project away from "try larger PCA on bigger samples" and toward "learn or approximate the oracle coefficient choice from legal context features."

## Boundary

This does **not** mean the project has a working Hutter Prize improvement yet.

The Gate 3 result is deliberately allowed to see the true next token. A real compressor cannot do that. Therefore the result should be read as:

- Strong evidence that low-rank residual directions can carry coding value.
- Strong evidence that mixer-aware objectives matter.
- No evidence yet that a non-oracle model can predict those directions cheaply enough.

## Next Branch

Recommended next experiment:

**Train a non-oracle coefficient predictor.**

The target is the oracle latent vector `z` or a cheaper approximation to it. The predictor may use only legal information available before the next token is known, such as:

- PPMD probability shape
- Transformer probability shape
- entropy, margin, top-k mass
- token/context features already available to the compressor
- previous latent coefficients or local state

The next gate should be stricter:

- It must not use the true next token.
- It must be tested cross-slice.
- It should first target rank8/rank16, because Gate 3 shows even low ranks have large upper-bound value.
- A useful first pass would be recovering at least 10-20% of the original PPMD-to-Transformer gap without oracle leakage.

## Final Summary

The project should continue, but the direction changes.

Do not spend the next round mainly on 10MB/100MB scale-up. The current evidence already says the geometry generalizes across small slices. The bottleneck is coefficient prediction.

Best next question:

**Can we predict a useful part of the oracle low-rank correction without seeing the next byte?**

