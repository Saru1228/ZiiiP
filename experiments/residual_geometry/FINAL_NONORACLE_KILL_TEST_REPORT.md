# Final Non-Oracle Kill Test Report

Date: 2026-10-09

## Final Decision

**STOP**

The final non-oracle test asked whether a small legal predictor can recover at least 10% of the PPMD-to-Transformer gap without using the true next byte and without running Transformer at inference.

It did not.

## Leak Audit

**PASS**

Inference used only:

- fixed `slice_0` PCA basis
- trained predictor weights
- `ppmd.float16`
- PPMD-derived features

Inference did not load:

- true next-byte labels
- token arrays
- Transformer probabilities
- oracle latents
- future bytes

Audit file: `experiments/residual_geometry/LEAK_AUDIT.md`

## Dataset Split

| role | slice | offset |
|---|---|---:|
| train | slice_0 | 0 |
| train | slice_1 | 10,000,000 |
| validation | slice_2 | 6,000,000 |
| final test | slice_3 | 15,000,000 |
| final test | slice_4 | 25,000,000 |
| final test | slice_5 | 60,000,000 |

The suggested `20,000,000` offset was rejected before training because its preprocessed vocabulary had only 203 active token values, incompatible with the fixed 205-way pipeline.

## Models

Linear:

- Target: PCA coefficient
- Ranks: 8, 16, 32
- Ridge lambdas: `1e-4`, `1e-3`, `1e-2`
- Parameters: 1,704 / 3,408 / 6,816

Tiny MLP:

- Target: PCA coefficient
- Ranks: 8, 16, 32
- Learning rates: `1e-3`, `3e-4`
- Architecture: 212 -> 256 -> 128 -> rank
- Best validation model: rank32, lr `1e-3`
- Parameters: 91,552

All models are under the 250k parameter limit.

## Validation Results

Best latent-validation model:

| model | target | rank | validation MSE | validation cosine |
|---|---|---:|---:|---:|
| tiny_mlp | pca | 32 | 16.0794 | 0.9079 |

Actual validation compression on `slice_2`:

| slice | PPMD | Transformer | predictor | recovered |
|---|---:|---:|---:|---:|
| slice_2 | 190,291 | 189,118 | 190,352 | -5.20% |

Validation was already negative in real `cmix` bytes. No additional model tuning was performed.

## Final Held-Out Results

Frozen model:

- `tiny_mlp`
- target `pca`
- rank 32
- alpha 0.8
- 91,552 parameters

| model | rank | slice | predictor bytes | recovered |
|---|---:|---|---:|---:|
| tiny_mlp | 32 | slice_3 | 189,463 | -3.18% |
| tiny_mlp | 32 | slice_4 | 190,641 | -4.69% |
| tiny_mlp | 32 | slice_5 | 184,142 | -4.25% |

Artifact: `experiments/residual_geometry/output/nonoracle_final_test.csv`

## Aggregate

- Mean recovery: **-4.04%**
- Median recovery: **-4.25%**
- Minimum recovery: **-4.69%**
- Negative final-test slices: **3 / 3**

## Best Legal Model

No legal model reached positive held-out recovery.

Best evaluated final model:

- model: `tiny_mlp`
- rank: 32
- parameters: 91,552
- mean recovered: -4.04%
- minimum recovered: -4.69%

## Runtime / Model Cost

The selected model is small enough for the parameter-count rule, but it did not produce compression gain.

Recorded cost metadata:

- `experiments/residual_geometry/nonoracle_models/model_cost.json`

The generated local model files were removed during cleanup because the model is reproducible from the scripts and small result tables are retained.

## Shuffle Sanity Check

Not completed.

After all three final held-out slices were already negative, the user explicitly stopped further optional checks. This does not affect the STOP decision because the primary final-test condition had already failed.

## Interpretation

The project established that:

- log-ratio residual geometry is more useful than raw-probability geometry
- the residual basis generalizes across 1MB slices
- oracle coefficient choice can be extremely strong

But the final legal test did not support the stronger claim:

> A cheap non-oracle predictor can use legal context to recover a stable useful part of the Transformer coding advantage.

The practical bottleneck is coefficient prediction, not the existence of a low-rank space.

## Decision Rationale

The predefined STOP rules in `GPTreview/010.md` include:

- stop if mean final-test recovery is below 10%
- stop if any two final-test slices are `<= 0%`

Observed:

- mean final-test recovery: -4.04%
- three final-test slices `<= 0%`

Therefore the decision is:

```text
STOP
```

## Next Action

Archive the project.

Do not continue with:

- larger predictors
- LSTM/Transformer predictors
- more ranks
- more alpha sweeps
- 10MB/100MB scale-up

