# Leak Audit

Date: 2026-10-09

Status: **PASS for the planned non-oracle inference pipeline**

## Dataset Split

The split is frozen before training:

| role | slice | offset | note |
|---|---|---:|---|
| train | slice_0 | 0 | existing valid slice |
| train | slice_1 | 10,000,000 | existing valid slice |
| validation | slice_2 | 6,000,000 | existing valid slice |
| final test | slice_3 | 15,000,000 | existing valid slice |
| final test | slice_4 | 25,000,000 | replacement for invalid 20,000,000 suggestion |
| final test | slice_5 | 60,000,000 | valid held-out replacement |

The originally suggested `20,000,000` offset was scanned before training and produced only 203 unique preprocessed token values, which is incompatible with the current fixed 205-way probability pipeline. It was replaced before model training or validation. The final test slices remain frozen after this audit.

All slice intervals are 1,000,000 raw bytes and do not overlap.

## Required Questions

1. Does predictor inference access `y_t`?

   **NO.** The inference script does not load `tokens.uint8`, labels, or true next-byte arrays.

2. Does predictor inference access Transformer probability?

   **NO.** `evaluate_nonoracle_predictor.py` loads only the trained model, fixed basis, and `ppmd.float16`.

3. Does predictor inference access future byte information?

   **NO.** The first feature builder uses only the PPMD probability row and scalar statistics derived from that row. It does not use byte context at all.

4. Is PCA basis fit on test slices?

   **NO.** The basis is loaded from `experiments/residual_geometry/killtest_work/slice_0/basis_rank64.npy`.

5. Is predictor trained on test slices?

   **NO.** Training uses `slice_0` and `slice_1`; validation uses `slice_2`; final test uses `slice_3`, `slice_4`, and `slice_5`.

6. Are test slices used for hyperparameter selection?

   **NO.** Hyperparameters are selected before final test, using training loss and validation latent metrics only. Final test is one-shot after model selection.

## Code Isolation

Training label generation:

- `generate_training_targets.py`
- Allowed to load Transformer probabilities.
- Produces PCA coefficient labels for training/validation only.

Training:

- `train_nonoracle_predictor.py`
- Loads PPMD features and precomputed labels.
- Does not run compression evaluation.

Inference/evaluation:

- `evaluate_nonoracle_predictor.py`
- Does not import the target-generation script.
- Does not load Transformer probabilities.
- Does not load labels.
- Does not load token arrays.
- Generates external probability files from PPMD-only features and runs `cmix`.

## Current Feature Set

Allowed inference features:

- centered `log(pPPMD + eps)`
- entropy
- top1 probability
- top2 probability
- top1-top2 margin
- top4 mass
- top8 mass
- top16 mass

Forbidden features are not used:

- Transformer probability/logits/hidden states
- true next byte
- current target byte
- future bytes
- oracle latent from the current position
- future PPMD or cmix state

## Audit Decision

**PASS**

The non-oracle final test may proceed.

