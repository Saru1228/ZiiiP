# Mini Residual SVD Probe

## Summary

This is a pipeline smoke test on a 1 MB prefix of `enwik9`, compressed through
the dictionary `-c` path with the repository's quantized Transformer weights.
It is not yet a representative Hutter Prize result, but it verifies that the
local WSL/C++ probability-dump path can produce aligned PPMD and Transformer
probability matrices and that the residual-rank analysis can run end to end.

## Inputs

- Raw prefix: 1,000,000 bytes from `data/enwik9`
- Preprocessed/token stream: 600,742 tokens
- Evaluated next-token predictions: 600,741
- Vocabulary size: 205
- PPMD probabilities: `/tmp/fx2-head-c/ppmd.float16`
- Saved downstream Transformer probabilities: `/tmp/fx2-head-c/transformer.float16`

## Main Metrics

- PPMD-only: `2.979086` bpb
- Saved downstream Transformer stream: `11.984075` bpb
- Saved downstream Transformer delta vs PPMD: `-9.004989` bpb

The saved Transformer stream is not the full final arithmetic-coder
distribution. It is the float16 distribution passed downstream in place of the
old LSTM output. In a normal `-c` run, later mixers can still change the final
bit predictions, so this number should be treated as a diagnostic for the
dumped rows, not as the final compressor's bpb.

## Rank Diagnostics

| rank | cumulative explained variance | reconstructed bpb |
|---:|---:|---:|
| 1 | 0.3993 | 9.771562 |
| 2 | 0.5152 | 12.932505 |
| 4 | 0.6045 | 11.785316 |
| 8 | 0.6510 | 11.989718 |
| 16 | 0.7007 | 12.141648 |
| 32 | 0.7634 | 12.390256 |
| 64 | 0.8439 | 12.447159 |
| 128 | 0.9399 | 12.322327 |

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
