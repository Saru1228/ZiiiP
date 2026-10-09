# Phase 1

## Hypothesis

在继续任何新实验前，必须确认实验尺子没有偏差：完整 Transformer 路径和保存概率后回灌路径应给出相同压缩大小。

## Method

使用同一个 patched `cmix` 二进制，对 1MB `enwik9` 前缀比较：

- 直接运行 Transformer
- 加载已保存的 Transformer downstream probability

## Results

| method | bytes | vs PPMD | recovered |
|---|---:|---:|---:|
| PPMD downstream control | 176,338 | 0 | 0.0% |
| Transformer direct | 175,130 | 1,208 | 100.0% |
| Transformer roundtrip | 175,130 | 1,208 | 100.0% |

## Observation

roundtrip 已经和 direct Transformer 对齐。

## Interpretation

上一轮发现的 153 bytes 差异来自回灌路径没有继续运行 PPMD，导致后续 mixer 少了 PPMD bit-level 信号。patched 版本已修复这个实验管道偏差。

## Decision

CONTINUE.

## Next experiment

执行 `004.md` 指定的第一条具体实验：Log-ratio residual PCA。

# Phase 3

## Hypothesis

Transformer 相对于 PPMD 的有效修正可能不适合直接在 probability space 中做 PCA，而更适合在 log-ratio 空间中表示：

```text
r_i = log(pTransformer_i + eps) - log(pPPMD_i + eps)
```

如果这个假设成立，log-ratio PCA 应该比 raw probability PCA 追回更多 Transformer gap。

## Method

使用 1MB 样本，固定输入、alphabet、概率文件格式、量化方式和 patched `cmix`。

对每个位置计算：

```text
r = log(pT + 1e-9) - log(pP + 1e-9)
r = r - mean(r)
```

然后测试：

```text
rank = 1, 2, 4, 8, 16, 32, 64
alpha = 0.1, 0.2, 0.4, 0.6, 0.8, 1.0
```

概率重构：

```text
q_i ∝ pP_i * exp(alpha * r_hat_i)
```

每个 `rank × alpha` 都完整执行：

```text
probability generation
-> downstream probability injection
-> arithmetic compression
-> byte count
```

完整结果在：

```text
experiments/residual_geometry/output/logratio_pca_results.csv
```

## Results

最佳结果：

| method | bytes | vs PPMD | recovered |
|---|---:|---:|---:|
| Transformer | 175,130 | 1,208 | 100.0% |
| PPMD | 176,338 | 0 | 0.0% |
| raw PCA rank64 | 176,294 | 44 | 3.6% |
| log-ratio PCA rank32 alpha0.6 | 176,205 | 133 | 11.0% |
| log-ratio PCA rank64 alpha0.6 | 176,078 | 260 | 21.5% |
| log-ratio PCA rank64 alpha0.8 | 176,064 | 274 | 22.7% |
| log-ratio PCA rank64 alpha1.0 | 176,098 | 240 | 19.9% |

Top 12 rows by compressed bytes:

| rank | alpha | bytes | recovered |
|---:|---:|---:|---:|
| 64 | 0.8 | 176,064 | 22.7% |
| 64 | 0.6 | 176,078 | 21.5% |
| 64 | 1.0 | 176,098 | 19.9% |
| 64 | 0.4 | 176,109 | 19.0% |
| 64 | 0.2 | 176,178 | 13.2% |
| 32 | 0.6 | 176,205 | 11.0% |
| 32 | 0.8 | 176,210 | 10.6% |
| 32 | 0.4 | 176,221 | 9.7% |
| 64 | 0.1 | 176,231 | 8.9% |
| 32 | 1.0 | 176,246 | 7.6% |
| 32 | 0.2 | 176,252 | 7.1% |
| 16 | 0.8 | 176,258 | 6.6% |

## Observation

Log-ratio PCA clearly outperforms raw probability PCA on this 1MB slice.

Best raw PCA result recovered about 3.6% of the PPMD -> Transformer gap. Best log-ratio result recovered about 22.7%.

The best region is not at the smallest rank. Performance improves substantially from rank16 to rank64:

```text
rank16 best: 176,258 bytes, 6.6% recovered
rank32 best: 176,205 bytes, 11.0% recovered
rank64 best: 176,064 bytes, 22.7% recovered
```

Alpha is important. For high rank, `alpha=0.6` to `0.8` is best. For low rank, larger alpha often hurts compression.

Offline NLL is not a reliable replacement for final byte count in this setup. Some variants with worse direct NLL still produce better final compressed bytes after the downstream cmix mixer.

## Interpretation

This supports the probability-geometry hypothesis: the residual has more usable structure in log-ratio coordinates than in raw probability coordinates.

This does not prove a deployable codec yet. This is still an ORACLE structural probe: it uses the full Transformer probability file to fit PCA and produce residuals. It does not include the cost of transmitting or predicting latent variables.

The result is strong enough to continue structural probing, because it crosses the `004.md` 20% "明显正向" threshold.

## Decision

CONTINUE.

Do not expand to 10MB yet. The next step should follow `004.md`: first analyze where Transformer wins, then test whether local/context-conditioned low-rank beats global low-rank.

## Next experiment

1. Run per-position gain analysis:

```text
gain_t = log2(pT_t[y_t] / pP_t[y_t])
```

and compare gain against PPMD entropy / top1 probability.

2. Then run entropy-conditioned local PCA and compare:

```text
global rank32
vs
4 buckets × local rank8
```

This will test whether the residual is globally low-dimensional or context-conditioned / piecewise low-dimensional.

# Phase 4

## Hypothesis

Transformer's useful corrections may concentrate in difficult PPMD contexts. If so, per-position gain should correlate with PPMD uncertainty measures such as entropy, top1 probability, and top1-top2 margin.

## Method

For each evaluated position:

```text
gain_t = log2(pTransformer_t[y_t] / pPPMD_t[y_t])
```

Then compare gain against:

- PPMD entropy
- PPMD top1 probability
- PPMD top1-top2 margin

Outputs:

```text
experiments/residual_geometry/output/gain_summary.json
experiments/residual_geometry/output/gain_by_entropy_bucket.csv
experiments/residual_geometry/output/gain_by_top1_bucket.csv
experiments/residual_geometry/output/gain_by_margin_bucket.csv
```

## Results

Summary:

| metric | value |
|---|---:|
| positive gain fraction | 29.9% |
| negative gain fraction | 70.0% |
| mean gain | -7.14 bits |
| median gain | -1.42 bits |
| corr(gain, entropy) | 0.434 |
| corr(gain, top1) | -0.428 |
| corr(gain, margin) | -0.414 |

Entropy buckets:

| entropy bucket | count | mean gain | positive gain fraction |
|---:|---:|---:|---:|
| Q1 lowest entropy | 150,185 | -13.94 | 18.3% |
| Q2 | 150,185 | -9.84 | 28.4% |
| Q3 | 150,185 | -3.49 | 37.4% |
| Q4 highest entropy | 150,186 | -1.28 | 35.4% |

## Observation

Gain is less negative in high-entropy contexts. Transformer-like corrections are more relevant when PPMD is uncertain.

However, the saved downstream Transformer distribution by itself is not a final codec distribution. Direct target gain is mostly negative, while the full compressor still benefits through the downstream cmix mixer.

## Interpretation

Entropy is a plausible context feature for local structure, but direct gain should be treated as diagnostic only. Final byte count remains the metric.

## Decision

CONTINUE to entropy-conditioned local PCA.

## Next experiment

Compare global log-ratio PCA against entropy-bucketed local PCA.

# Phase 5

## Hypothesis

If residual geometry is context-conditioned, splitting by PPMD entropy and fitting a separate local PCA in each bucket should beat a global PCA at comparable parameter budget.

The key comparison from `004.md` is:

```text
global rank32
vs
4 buckets x local rank8
```

## Method

Create 4 equal-count buckets by PPMD entropy. In each bucket, fit log-ratio PCA separately.

Test:

```text
local rank = 4, 8, 16
alpha = 0.2, 0.4, 0.6, 0.8, 1.0
```

Outputs:

```text
experiments/residual_geometry/output/local_pca_results.csv
experiments/residual_geometry/probs/local_pca_manifest.csv
experiments/residual_geometry/probs/local_pca_bucket_explained.csv
```

## Results

Best local PCA rows:

| method | bytes | vs PPMD | recovered |
|---|---:|---:|---:|
| global rank32 alpha0.6 | 176,205 | 133 | 11.0% |
| local 4 x rank8 alpha0.6 | 176,281 | 57 | 4.7% |
| local 4 x rank16 alpha0.6 | 176,270 | 68 | 5.6% |
| local 4 x rank16 alpha0.4 | 176,271 | 67 | 5.5% |

## Observation

Entropy-conditioned local PCA underperforms global PCA.

The required comparison fails:

```text
global rank32:       176,205 bytes, 11.0% recovered
4 x local rank8:     176,281 bytes,  4.7% recovered
```

Even local rank16 does not catch up to global rank32.

## Interpretation

This does not support the entropy-conditioned piecewise-low-dimensional hypothesis, at least for this first entropy bucketing scheme.

It does not rule out all local structure, but it says PPMD entropy alone is not a useful gate here.

## Decision

MODIFY. Do not pursue entropy-local PCA further before testing other simpler structure probes.

## Next experiment

Run sparse / top-k correction.

# Phase 6

## Hypothesis

Transformer's useful correction may mainly adjust the top few symbols under PPMD. If so, allowing correction only on PPMD top-k symbols should recover much of the gain.

## Method

Use the best global log-ratio setting as the base residual:

```text
rank = 64
alpha = 0.8
```

Only apply the correction to PPMD top-k symbols:

```text
top-k = 4, 8, 16, 32, 64
```

All other symbols keep PPMD probabilities, followed by renormalization.

Outputs:

```text
experiments/residual_geometry/output/sparse_topk_results.csv
```

## Results

| top-k | bytes | vs PPMD | recovered |
|---:|---:|---:|---:|
| 4 | 176,571 | -233 | -19.3% |
| 8 | 176,516 | -178 | -14.7% |
| 16 | 176,428 | -90 | -7.5% |
| 32 | 176,327 | 11 | 0.9% |
| 64 | 176,234 | 104 | 8.6% |
| full 205, rank64 alpha0.8 | 176,064 | 274 | 22.7% |

## Observation

Small top-k correction is harmful. top64 is positive but still far below the full 205-symbol rank64 correction.

## Interpretation

The useful correction is not confined to the top few PPMD symbols. Tail or mid-probability mass movement appears important.

This weakens the sparse-top-k hypothesis for the first implementation.

## Decision

STOP this branch for now.

## Next experiment

The best current structural signal remains global log-ratio PCA rank64 alpha0.8.

Before 10MB, the next useful small experiment is either:

1. test independent 1MB slices for the same log-ratio rank64 signal, or
2. run a coding-aware oracle low-rank test to see whether rank64's remaining gap is due to PCA's L2 objective rather than representation capacity.

Given `004.md`, coding-aware low-rank is the next deeper mechanism test, but it should be explicitly labeled ORACLE.
