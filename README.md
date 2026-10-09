# ZiiiP Hutter Prize Experiments

This repository syncs the lightweight project state for the Hutter Prize / PPMD
plus Transformer residual experiments.

Large local files are intentionally not tracked:

- `enwik9`
- generated `data/` files such as `ppmd-probs.float16`
- Transformer probability dumps
- checkpoints and build outputs
- the cloned upstream `fx2-cmix-transformer-v1/` source tree

On a new machine:

```bash
git clone https://github.com/Saru1228/ZiiiP.git hunterprize
cd hunterprize
git clone https://github.com/astOwOlfo/fx2-cmix-transformer-v1.git
```

Then place or download `enwik9` locally in the project root or in the upstream
repo's expected `data/` path, depending on the experiment script being run.

The main research notes are in `GPTreview/`.

## Reproduce The Recorded Results

The repository keeps the reproducible control plane: scripts, reports, patches,
and small CSV/JSON result tables. It intentionally does not keep raw datasets,
generated probability matrices, model checkpoints from upstream, or compiled
binaries.

Required local inputs:

- `enwik9` in the repository root.
- A fresh clone of `fx2-cmix-transformer-v1/`.
- The upstream model file used by the experiments, especially
  `models/6m-q4-fp32.tfwc2`.

Apply the local round-trip patch before rerunning injection tests:

```bash
cd fx2-cmix-transformer-v1
git apply ../patches/fx2-roundtrip-ppmd-fix.patch
make CFLAGS_DEFINES='-DSEED=923 -DUPDATE_LIMIT=3000'
cp cmix run/cmix
cd ..
```

Key summaries:

- `GPTreview/009.md`: readable oracle kill-test summary.
- `experiments/residual_geometry/FINAL_KILL_TEST_REPORT.md`: oracle upper-bound
  report.
- `experiments/residual_geometry/FINAL_NONORACLE_KILL_TEST_REPORT.md`: final
  legal non-oracle STOP report.
- `experiments/residual_geometry/output/`: compact result tables.

Final project state:

```text
Oracle upper-bound test: STRONG GO for the research direction only.
Legal non-oracle predictor test: STOP.
```
