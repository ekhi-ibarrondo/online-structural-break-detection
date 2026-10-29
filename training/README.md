# Training

This folder holds the recipes used to train every artifact of the final
committee: ten members whose per-step percentiles are fused in log-odds space
with fixed weights (written by `committee/build_committee.py`).  They were
cleaned up from the training code of the submission and were not re-run end to
end for this release; section 5 says what can be reproduced exactly.  To check
that the scripts run, the four CPU members and the feature extraction were run
on 50 training series (`--limit 50`), and `networks.train_full` with `--smoke`.  Nothing
here is imported at inference time.

All commands run from this folder as modules (`python -m <package>.<script>`).
Paths come from command-line options or from the environment variables
`DATA_DIR` (competition training data) and `OUT_DIR` (all outputs).

## 1. Data

The training data belong to the ADIA Lab Structural Break challenge on
CrunchDAO. They are **not** included here and may not be redistributed; each
user downloads them with the CrunchDAO command-line tool:

```sh
pip install crunch-cli --upgrade
# the setup token is shown on the competition's submission page
crunch setup structural-break-real-time <your-model-name> --token <TOKEN>
cd <your-model-name>
crunch download        # only if data/ is still empty
export DATA_DIR="$PWD/data"
```

(`crunch --help` shows the exact syntax of the installed version.)  The recipes
read `X_train.parquet`, `y_train.parquet` and `y_train_index.parquet`
(10,000 series).

## 2. Environment

* Python 3.12, `pip install -r requirements.txt` (numpy 2.5.2, numba 0.67.0,
  llvmlite 0.49.0, lightgbm 4.7.0, pandas, pyarrow).  The numba kernels refuse
  to run with other numba/llvmlite versions.
* reads_two_rebuilds and the three sequence networks also need PyTorch with CUDA
  (`requirements-gpu.txt`).
* reads_two_rebuilds and reads_three_others read part of their per-point features (36 summary
  features, 22 detector traces, 4 context values) from the inference package
  `breakdetect` (`pip install -e .` at the repository root), so training and
  inference share one implementation; see `common/inference_features.py`.
* BLAS is pinned to one thread by the scripts.

## 3. Members, scripts and artifacts

| Member | Script(s) | Artifacts | Hardware | Approx. time |
|---|---|---|---|---|
| surprises | `python -m surprises.train all` | `surprises/model_full_surprises_s{0,1,2}/`, `surprises/maps_surprises_train10k_s{0,1,2}.npz` | CPU | not recorded |
| pattern_removed | `python -m pattern_removed.train all` | `pattern_removed/model_full_pattern_removed_s{0,1,2}/`, `pattern_removed/maps_pattern_removed_train10k_s{0,1,2}.npz` | CPU | not recorded |
| order_not_size | `python -m order_not_size.train all` | `order_not_size/model_full_order_not_size_s0/`, `order_not_size/maps_order_not_size_train10k_s{0,1,2}.npz` | CPU | features ~5 min with 4 workers |
| slow_and_fast | `python -m slow_and_fast.train all` | `slow_and_fast/model_full_slow_and_fast_s{0,1,2}/`, `slow_and_fast/maps_slow_and_fast_train10k_s{0,1,2}.npz` | CPU | not recorded |
| reads_two_rebuilds | `python -m features.extract`, then `python -m reads_two_rebuilds.train fit-full`, `fit-oof`, `score-oof`, `map`, `head` | `reads_two_rebuilds/model_full_reads_two_rebuilds_s0/` (3 learners + `head.json`), `reads_two_rebuilds/maps_reads_two_rebuilds_train10k_s0.npz` | GPU for `fit-*`, CPU otherwise | `fit-full` ~7-10 min, `fit-oof` ~25-35 min (estimates) |
| float_grid | `python -m float_grid.build_map` | `float_grid/model_full_float_grid_s0/float_grid.json`, `float_grid/maps_float_grid_train10k_s0.npz` | CPU | not recorded |
| reads_three_others | `python -m features.extract`, then `python -m reads_three_others.train` | `reads_three_others/model_reads_three_others/`, `reads_three_others/map_reads_three_others.npz` | CPU (2 threads) | not recorded |
| broken_or_not | `python -m common.holdout`, `networks.train_full --variant pointwise`, `networks.export_weights`, `networks.calibrate` | `broken_or_not/member.npz`, `broken_or_not/fallback_raw.npy`, `broken_or_not/member_config.json`, `broken_or_not/broken_or_not_map.npz` | GPU | ~0.65 h per fit on a T4 (estimate) |
| which_one_broke | `networks.train_full --variant richer`, `networks.finetune_pairwise`, `networks.export_weights`, `networks.calibrate` | `which_one_broke/member.npz`, `which_one_broke/fallback_raw.npy`, `which_one_broke/member_config.json`, `which_one_broke/which_one_broke_map.npz` | GPU (NVIDIA L4) | first fit ~63 min, fine-tune ~10-15 min (estimate) |
| blind_to_size | `networks.train_full --variant richer_scale_blind`, `networks.export_weights`, `networks.calibrate` | `blind_to_size/member.npz`, `blind_to_size/fallback_raw.npy`, `blind_to_size/member_config.json`, `blind_to_size/blind_to_size_map.npz` | GPU (NVIDIA L4) | not recorded |
| committee | `python -m committee.build_committee` | `committee/committee.json`, `committee/SHA256SUMS` + copies of all of the above | CPU | seconds |

Paths in the "Artifacts" column are relative to `$OUT_DIR`.

### What each member learns

* **surprises** - AR innovations of the series against its own history, window
  statistics standardised by the historical null, CUSUM statistics; one
  pooled LightGBM (150 trees) with pair-count weights.
* **pattern_removed** - multi-window level/scale/dependence statistics and 18 detector
  columns, the detector columns PIT-transformed per (step, history length)
  cell; LightGBM, 128 trees in 4 stages.
* **order_not_size** - 18 rank/PIT-native detector columns computed with portable numba
  kernels, conditional PIT and a closed-form weighted ridge (no random seed).
* **slow_and_fast** - spectral (STFT, Haar, cepstrum) statistics replayed against the
  series' own historical null; LightGBM, 200 trees.
* **reads_two_rebuilds** - a 2-layer GRU over 354 streaming features (the
  surprises and pattern_removed inputs, 36 summary features, 22 detector traces,
  4 context values); 3 learners averaged in logit.
* **float_grid** - no supervised training: membership likelihood of each
  online value in a discrete grid of float values fitted on the history.
* **reads_three_others** - one LightGBM over the 468 streaming features (the
  reads_two_rebuilds inputs plus the slow_and_fast and order_not_size inputs) at
  every 16th step.
* **broken_or_not / which_one_broke / blind_to_size** - a causal GRU sequence network pretrained on
  synthetic breaks spliced into the series' own history, then fine-tuned on the
  labels (`networks/sequence_net.py`).  which_one_broke and blind_to_size use a
  richer pretext (`networks/variants.py`); which_one_broke replaces the fine-tune
  loss with a pairwise ranking loss; blind_to_size feeds the network a
  scale-normalised series.  The three
  are trained on the 8,000 series outside a fixed internal holdout of 2,000
  series (`common/holdout.py`).

Every member's raw score is mapped to (0, 1) by a per-step CDF fitted on
out-of-fold or training scores (`common/step_maps.py`, `reads_three_others/anchor_maps.py`,
`networks/calibrate.py`).

## 4. Full run

```sh
cd training
export DATA_DIR=/path/to/data OUT_DIR=/path/to/outputs

# members trained on the 10,000 series (CPU)
python -m surprises.train all
python -m pattern_removed.train all
python -m order_not_size.train all
python -m slow_and_fast.train all
python -m float_grid.build_map --out-dir "$OUT_DIR/float_grid"

# streaming features (~9.4 GB), reads_two_rebuilds and reads_three_others
python -m features.extract --out-dir "$OUT_DIR/features" --workers 8
F="$OUT_DIR/features"; O="$OUT_DIR/reads_two_rebuilds"
python -m reads_two_rebuilds.train fit-full  --features-dir "$F" --out-dir "$O"
python -m reads_two_rebuilds.train fit-oof   --features-dir "$F" --out-dir "$O"
python -m reads_two_rebuilds.train score-oof --features-dir "$F" --out-dir "$O" --workers 8
python -m reads_two_rebuilds.train map       --features-dir "$F" --out-dir "$O"
python -m reads_two_rebuilds.train head      --out-dir "$O"
python -m reads_three_others.train --features-dir "$F" --out-dir "$OUT_DIR/reads_three_others"

# sequence networks (GPU), trained on the 8,000 search series
python -m common.holdout --out-dir "$OUT_DIR/search_split"
export SEARCH_DATA_DIR="$OUT_DIR/search_split"
python -m networks.train_full --variant pointwise --out "$OUT_DIR/broken_or_not"
python -m networks.train_full --variant richer --out "$OUT_DIR/which_one_broke_base"
python -m networks.finetune_pairwise --source "$OUT_DIR/which_one_broke_base/full" --out "$OUT_DIR/which_one_broke"
python -m networks.train_full --variant richer_scale_blind --out "$OUT_DIR/blind_to_size"
for m in broken_or_not which_one_broke blind_to_size; do
  python -m networks.export_weights "$OUT_DIR/$m/full/net_ft.pt" "$OUT_DIR/$m/member.npz"
  python -m networks.calibrate --preds "$OUT_DIR/$m/full/preds_full.npz" --data "$SEARCH_DATA_DIR" --out "$OUT_DIR/$m"
done

# committee folder + committee.json
python -m committee.build_committee --committee-dir "$OUT_DIR/committee"
```

The CPU members accept `--limit N` (first N series) for a quick smoke test,
`networks.train_full` accepts `--smoke` (CPU, 120 series, 1-2 epochs).

## 5. Reproducibility

* **surprises, pattern_removed, slow_and_fast, order_not_size** - deterministic
  (single-threaded LightGBM with fixed seeds; order_not_size has no seed), so the models and maps are expected to match bit
  for bit with the same library versions and platform.
* **float_grid** - deterministic; exact reproduction needs Linux/glibc libm.
* **reads_three_others** - deterministic LightGBM (2 threads); the original fit ran
  on Windows and trees may differ in the last bits on another platform.
* **reads_two_rebuilds and the three sequence networks** - trained on a GPU
  (cuDNN GRU kernels, mixed precision for the networks, a float16 input table
  for reads_two_rebuilds): the recipes, seeds
  and data are fixed but a re-run gives slightly different weights, hence
  different maps.

## 6. From these outputs to the released weights

`committee.build_committee` writes the committee folder in the layout of the
original submission.  For the GitHub Release that folder was repackaged into
the layout that `breakdetect.load_model` reads (see `MANIFEST.sha256` in the
weights): numeric files (`.npz`, `.npy`) were copied unchanged under shorter
names, and the JSON configuration and metadata were rewritten without internal
names or training ids.  The one-off repackaging script is not included; the
scores of the released weights were checked bit for bit against the
submission.
