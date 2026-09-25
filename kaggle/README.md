# Kaggle workflow

Kaggle has no live/interactive equivalent to `colab-mcp`. Instead, every
experiment here is a **script kernel**: pushed via the Kaggle CLI, run
server-side, polled for completion, then its log/output pulled back down.

## One-time setup

1. Create a Kaggle account, then get an API token at kaggle.com/settings
   ("Create New Token"). Put it in `.env` at the repo root:
   ```
   KAGGLE_API_TOKEN=KGAT_...
   ```
2. `uv sync` installs the `kaggle` CLI (tracked in `pyproject.toml`).
3. The two private datasets under `datasets/` need to exist on Kaggle before
   any experiment can run — push them once:
   ```powershell
   $env:KAGGLE_API_TOKEN = (Get-Content .env | Select-String KAGGLE_API_TOKEN).ToString().Split('=')[1]
   kaggle datasets create -p kaggle/datasets/hf_token
   kaggle datasets create -p kaggle/datasets/checkpoint
   ```
   `hf_token/hf_token.txt` holds the Hugging Face token (gated Llama-3.2-1B
   access) — Kaggle Secrets don't survive CLI-pushed kernel versions, only
   UI-saved ones, so we read it from this dataset instead
   (`/kaggle/input/datasets/mackeykumi/fm-llm-hf-token/hf_token.txt`, with a
   fallback to the non-nested path — Kaggle's dataset mount path has been
   observed to vary between kernels).

## Running an experiment

```powershell
$env:PYTHONUTF8 = 1   # avoids a console encoding crash on Windows when pulling logs
cd kaggle/experiments/<name>
kaggle kernels push -p .
kaggle kernels status mackeykumi/<kernel-id>      # poll until COMPLETE or ERROR
kaggle kernels output mackeykumi/<kernel-id> -p ./output -o
```

The `<kernel-id>` is the `id` field in that experiment's `kernel-metadata.json`
(Kaggle sometimes derives a slightly different slug from the title on first
push — the push output tells you the actual URL if so; update `id` to match).

## Experiments, in order

| Folder | Question it answered | Result |
|---|---|---|
| `01_batch256_single_gpu` | Does the paper's batch=256 fit on one Kaggle T4? | OOM at 15.28GB (matches Colab's free-tier T4 exactly) |
| `02_batch256_dual_gpu` | Does splitting batch=256 across Kaggle's two T4s (data parallel) fit? | Yes — ~12.3GB per GPU |
| `03_full_train_dual_gpu` | Full ETTh1 training at batch=256 on T4x2 | 16 epochs, early-stopped, best val_loss=1.0228 (epoch 13). Checkpoint saved to `datasets/checkpoint/etth1_96_best.pt` |
| `04_test_eval` | Real MSE/MAE on the held-out test set | MSE=0.4911, MAE=0.4729 vs. paper's 0.342/0.380 (Table A.12, ETTh1-96) |
| `05_dropout_retrain` | Does adding Table 2's dropout (0.2) improve on the no-dropout checkpoint? | 30 epochs (hit the ceiling, never early-stopped), best val_loss=0.9589 (vs. `03`'s 1.0228). ~4h16min wall-clock (dropout kept the model improving longer than `03`'s early stop at epoch 16). Checkpoint saved to `datasets/checkpoint/etth1_96_dropout_best.pt` |
| `06_rollout_eval` | What's the real MSE/MAE at horizons 192/336/720, no-dropout vs. dropout? | Dropout wins at every horizon. ~27min total (single T4, inference-only, both checkpoints). Superseded by `10`'s four-way table below |
| `07_dropout_retrain_longer` | Was `05`'s NUM_EPOCHS=30 ceiling cutting off real improvement? | Yes, partially — early-stopped naturally at epoch 35 (patience=3), best val_loss=0.9540 (small gain over `05`'s 0.9589). ~4h54min. Checkpoint saved to `datasets/checkpoint/etth1_96_dropout_longer_best.pt` |
| `08_rollout_eval_longer` | Does `07`'s marginal val_loss gain translate to a bigger rollout gain? | Yes, more than expected — MSE improves 2.8%/4.5%/8.0% at 192/336/720 over `05`'s checkpoint, growing with horizon. Surfaced that our rollout error still grows much faster across horizons (+26%, 192→720) than the paper's (+5%) even on the best checkpoint. ~40min, three checkpoints |
| `09_instance_norm_retrain` | Does per-window (instance) normalization on top of dropout fix the horizon-growth problem? | Best val_loss=0.9401, early-stopped faster too (epoch 26 vs. `07`'s 35). ~3h30min — faster **and** better than `07`. Checkpoint saved to `datasets/checkpoint/etth1_96_instnorm_best.pt` |
| `10_rollout_eval_instnorm` | Does it hold up at the rollout horizons? | Yes, decisively — best result by far. See the four-way table below. ~46min, four checkpoints |
| `11_app_forecasts` | Precompute forecasts for the GridForecast demo app | Same rollout as `10`, all four checkpoints; saves every 24th window's 720-step forecast to `forecasts.npz` (copy to `app/precomputed/`) and re-prints the full-test MSE/MAE as a cross-check |

| Horizon | no_dropout | dropout | dropout_longer | instnorm | Paper (Table A.12) |
|---|---|---|---|---|---|
| 192 | 0.5089 / 0.4882 | 0.4417 / 0.4571 | 0.4292 / 0.4445 | **0.3945 / 0.4225** | 0.377 / 0.403 |
| 336 | 0.5423 / 0.5158 | 0.4858 / 0.4925 | 0.4641 / 0.4730 | **0.4149 / 0.4402** | 0.395 / 0.415 |
| 720 | 0.6381 / 0.5792 | 0.5894 / 0.5661 | 0.5422 / 0.5350 | **0.4721 / 0.4839** | 0.397 / 0.429 |

Instance normalization (`instnorm`) closes the gap to the paper to ~5% at
192/336, and shrinks it from ~37% to ~19% at 720. It isn't part of the
paper's stated equations, but the paper explicitly borrows PatchTST's
channel-independence design (Section 3.2.1), and PatchTST's own
architecture is built around RevIN-style per-instance normalization — it's
plausible this was inherited without being called out as a contribution.
Implementation: `instance_normalize`/`instance_denormalize` are pure,
parameter-free pre/post-processing sitting on top of (not replacing) the
existing global per-channel normalization, so reported MSE/MAE stay on the
same globally-normalized scale as the paper's numbers. See
`09_instance_norm_retrain/kaggle_train_full.py`'s docstring for the full
mechanics, including how the fixed per-window mean/std carries through the
autoregressive rollout unchanged.

Also found along the way: Kaggle's **P100** accelerator can't run at all under
the current PyTorch build (Pascal/sm_60 isn't in the supported CUDA
capability list) — use `"machine_shape": "NvidiaTeslaT4"` in
`kernel-metadata.json`, which reliably gives you the T4x2 pair.
