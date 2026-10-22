# FM-LLM: reproduction and Accra air-quality forecasting

Final-year project (Mackey Junior Kumi & Arnold Kyei Baffour), in two phases:

1. **Reproduction** of *FM-LLM: A frequency-enhanced mixture-of-experts framework for
   adapting LLMs to time series forecasting* (Gu et al., Knowledge-Based Systems,
   2026): Fourier Embedding Module, FAN-MoE decoder with load-balanced routing,
   hybrid time-frequency loss. Done on ETTh1 (below).
2. **Application: PM2.5 early warning for Accra.** We train FM-LLM on hourly
   readings from Accra's live low-cost air sensors (OpenAQ) to warn people 1–2
   days ahead, with honest likelihoods, scored sensor by sensor. Work in progress
   (experiment 12 on).

This repo holds the research, data and model training only. The web and mobile
apps live in Arnold's separate app repo and only read the forecasts this repo
produces. The September 2026 GridForecast demo (ETTh1) was removed and is kept
under the git tag `gridforecast-demo`.

## Status: phase 1 (ETTh1 reproduction)

ETTh1 (input=672, predict=96) is trained and evaluated end-to-end. Four
checkpoints have been compared so far — no-dropout, dropout, dropout with a
longer training ceiling, and dropout + instance normalization — and the
last one is decisively the best:

- **Dropout (Table 2: 0.2)** was missing from the first pass; adding it and
  retraining from scratch (`05_dropout_retrain`, then `07` with a longer
  epoch ceiling once `05` turned out to still be improving at cutoff)
  brought best val_loss from 1.0228 down to 0.9540 across two retrains.
- 96-step test-set evaluation (`04_test_eval`, no-dropout checkpoint):
  MSE=0.4911, MAE=0.4729 vs. the paper's 0.342/0.380 (Table A.12).
- **Autoregressive rollout** (`06`/`08`/`10_rollout_eval*`) reaches
  192/336/720 steps by feeding the model's own predictions back in as input
  (sliding 7-token context), matching how the paper itself reaches these
  horizons. That rollout comparison surfaced a real problem: our error grew
  much faster across horizons (+26% MSE, 192→720) than the paper's own
  numbers do (+5%) — the classic symptom of a model not robust to each
  window's own local level/scale.
- **Instance (per-window) normalization** (`09_instance_norm_retrain`), a
  second local normalization layer sitting on top of the existing global
  one — not part of the paper's stated equations, but plausibly inherited
  from PatchTST's channel-independence design (Section 3.2.1), whose own
  architecture is built around exactly this — fixed most of it:

  | Horizon | No-dropout | Dropout | Dropout (longer) | **Instance norm** | Paper (Table A.12) |
  |---|---|---|---|---|---|
  | 192 | 0.5089 / 0.4882 | 0.4417 / 0.4571 | 0.4292 / 0.4445 | **0.3945 / 0.4225** | 0.377 / 0.403 |
  | 336 | 0.5423 / 0.5158 | 0.4858 / 0.4925 | 0.4641 / 0.4730 | **0.4149 / 0.4402** | 0.395 / 0.415 |
  | 720 | 0.6381 / 0.5792 | 0.5894 / 0.5661 | 0.5422 / 0.5350 | **0.4721 / 0.4839** | 0.397 / 0.429 |

  Instance normalization also trained faster and to a better val_loss
  (0.9401, early-stopped at epoch 26) than the longer dropout-only run
  (0.9540, epoch 35). The gap to the paper is now ~5% at 192/336, and down
  from ~37% to ~19% at 720 — real progress, but the paper still stays
  flatter across horizons than we do, so some residual error-accumulation
  gap remains at the longest horizon.

Open questions from phase 1: what still drives the residual 720-horizon gap
(the FAN single-layer ambiguity in Eq. 8, or hyperparameters beyond Table 2's
defaults; FAN's `p_ratio` split is our own invented value). The app needs
1–2 day horizons, where we're closest to the paper, so this is no longer on
the critical path.

## Status: phase 2 (Accra PM2.5)

```powershell
python scripts/fetch_openaq.py          # hourly PM2.5 for every live Accra sensor -> data/raw/openaq_accra/ (needs OPENAQ_API_KEY)
python scripts/build_kaggle_bundle.py   # fmllm + data -> kaggle/datasets/accra_aq/ (the dataset experiment 12 attaches)
python kaggle/experiments/12_aq_first_train/kaggle_train_aq.py --smoke   # CPU check of the whole pipeline
```

Experiment 12 trains with the ETTh1 recipe (dropout, instance norm, batch 256
on T4x2). It splits by time: train to Sep 2025, validate Oct–Nov 2025, and test on
the Dec 2025 – Feb 2026 Harmattan. It reports MAE at 24 h and 48 h and warning
scores per sensor, against simple rules (same as the last hour / last 24 h /
last week). Design constraints from the September meeting are in
`docs/meeting_notes/`.

## Environment

This machine has **no NVIDIA GPU**. All real training/eval runs on Kaggle
(script kernels, pushed via CLI — see [`kaggle/README.md`](kaggle/README.md)
for the full workflow). This machine is for code editing, git, and tiny CPU
smoke tests only.

```powershell
uv sync          # creates .venv/ and installs CPU-only deps
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Select `.venv/Scripts/python.exe` as the VS Code interpreter (already set as
default in `.vscode/settings.json`).

Copy `.env.example` to `.env` and fill in `HF_TOKEN` (Hugging Face) and
`KAGGLE_API_TOKEN` (Kaggle, see `kaggle/README.md`) — `.env` is gitignored,
never commit it.

## Project layout

```
fmllm/            # shared package (phase 2): model, losses, forecast + baselines, warning scores, data windows
scripts/          # fetch_openaq.py, build_kaggle_bundle.py
kaggle/
  README.md       # full Kaggle CLI workflow (push/poll/pull, one-time setup)
  datasets/        # private Kaggle datasets we maintain (HF token, checkpoints, accra_aq bundle)
  experiments/      # one script kernel per experiment, numbered in run order
tests/            # CPU-only; no GPU, Llama download or data needed
docs/             # meeting notes, handoff
data/, checkpoints/, outputs/   # gitignored — populated locally or pulled from Kaggle
```

Experiments 01–10 (phase 1) are self-contained: each kernel carries its own
copy of the architecture, a faithful port of the paper's equations (Fourier
Embedding Module Eq. 9-11, FAN-MoE decoder Eq. 13-18, hybrid loss Eq. 19-23).
From experiment 12 on, kernels import the shared `fmllm` package instead,
shipped to Kaggle inside the `fm-llm-accra-aq` dataset. `fmllm` matches
experiment 09's code exactly (checked by `tests/test_fmllm.py`), and the ETTh1
checkpoints load into it.
