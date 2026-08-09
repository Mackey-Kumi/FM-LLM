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

Also found along the way: Kaggle's **P100** accelerator can't run at all under
the current PyTorch build (Pascal/sm_60 isn't in the supported CUDA
capability list) — use `"machine_shape": "NvidiaTeslaT4"` in
`kernel-metadata.json`, which reliably gives you the T4x2 pair.
