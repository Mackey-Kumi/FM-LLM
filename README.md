# FM-LLM Reproduction

Reproduction of *FM-LLM: A frequency-enhanced mixture-of-experts framework for
adapting LLMs to time series forecasting* (Gu et al., Knowledge-Based Systems,
2026), starting from a small prototype notebook and building toward the full
paper: Fourier Embedding Module, FAN-MoE decoder with load-balanced routing,
hybrid time-frequency loss, and evaluation across all 11 benchmark datasets.

## Environment situation (read this first)

This machine has **no NVIDIA GPU** (integrated Intel graphics only) and
**~12GB free disk**. Two environments are used together:

| | This machine (local) | Colab (remote) |
|---|---|---|
| Role | Code editing, git, tiny CPU smoke tests | Actual training/eval |
| Python env | `.venv/` via `uv`, CPU-only torch | GPU torch, installed fresh each session |
| Storage | Code only | Google Drive (`MyDrive/fm_llm/`) for data/checkpoints/HF cache |

See [`colab/README.md`](colab/README.md) for the full VS Code ⇄ Colab GPU
workflow (SSH tunnel via `colab-ssh`/ngrok, Drive-backed persistent storage).

## Local setup

```powershell
uv sync          # creates .venv/ and installs CPU-only deps
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Select `.venv/Scripts/python.exe` as the VS Code interpreter (already set as
default in `.vscode/settings.json`).

Copy `.env.example` to `.env` and fill in your Hugging Face token (`.env` is
gitignored — never commit it).

## Project layout

```
src/fm_llm/
  data/       # dataset loading, patch tokenization (Table 1 datasets)
  models/     # Fourier embedding module, FAN-MoE decoder, LLM backbone wrapper
  losses/     # hybrid time-frequency loss, sequence-wise balance loss
  training/   # training loop, autoregressive rollout
configs/      # per-dataset hyperparameters (Table 2)
scripts/      # CLI entry points (train, evaluate, download data)
notebooks/    # 00_prototype_forward_pass.ipynb — the original toy forward-pass demo
colab/        # Colab GPU bootstrap + VS Code Remote-SSH connection kit
tests/
data/, checkpoints/, outputs/   # gitignored — populated locally or on Drive
```

## Status

Environment is set up and verified (this step). The prototype notebook in
`notebooks/` is a toy forward-pass demo, not yet the paper's actual
architecture — that's the next phase of work.
