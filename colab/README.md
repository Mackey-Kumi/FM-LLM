# Remote GPU workflow: VS Code (local) → Colab (GPU) → Google Drive (storage)

This machine has no NVIDIA GPU and very little free disk (~13GB), so it is
**dev-only**: editing code, running tiny CPU smoke tests, git history. All real
training/eval runs on a Colab GPU runtime, with datasets/checkpoints/HF cache
persisted on Google Drive (never on the ephemeral Colab VM disk, and never on
this machine).

```
local machine (this repo)        Colab VM (GPU, ephemeral disk)     Google Drive (persistent)
  code + git history      --git-->  clone of the repo                data/, checkpoints/,
  CPU-only .venv (dev)              .venv or pip installs w/ CUDA     outputs/, hf_cache/
  VS Code, editing        <--SSH--  colab-ssh + ngrok tunnel
```

## One-time setup

1. **Push this repo to a GitHub remote** (not done yet — ask me when you're
   ready, since that's a "visible to others" action). Colab pulls code via
   `git clone`/`git pull`, not by copying files by hand.
2. **Get a free ngrok authtoken**: https://dashboard.ngrok.com/get-started/your-authtoken
   — needed for `colab-ssh` to expose an SSH endpoint from the Colab VM.
3. Open `colab/colab_bootstrap.ipynb` in Google Colab (Runtime → Change runtime
   type → GPU), and run its cells top to bottom. It will:
   - Mount your Google Drive and create `MyDrive/fm_llm/{data,checkpoints,outputs,hf_cache}`
   - Confirm the GPU is visible (`nvidia-smi`, `torch.cuda.is_available()`)
   - Install `colab-ssh`, prompt for your ngrok token, and open an SSH tunnel
   - Clone/pull this repo into `/content/fm_llm` and install dependencies
     (GPU-enabled `torch` this time — Colab's default index already has CUDA
     wheels, no special index needed there)
   - Print an SSH host block to paste into your local SSH config

## Every session after that

1. Run `colab/colab_bootstrap.ipynb` in Colab, wait for the SSH block to print.
2. In VS Code (this machine): `Remote-SSH: Connect to Host...` → paste/select
   the printed host → it opens a VS Code window running **inside the Colab
   VM**, with the GPU, your cloned repo, and Drive mounted at `/content/drive`.
3. Do the actual training/eval work in that remote window. `git push` from
   there when you want to save code changes back to GitHub; `git pull` here
   locally to sync.
4. Datasets, model weights, and checkpoints under `MyDrive/fm_llm/` survive
   across Colab sessions even after the VM is recycled — only the VM's local
   disk (`/content/` outside the Drive mount) is ephemeral.

## Why not just use Colab's own web notebook?

You can — nothing here stops you from working directly in the Colab UI. This
kit exists because you asked for VS Code-native development: real Pylance/
LSP support, multi-file editing, and git integration, none of which the Colab
web notebook UI gives you.
