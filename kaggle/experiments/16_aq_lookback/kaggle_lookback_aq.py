"""
FM-LLM on Accra PM2.5: how much history should the model read? (experiment 16)

The Prof's September point 4: try other look-back windows, e.g. leave out the most recent
days and see whether the forecast changes. Run 2 (experiment 12) reads the last 28 days.
This kernel trains three more models with exactly run 2's recipe (dropout 0.2, instance
norm, hybrid loss, batch 256 over T4x2, lr 2e-4, patience 3, seed 0, same split), changing
only what the model reads:

  L16        the last 16 days (384 h, 4 tokens)
  L16_gap7   16 days that END 7 days before the forecast (the most recent week left out)
  L56        the last 56 days (1344 h, 14 tokens)

Window lengths are whole tokens of 4 days (96 h), as the architecture requires. L16 and
L16_gap7 differ only in the gap, so together they measure what the most recent week is
worth. Runs go shortest first; each writes its own folder (checkpoint + the usual test
results), so a later run failing never loses an earlier one.

The fair comparison (every model, run 2 included, on the same test forecasts, plus Mar-Oct
2026 and the new sensors) is done afterwards on CPU by scripts/compare_lookback.py.

    python kaggle_lookback_aq.py --smoke     # CPU, tiny stand-in for Llama, a few batches
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12_aq_first_train"))   # locally; on Kaggle the bundle is found below

# Same bundle discovery as experiment 12 (Kaggle dataset or local repo).
# pylint: disable=wrong-import-position,wrong-import-order
try:
    from kaggle_train_aq import find_bundle, find_csv, scale_to_json   # local run
except ImportError:  # on Kaggle only this script is uploaded: define the same helpers here
    import zipfile

    def _package_roots(search_dir):
        """
        Args:
            search_dir: folder to search.

        Returns:
            Folders that directly contain an fmllm/ package.
        """
        return [i.parent.parent for i in search_dir.rglob("fmllm/__init__.py")] if search_dir.exists() else []

    def find_bundle(inputs=Path("/kaggle/input"), work=Path("/kaggle/working/bundle")):
        """
        Args:
            inputs: where Kaggle mounts attached datasets.
            work: where to unpack the bundle zips if Kaggle kept them zipped.

        Returns:
            The folder holding fmllm/ (added to sys.path).
        """
        roots = _package_roots(inputs)
        if not roots:
            for z in [*inputs.rglob("fmllm.zip"), *inputs.rglob("data.zip")]:
                with zipfile.ZipFile(z) as f:
                    prefixed = all(n.startswith(z.stem + "/") for n in f.namelist())
                    f.extractall(work if prefixed else work / z.stem)
            roots = _package_roots(work)
        if not roots:
            raise RuntimeError("fmllm package not found: is the fm-llm-accra-aq dataset attached?")
        sys.path.insert(0, str(roots[0]))
        return roots[0]

    def find_csv():
        """Returns: path to pm25_hourly.csv anywhere in the attached data."""
        for top in (Path("/kaggle/input"), Path("/kaggle/working")):
            hit = next(top.rglob("pm25_hourly.csv"), None) if top.exists() else None
            if hit:
                return hit
        raise FileNotFoundError("pm25_hourly.csv not found")

    def scale_to_json(scale):
        """
        Args:
            scale: {sensor_id: (mean, std)}.

        Returns:
            The same dict with string keys (JSON-safe).
        """
        return {str(k): v for k, v in scale.items()}

    find_bundle()

import numpy as np
import torch
from torch import nn

from fmllm.aq import EVENTS, SPLIT, build_splits
from fmllm.evaluation import collect_records, print_summary, write_results
from fmllm.losses import hybrid_loss, sequence_balance_loss
from fmllm.model import FMLLM, LlamaBackbone, ModelConfig, TinyBackbone, save_checkpoint
# pylint: enable=wrong-import-position,wrong-import-order

BATCH_SIZE, LR, PATIENCE, MAX_EPOCHS, LAMBDA_BALANCE = 256, 2e-4, 3, 30, 1.0   # run 2's recipe
DAY = 24
RUNS = (("L16", 16, 0), ("L16_gap7", 16, 7), ("L56", 56, 0))   # (name, history days, gap days)


# One training run end to end, kept in one piece so it reads like experiment 12's main().
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals,too-many-statements
def train_one(name, history_days, gap_days, out_root, backbone, device, num_gpus, smoke):
    """
    Train and test one look-back variant, exactly like run 2 except for the history it reads.

    Args:
        name: run name (also its output folder).
        history_days: days of history the model reads (a multiple of 4).
        gap_days: days left out between the history and the forecast.
        out_root: folder that receives one subfolder per run.
        backbone: the shared Llama (or tiny stand-in), loaded once for all runs.
        device: torch device.
        num_gpus: GPUs to split each batch over.
        smoke: tiny model, a few batches, numbers meaningless.

    Writes (to out_root/name/):
        aq_pm25_best.pt (with seq_len and gap in its metadata), test_*.csv/json, run.json.
    """
    torch.manual_seed(0)
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    kw = {"mlp_dim": 32, "llm_dim": 64, "hidden_dim": 32} if smoke else {}
    config = ModelConfig(seq_len=history_days * DAY, **kw)
    gap = gap_days * DAY
    data = build_splits(find_csv(), config, gap=gap)
    train_ds, val_ds, test_ds = data.train, data.val, data.test
    print(f"\n=== {name}: reads {history_days} days, gap {gap_days} days ===")
    print(f"windows  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"sensors train={len({j for j, _ in train_ds.items})}", flush=True)
    if smoke and len(val_ds) == 0:
        val_ds = test_ds = train_ds

    model = FMLLM(config, backbone).to(device)
    if device.type == "cuda":
        model.set_decoder_dtype(torch.bfloat16)
    net = nn.DataParallel(model, device_ids=list(range(num_gpus))) if num_gpus > 1 else model
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    loader_kw = {"batch_size": 16 if smoke else BATCH_SIZE, "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)
    max_batches = 3 if smoke else None

    ckpt = out / "aq_pm25_best.pt"
    best, bad, history = float("inf"), 0, []
    t_start = time.time()
    for epoch in range(1 if smoke else MAX_EPOCHS):
        t0 = time.time()
        model.train()
        losses = []
        for i, (hist, fut, _, _) in enumerate(train_loader):
            if max_batches and i >= max_batches:
                break
            pred, top_indices, scores = net(hist.to(device))
            loss = hybrid_loss(pred, fut.to(device)) + LAMBDA_BALANCE * sequence_balance_loss(
                scores, top_indices, config.num_routed, config.top_k)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.decoder.gate.update_bias(top_indices)
            losses.append(loss.item())
        model.eval()
        val = []
        with torch.no_grad():
            for i, (hist, fut, _, _) in enumerate(val_loader):
                if max_batches and i >= max_batches:
                    break
                val.append(hybrid_loss(net(hist.to(device))[0], fut.to(device)).item())
        val_loss = float(np.mean(val))
        history.append({"epoch": epoch + 1, "train": float(np.mean(losses)), "val": val_loss})
        print(f"{name} epoch {epoch + 1}  train={np.mean(losses):.4f}  val={val_loss:.4f}  "
              f"time={(time.time() - t0) / 60:.1f}min", flush=True)
        if val_loss < best:
            best, bad = val_loss, 0
            save_checkpoint(model, ckpt, epoch=epoch, val_loss=val_loss, scale=json.dumps(scale_to_json(data.scale)),
                            split=json.dumps(SPLIT), gap=gap)
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"{name}: early stopping")
                break

    state = torch.load(ckpt, map_location=device)
    model.embedding_module.load_state_dict(state["embedding_module"])
    model.decoder.load_state_dict(state["decoder"])
    records = collect_records(model, test_ds, device, EVENTS, max_batches=max_batches)
    summary, per_sensor, boot = write_results(records, EVENTS, out, extra={"split": SPLIT, "run": name})
    print_summary(summary, per_sensor, boot, EVENTS)
    (out / "run.json").write_text(json.dumps({
        "name": name, "history_days": history_days, "gap_days": gap_days, "seq_len": config.seq_len, "gap": gap,
        "best_val": best, "epochs": history, "minutes": (time.time() - t_start) / 60,
        "windows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}}, indent=1))
    del model, net, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def preflight_llama_access():
    """
    Before loading Llama, check the token and one small gated download, and say which step
    fails (the first two pushes of this kernel died with a 401 that couldn't be reproduced
    elsewhere). Prints no secret. The token is also put in HF_TOKEN, so every Hugging Face
    call in this process uses it, whatever transformers version Kaggle ships.

    Raises:
        RuntimeError: no token found, or Hugging Face refuses it.
    """
    import os  # pylint: disable=import-outside-toplevel
    import huggingface_hub  # pylint: disable=import-outside-toplevel
    from fmllm.model import hf_token  # pylint: disable=import-outside-toplevel
    tok = hf_token()
    print(f"preflight: token found={tok is not None}, HF env vars={sorted(k for k in os.environ if k.startswith('HF_'))}",
          flush=True)
    if not tok:
        raise RuntimeError("no Hugging Face token: is the fm-llm-hf-token dataset attached?")
    os.environ["HF_TOKEN"] = tok
    huggingface_hub.whoami(token=tok)
    huggingface_hub.hf_hub_download("meta-llama/Llama-3.2-1B", "config.json", token=tok)
    print("preflight: token valid and Llama download allowed", flush=True)


def main():
    """
    Train every variant in RUNS, one after another, sharing one Llama.

    Command-line:
        --smoke: CPU, tiny stand-in backbone, a few batches (checks the code path only).
        --only: comma-separated run names to train (default: all).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    out_root = (Path("/kaggle/working") if Path("/kaggle/working").exists()
                else HERE / ("output_smoke" if args.smoke else "output"))
    if args.smoke:
        backbone, device, num_gpus = TinyBackbone(64), torch.device("cpu"), 0
    else:
        assert torch.cuda.is_available(), "No GPU: check the kernel's accelerator setting"
        device, num_gpus = torch.device("cuda:0"), torch.cuda.device_count()
        preflight_llama_access()
        backbone = LlamaBackbone(device)
    wanted = set(args.only.split(",")) if args.only else None
    for name, days, gap_days in RUNS:
        if wanted is None or name in wanted:
            try:
                train_one(name, days, gap_days, out_root, backbone, device, num_gpus, args.smoke)
            except Exception as e:   # pylint: disable=broad-except  # keep going: the other runs still count
                print(f"{name} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
