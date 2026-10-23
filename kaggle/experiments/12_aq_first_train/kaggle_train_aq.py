"""
FM-LLM on Accra PM2.5 (OpenAQ): first training run.

Same architecture and training recipe as 09_instance_norm_retrain (dropout
0.2, instance norm, hybrid loss, batch 256 split over T4x2, lr 2e-4,
patience 3), now on hourly PM2.5 from Accra's low-cost sensors. Each sensor is
one series. The model reads 28 days (672 h) and predicts the next 96 h in one
step. We score the first 24 h and 48 h, which is the horizon that matters for
the app (September meeting, point 1).

Split by time, so nothing from the future leaks into training:
  train   up to 30 Sep 2025       (includes the Dec 2024 - Feb 2025 Harmattan)
  val     Oct - Nov 2025          (early stopping)
  test    Dec 2025 - Feb 2026     (the 2025-26 Harmattan: the season the app is for)
Errors are reported per sensor in µg/m³, against the simple rules (same as the
last hour / last 24 h / last week), plus warning scores per sensor
(September meeting, point 3) for the events in EVENTS (main one: Ghana's
24-hour PM2.5 standard, 35 µg/m³, GS 1236:2019).

Data cleaning (fmllm.data.load_hourly): readings outside 0-1000 µg/m³ and runs
of 3+ hours of exact zeros (a fault: broken sensors keep sending 0) are
treated as missing.

Runs on Kaggle with the `fm-llm-accra-aq` dataset attached (built by
scripts/build_kaggle_bundle.py: the fmllm package + the OpenAQ CSVs) and the
HF-token dataset. Locally: `python kaggle_train_aq.py --smoke` runs a few
batches on CPU with a tiny stand-in for Llama, to check the plumbing.
"""

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def package_roots(search_dir):
    """
    Args:
        search_dir: folder to search (recursively).

    Returns:
        List of folders that directly contain an fmllm/ package (the folder you'd
        add to sys.path), i.e. the parent of each fmllm/__init__.py found.
    """
    if not search_dir.exists():
        return []
    return [init.parent.parent for init in search_dir.rglob("fmllm/__init__.py")]


def unpack_zips(inputs, work):
    """
    Unpack fmllm.zip and data.zip from the attached dataset, in case Kaggle kept
    the uploaded folders zipped. Each zip goes into its own folder named after it,
    so the result is work/fmllm/... whether or not the zip stores an fmllm/ prefix.

    Args:
        inputs: /kaggle/input.
        work: folder to unpack into (writable, e.g. /kaggle/working/bundle).
    """
    for z in [*inputs.rglob("fmllm.zip"), *inputs.rglob("data.zip")]:
        with zipfile.ZipFile(z) as f:
            prefixed = all(n.startswith(z.stem + "/") for n in f.namelist())
            f.extractall(work if prefixed else work / z.stem)
        print(f"unpacked {z} -> {work}")


def find_bundle(inputs=Path("/kaggle/input"), work=Path("/kaggle/working/bundle"), local_root=None):
    """
    Locate the folder holding the fmllm package: the attached Kaggle dataset, or
    the local repo when run on this machine. The folder is added to sys.path.

    Args:
        inputs: where Kaggle mounts attached datasets.
        work: where to unpack the bundle zips if Kaggle didn't.
        local_root: the repo root when running locally. Default: three levels up from
            this script's folder (12_.../ -> experiments/ -> kaggle/ -> repo). On Kaggle the
            script runs from /kaggle/src, which has no such folder, so it's skipped.

    Returns:
        Path of the folder that contains fmllm/.

    Raises:
        RuntimeError: if the package can't be found (dataset not attached).
    """
    if local_root is None and len(HERE.parents) > 2:
        local_root = HERE.parents[2]
    roots = package_roots(inputs)
    if not roots and inputs.exists() and next(inputs.rglob("fmllm.zip"), None):
        unpack_zips(inputs, work)
        roots = package_roots(work)
    if local_root is not None and (local_root / "fmllm" / "__init__.py").exists():
        roots.append(local_root)
    if not roots:
        raise RuntimeError(f"fmllm package not found under {inputs} or the local repo: "
                           "is the fm-llm-accra-aq dataset attached?")
    sys.path.insert(0, str(roots[0]))
    return roots[0]


BUNDLE = find_bundle()

# Imported only now: fmllm can be imported only after find_bundle() has put it on sys.path.
# pylint: disable=wrong-import-position,wrong-import-order
import numpy as np
import torch
from torch import nn

from fmllm.aq import EVENTS, SPLIT, build_splits
from fmllm.evaluation import collect_records, print_summary, write_results
from fmllm.losses import hybrid_loss, sequence_balance_loss
from fmllm.model import FMLLM, LlamaBackbone, ModelConfig, TinyBackbone, save_checkpoint
# pylint: enable=wrong-import-position,wrong-import-order

# Data rules, split and warning events live in fmllm/aq.py (shared with experiment 13).
BATCH_SIZE, LR, PATIENCE, MAX_EPOCHS, LAMBDA_BALANCE = 256, 2e-4, 3, 30, 1.0


def find_csv():
    """
    Locate the PM2.5 data inside the bundle (the folder layout differs between Kaggle and the local repo).

    Returns:
        Path to pm25_hourly.csv.

    Raises:
        FileNotFoundError: if it isn't in any of the expected folders.
    """
    for d in (BUNDLE / "data", BUNDLE / "data" / "raw" / "openaq_accra", BUNDLE):
        if (d / "pm25_hourly.csv").exists():
            return d / "pm25_hourly.csv"
    for top in (Path("/kaggle/input"), Path("/kaggle/working")):  # anywhere in the attached data
        if top.exists():
            hit = next(top.rglob("pm25_hourly.csv"), None)
            if hit:
                return hit
    raise FileNotFoundError("pm25_hourly.csv not found in the bundle")


def main():
    """
    Train FM-LLM on Accra PM2.5 and evaluate it on the test period.

    Steps:
        1. Build the model: real Llama on GPU, or a tiny stand-in with --smoke.
        2. Load the data and cut it into train / validation / test windows (fmllm.aq.build_splits).
        3. Train with early stopping on the validation loss, keeping the best checkpoint.
        4. Reload the best checkpoint and score it on the test period (fmllm.evaluation).

    Command-line:
        --smoke: run on CPU with a tiny stand-in for Llama and only a few batches,
            to check the code runs end to end. The numbers it prints mean nothing.

    Writes (to /kaggle/working on Kaggle, ./output locally, ./output_smoke for --smoke):
        aq_pm25_best.pt, test_forecasts.csv, test_mae_per_sensor.csv, test_results.json, test_bootstrap.csv
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny stand-in backbone, a few batches")
    args = ap.parse_args()
    torch.manual_seed(0)

    # smoke runs get their own folder, so cleaning up after one never touches real results in output/
    out_dir = (Path("/kaggle/working") if Path("/kaggle/working").exists()
               else HERE / ("output_smoke" if args.smoke else "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. model settings and backbone
    if args.smoke:
        config = ModelConfig(mlp_dim=32, llm_dim=64, hidden_dim=32)
        backbone, device, num_gpus = TinyBackbone(config.llm_dim), torch.device("cpu"), 0
    else:
        assert torch.cuda.is_available(), "No GPU: check the kernel's accelerator setting"
        config = ModelConfig()
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        backbone = LlamaBackbone(device)

    # 2. data: train / validation / test windows (rules in fmllm/aq.py)
    data = build_splits(find_csv(), config)
    scale, train_ds, val_ds, test_ds = data.scale, data.train, data.val, data.test
    print(f"{data.wide.shape[1]} sensors, {len(data.wide)} hours ({data.wide.index.min()} -> {data.wide.index.max()})")
    print(f"windows  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"sensors  train={len({j for j, _ in train_ds.items})}  test={len({j for j, _ in test_ds.items})}")
    empty = [n for n, d in (("train", train_ds), ("val", val_ds), ("test", test_ds)) if len(d) == 0]
    if empty and not args.smoke:
        raise RuntimeError(f"no gap-free windows in {empty}: check the split dates or the gap rule")
    if empty:  # smoke runs on partial data: just exercise the code path
        val_ds = test_ds = train_ds

    # 3. training
    model = FMLLM(config, backbone).to(device)
    if device.type == "cuda":
        model.set_decoder_dtype(torch.bfloat16)
    net = nn.DataParallel(model, device_ids=list(range(num_gpus))) if num_gpus > 1 else model
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    print(f"trainable parameters: {sum(p.numel() for p in model.trainable_parameters()):,}")

    loader_kw = {"batch_size": BATCH_SIZE if not args.smoke else 16, "num_workers": 0}
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kw)
    max_batches = 3 if args.smoke else None

    ckpt_path = out_dir / "aq_pm25_best.pt"
    best, bad_epochs = float("inf"), 0
    for epoch in range(1 if args.smoke else MAX_EPOCHS):
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
            if (i + 1) % 20 == 0:
                print(f"  epoch {epoch + 1} batch {i + 1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        model.eval()
        val = []
        with torch.no_grad():
            for i, (hist, fut, _, _) in enumerate(val_loader):
                if max_batches and i >= max_batches:
                    break
                val.append(hybrid_loss(net(hist.to(device))[0], fut.to(device)).item())
        val_loss = float(np.mean(val))
        print(f"Epoch {epoch + 1}  train={np.mean(losses):.4f}  val={val_loss:.4f}  "
              f"time={(time.time() - t0) / 60:.1f}min", flush=True)
        if val_loss < best:
            best, bad_epochs = val_loss, 0
            save_checkpoint(model, ckpt_path, epoch=epoch, val_loss=val_loss, scale=json.dumps(scale_to_json(scale)),
                            split=json.dumps(SPLIT))
            print(f"  -> saved best checkpoint (val={val_loss:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    # 4. evaluate the best checkpoint on the test period (one row per forecast + bootstrap intervals)
    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.embedding_module.load_state_dict(best_ckpt["embedding_module"])
    model.decoder.load_state_dict(best_ckpt["decoder"])
    records = collect_records(model, test_ds, device, EVENTS, max_batches=max_batches)
    summary, per_sensor, boot = write_results(records, EVENTS, out_dir, extra={"split": SPLIT})
    print_summary(summary, per_sensor, boot, EVENTS)


def scale_to_json(scale):
    """
    Args:
        scale: {sensor_id: (mean, std)} from train_scale.

    Returns:
        The same dict with string keys, so it can be stored as JSON inside the checkpoint.
    """
    return {str(k): v for k, v in scale.items()}


if __name__ == "__main__":
    main()
