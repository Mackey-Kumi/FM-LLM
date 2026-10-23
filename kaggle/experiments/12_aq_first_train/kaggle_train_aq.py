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
import pandas as pd
import torch
from torch import nn

from fmllm.data import SensorWindows, fill_short_gaps, load_hourly, train_scale
from fmllm.events import event_occurs, score, score_by
from fmllm.forecast import BASELINES, baseline_forecast, forecast
from fmllm.losses import hybrid_loss, sequence_balance_loss
from fmllm.model import FMLLM, LlamaBackbone, ModelConfig, TinyBackbone, save_checkpoint
# pylint: enable=wrong-import-position,wrong-import-order

TRAIN_END, VAL_END, TEST_END = "2025-10-01", "2025-12-01", "2026-03-01"
HORIZONS = (24, 48)
# Warning events: name -> (statistic over the next `hours`, hours, limit in µg/m³).
#   ghana_24h_mean      Ghana's ambient standard, GS 1236:2019 Table 2: PM2.5 35 µg/m³, 24-hour average. Main event.
#   unhealthy_24h_mean  US AQI "unhealthy" tier (24-hour average 55.5): rarer, more serious days.
#   any_hour_35.5       the run-1 definition (any single hour >= 35.5), kept only for comparison:
#                       it happened in 93% of forecasts, so it can't separate good from lazy forecasters.
EVENTS = {
    "ghana_24h_mean": ("mean", 24, 35.0),
    "unhealthy_24h_mean": ("mean", 24, 55.5),
    "any_hour_35.5": ("max", 24, 35.5),
}
TEST_ISSUE_HOURS = (0, 6, 12, 18)   # test forecasts issued at fixed clock hours (Ghana time = UTC)
MAX_GAP = 12        # history: interpolate gaps up to 12 h (longer gaps -> window skipped).
                    # The scored part of the future (first 48 h) must be real readings only.
TRAIN_STRIDE, EVAL_STRIDE = 1, 6   # all training windows (far less data than ETTh1); every 6th for eval
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
        1. Load the hourly data, fill short gaps, compute per-sensor scaling (training period only).
        2. Build the model: real Llama on GPU, or a tiny stand-in with --smoke.
        3. Cut the timeline into train / validation / test windows.
        4. Train with early stopping on the validation loss, keeping the best checkpoint.
        5. Reload the best checkpoint and evaluate it on the test windows (see evaluate()).

    Command-line:
        --smoke: run on CPU with a tiny stand-in for Llama and only a few batches,
            to check the code runs end to end. The numbers it prints mean nothing.

    Writes (to /kaggle/working on Kaggle, ./output locally, ./output_smoke for --smoke):
        aq_pm25_best.pt, test_mae_per_sensor.csv, test_results.json
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="CPU, tiny stand-in backbone, a few batches")
    args = ap.parse_args()
    torch.manual_seed(0)

    # smoke runs get their own folder, so cleaning up after one never touches real results in output/
    out_dir = (Path("/kaggle/working") if Path("/kaggle/working").exists()
               else HERE / ("output_smoke" if args.smoke else "output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. data
    raw = load_hourly(find_csv())
    wide = fill_short_gaps(raw, max_gap=MAX_GAP)
    scale = train_scale(wide, TRAIN_END)
    print(f"{wide.shape[1]} sensors, {len(wide)} hours ({wide.index.min()} -> {wide.index.max()})")

    # 2. model settings and backbone
    if args.smoke:
        config = ModelConfig(mlp_dim=32, llm_dim=64, hidden_dim=32)
        backbone, device, num_gpus = TinyBackbone(config.llm_dim), torch.device("cpu"), 0
    else:
        assert torch.cuda.is_available(), "No GPU: check the kernel's accelerator setting"
        config = ModelConfig()
        num_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
        backbone = LlamaBackbone(device)

    # 3. windows for each split
    pred_len = config.patch_len
    common = {"seq_len": config.seq_len, "pred_len": pred_len, "scale": scale, "observed": raw,
              "real_future_hours": max(HORIZONS)}
    train_ds = SensorWindows(wide, end=TRAIN_END, stride=TRAIN_STRIDE, **common)
    val_ds = SensorWindows(wide, start=TRAIN_END, end=VAL_END, stride=EVAL_STRIDE, **common)
    test_ds = SensorWindows(wide, start=VAL_END, end=TEST_END, issue_hours=TEST_ISSUE_HOURS, **common)
    print(f"windows  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"sensors  train={len({j for j, _ in train_ds.items})}  test={len({j for j, _ in test_ds.items})}")
    empty = [n for n, d in (("train", train_ds), ("val", val_ds), ("test", test_ds)) if len(d) == 0]
    if empty and not args.smoke:
        raise RuntimeError(f"no gap-free windows in {empty}: check the split dates or MAX_GAP")
    if empty:  # smoke runs on partial data: just exercise the code path
        val_ds = test_ds = train_ds

    # 4. training
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
                            split=json.dumps({"train_end": TRAIN_END, "val_end": VAL_END, "test_end": TEST_END}))
            print(f"  -> saved best checkpoint (val={val_loss:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping")
                break

    # 5. evaluate the best checkpoint on the test period
    model.embedding_module.load_state_dict(torch.load(ckpt_path, map_location=device)["embedding_module"])
    model.decoder.load_state_dict(torch.load(ckpt_path, map_location=device)["decoder"])
    evaluate(model, test_ds, device, out_dir, max_batches=max_batches)


def scale_to_json(scale):
    """
    Args:
        scale: {sensor_id: (mean, std)} from train_scale.

    Returns:
        The same dict with string keys, so it can be stored as JSON inside the checkpoint.
    """
    return {str(k): v for k, v in scale.items()}


@torch.no_grad()
def evaluate(model, ds, device, out_dir, max_batches=None):
    """
    Score the model against the simple rules on every test forecast, per sensor.

    Each test forecast is issued at one of TEST_ISSUE_HOURS. Four forecasts are
    compared with what happened: FM-LLM, same-as-last-hour, same-as-yesterday
    and same-as-last-week.
      Accuracy: mean absolute error in µg/m³ over the first 24 h and 48 h.
      Warnings: for each event in EVENTS, "did it happen" vs "did this method
        forecast it", counted as hit / miss / false alarm / correct negative.
    Warnings are scored twice: over all forecasts ("all", 4 a day, whose
    24-hour periods overlap) and over one forecast per day ("daily", issued at
    00:00), where each day counts once.

    Args:
        model: trained FMLLM (its output is scaled; we convert back to µg/m³ here).
        ds: SensorWindows for the test period.
        device: torch device the model is on.
        out_dir: folder to write the results into.
        max_batches: stop after this many batches (smoke runs only). None = all.

    Writes:
        test_mae_per_sensor.csv: one row per sensor, MAE at each horizon for each method.
        test_results.json: overall MAE per method; for every event and scope (all/daily),
            the base rate (share of forecasts where the event happened) and warning scores
            overall and per sensor; the event definitions used.
    """
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    horizon = max(HORIZONS)
    rows, records = [], []
    i0 = 0  # index of the batch's first sample, to look up issue times
    for b, (hist, _, js, starts) in enumerate(loader):  # truth comes unscaled from ds.values below
        if max_batches and b >= max_batches:
            break
        pred = forecast(model, hist.to(device), horizon).float().cpu().numpy()[:, 0]
        for k, (j, s) in enumerate(zip(js.tolist(), starts.tolist())):
            sensor = ds.sensors[j]
            daily = ds.issue_time(i0 + k).hour == 0
            mean, std = ds.scale.get(sensor, (0.0, 1.0))
            raw_hist = ds.values[s: s + ds.seq_len, j]
            truth = ds.values[s + ds.seq_len: s + ds.seq_len + horizon, j]
            methods = {"fm_llm": pred[k] * std + mean}
            methods.update({m: baseline_forecast(raw_hist[:, None], m, horizon)[:, 0] for m in BASELINES})
            happened = {e: event_occurs(truth, *spec) for e, spec in EVENTS.items()}
            for m, p in methods.items():
                rows.append({"sensor": str(sensor), "method": m,
                             **{f"mae_{h}h": float(np.abs(p[:h] - truth[:h]).mean()) for h in HORIZONS}})
                for e, spec in EVENTS.items():
                    records.append({"location": str(sensor), "method": m, "event_name": e, "daily": daily,
                                    "event": happened[e]["exceeds"], "first_hour": happened[e]["first_hour"],
                                    "warned": event_occurs(p, *spec)["exceeds"]})
        i0 += len(js)

    df = pd.DataFrame(rows)
    per_sensor = df.groupby(["sensor", "method"]).mean().unstack("method")
    overall = df.drop(columns="sensor").groupby("method").mean()
    methods = list(overall.index)
    per_sensor.to_csv(out_dir / "test_mae_per_sensor.csv")

    warnings = {}
    for e in EVENTS:
        warnings[e] = {}
        for scope in ("all", "daily"):
            sel = [r for r in records if r["event_name"] == e and (scope == "all" or r["daily"])]
            one_method = [r for r in sel if r["method"] == methods[0]]
            warnings[e][scope] = {
                "n": len(one_method),
                "base_rate": float(np.mean([r["event"] for r in one_method])) if one_method else None,
                "scores": {m: score(r for r in sel if r["method"] == m) for m in methods},
                "scores_by_sensor": {m: score_by(r for r in sel if r["method"] == m) for m in methods},
            }
    results = {"mae_overall": overall.to_dict(), "warnings": warnings, "n_windows": len(ds),
               "events": {e: {"stat": st, "hours": h, "limit": lim} for e, (st, h, lim) in EVENTS.items()},
               "issue_hours": TEST_ISSUE_HOURS}
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, default=str)

    print("\nTEST (Dec 2025 - Feb 2026), MAE in µg/m³, averaged over forecasts:")
    print(overall.round(2).to_string())
    for h in HORIZONS:
        wins = (per_sensor[f"mae_{h}h"].idxmin(axis=1) == "fm_llm").sum()
        print(f"  {h}h: FM-LLM is the best method at {wins}/{len(per_sensor)} sensors")
    for e, (st, h, lim) in EVENTS.items():
        for scope in ("all", "daily"):
            w = warnings[e][scope]
            br = f"{w['base_rate']:.0%}" if w["base_rate"] is not None else "n/a"
            print(f"\nWarnings: {e} ({st} of next {h} h >= {lim}), {scope}: n={w['n']}, happened in {br}")
            for m in methods:
                sc = w["scores"][m]
                print(f"  {m:14s} hits={sc['hits']:4d} misses={sc['misses']:4d} false_alarms={sc['false_alarms']:4d}"
                      f"  hit_rate={sc['hit_rate']}  FAR={sc['false_alarm_ratio']}  CSI={sc['csi']}")


if __name__ == "__main__":
    main()
