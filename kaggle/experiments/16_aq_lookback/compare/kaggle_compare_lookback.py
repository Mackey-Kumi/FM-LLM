"""
Experiment 16, part 2: the fair comparison of the look-back variants.

Each training run scored itself on its own test windows, and those differ (a 56-day window
needs a longer clean stretch than a 16-day one). Here every model (run 2 = 28 days, L16,
L16_gap7, L56) forecasts, and only the forecasts that ALL four could make are scored: the
same sensors at the same issue times. Periods, as in experiment 14 (rule A):

  test           Harmattan test, Dec 2025 - Feb 2026
  post_trained   Mar - Oct 2026, the sensors the models were trained on
  post_new       Mar - Oct 2026, sensors installed after training

Runs on Kaggle (GPU) with the two training kernels attached as inputs, because their output
files can't be downloaded from the user's network; the summary is printed to the log as
JSON (and written to compare_summary.json).
"""

import json
import os
import sys
from pathlib import Path


def find_bundle(inputs=Path("/kaggle/input")):
    """
    Args:
        inputs: where Kaggle mounts attached datasets and kernel outputs.

    Returns:
        The folder holding fmllm/ (added to sys.path).
    """
    roots = [i.parent.parent for i in inputs.rglob("fmllm/__init__.py")] if inputs.exists() else []
    # attached kernel outputs can hold OLD copies of fmllm: use the data bundle's copy first
    roots.sort(key=lambda r: "fm-llm-accra-aq" not in str(r))
    here = Path(__file__).resolve().parents
    if len(here) > 4 and (here[4] / "fmllm" / "__init__.py").exists():   # the local repo (not on Kaggle)
        roots.append(here[4])
    sys.path.insert(0, str(roots[0]))
    print("fmllm from:", roots[0], flush=True)
    return roots[0]


def set_hf_token(inputs=Path("/kaggle/input")):
    """
    Put the Hugging Face token (any hf_token.txt under the inputs) in HF_TOKEN, so every
    Hugging Face call uses it. Prints no secret.

    Args:
        inputs: where the token dataset is mounted.
    """
    found = next(inputs.rglob("hf_token.txt"), None) if inputs.exists() else None
    if found:
        os.environ["HF_TOKEN"] = found.read_text(encoding="utf-8").strip()
    print("HF token file found:", found is not None, flush=True)


find_bundle()
set_hf_token()

# pylint: disable=wrong-import-position,wrong-import-order
import numpy as np
import pandas as pd
import torch

from fmllm.aq import EVENTS, POST_TEST, SPLIT, build_period, new_sensor_ids
from fmllm.evaluation import collect_records
from fmllm.model import LlamaBackbone, load_checkpoint
# pylint: enable=wrong-import-position,wrong-import-order

N_BOOT = 2000
REFERENCE = "L28 (run 2)"


def find_checkpoints(inputs=Path("/kaggle/input")):
    """
    Args:
        inputs: where the training kernels' outputs are mounted.

    Returns:
        {model name: checkpoint path}: run 2 (a checkpoint with no run.json beside it) and the
        experiment 16 runs (named by the folder they were written to).
    """
    found = {}
    for ckpt in inputs.rglob("aq_pm25_best.pt"):
        found[ckpt.parent.name if (ckpt.parent / "run.json").exists() else REFERENCE] = ckpt
    return dict(sorted(found.items()))


def find_csv(inputs=Path("/kaggle/input")):
    """
    Args:
        inputs: where the data bundle is mounted.

    Returns:
        Path to pm25_hourly.csv.
    """
    return next(inputs.rglob("pm25_hourly.csv"))


def paired_bootstrap(a, b, rng):
    """
    Day-level paired bootstrap of the difference in average error.

    Args:
        a, b: DataFrames with columns day and err (one row per forecast, same forecasts).
        rng: numpy Generator.

    Returns:
        (difference a - b, 2.5th percentile, 97.5th percentile).
    """
    da, db = a.groupby("day")["err"].agg(["sum", "count"]), b.groupby("day")["err"].agg(["sum", "count"])
    days = da.index.to_numpy()
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.choice(days, size=len(days), replace=True)
        diffs.append(da.loc[pick, "sum"].sum() / da.loc[pick, "count"].sum()
                     - db.loc[pick, "sum"].sum() / db.loc[pick, "count"].sum())
    est = a["err"].mean() - b["err"].mean()
    return float(est), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def score_group(rows, keep):
    """
    Score one group on the forecasts every model made.

    Args:
        rows: {model: collect_records() DataFrame}.
        keep: function(DataFrame) -> bool mask selecting the group's rows.

    Returns:
        Dict with the number of shared forecasts and sensors, and per model: mae_24h, mae_48h,
        sensors where it beats same-as-yesterday, and the bootstrap difference against run 2.
    """
    fm = {}
    for name, df in rows.items():
        d = df[keep(df)]
        fm[name] = d[d.method == "fm_llm"].set_index(["sensor", "issue_time"])
    shared = set.intersection(*(set(d.index) for d in fm.values()))
    ref_rows = rows[REFERENCE][keep(rows[REFERENCE])].set_index(["sensor", "issue_time"])
    yesterday = ref_rows[ref_rows.method == "daily_naive"].loc[sorted(shared)]
    out = {"forecasts": len(shared), "sensors": len({s for s, _ in shared}),
           "same_as_yesterday": {"mae_24h": round(float(yesterday.mae_24h.mean()), 3),
                                 "mae_48h": round(float(yesterday.mae_48h.mean()), 3)}, "models": {}}
    rng = np.random.default_rng(0)
    ref = fm[REFERENCE].loc[sorted(shared)]
    for name, d in fm.items():
        d = d.loc[sorted(shared)]
        per = pd.DataFrame({"model": d.mae_24h.groupby(level=0).mean(),
                            "yday": yesterday.mae_24h.groupby(level=0).mean()})
        entry = {"mae_24h": round(float(d.mae_24h.mean()), 3), "mae_48h": round(float(d.mae_48h.mean()), 3),
                 "beats_yesterday_at": f"{int((per.model < per.yday).sum())}/{len(per)} sensors"}
        if name != REFERENCE:
            a = pd.DataFrame({"day": d["day"].to_numpy(), "err": d.mae_24h.to_numpy()})
            b = pd.DataFrame({"day": ref["day"].to_numpy(), "err": ref.mae_24h.to_numpy()})
            ok = a.err.notna() & b.err.notna()
            est, lo, hi = paired_bootstrap(a[ok], b[ok], rng)
            entry["vs_run2_mae_24h"] = {"diff": round(est, 3), "ci95": [round(lo, 3), round(hi, 3)]}
        out["models"][name] = entry
    return out


def main():
    """Forecast every period with every model, keep the shared forecasts, print and save the comparison."""
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    ckpts = find_checkpoints()
    print("models:", {k: str(v) for k, v in ckpts.items()}, flush=True)
    csv = find_csv()
    backbone = LlamaBackbone(device)
    rows = {"test": {}, "post": {}}
    new = None
    for name, path in ckpts.items():
        gap = int(torch.load(path, map_location="cpu").get("gap", 0))
        model = load_checkpoint(path, backbone, device)
        if device.type == "cuda":
            model.set_decoder_dtype(torch.bfloat16)
        periods = {"test": build_period(csv, model.config, SPLIT["val_end"], SPLIT["test_end"], gap=gap),
                   "post": build_period(csv, model.config, POST_TEST["start"], POST_TEST["end"], gap=gap)}
        if new is None:
            new = {str(s) for s in new_sensor_ids(periods["post"].raw)}
        for p, split in periods.items():
            rows[p][name] = collect_records(model, split.test, device, EVENTS)
            rows[p][name]["day"] = pd.to_datetime(rows[p][name]["day"])
            print(f"{name} {p}: {len(split.test)} forecasts (seq_len {model.config.seq_len}, gap {gap})", flush=True)
    summary = {
        "test": score_group(rows["test"], lambda d: np.ones(len(d), bool)),
        "post_trained": score_group(rows["post"], lambda d: ~d.sensor.isin(new)),
        "post_new": score_group(rows["post"], lambda d: d.sensor.isin(new)),
    }
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).parent / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare_summary.json").write_text(json.dumps(summary, indent=1))
    print("COMPARE_SUMMARY_JSON " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
