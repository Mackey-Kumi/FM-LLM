"""
Fit the likelihood calibration on validation and check it on test and post-test,
from experiment 15's ensemble files. Runs on CPU in seconds.

    python scripts/calibrate.py
    python scripts/calibrate.py --results kaggle/experiments/15_aq_ensemble_calibration/results

Writes, into the results folder:
  calibration_scores.csv   every method x period x outlook day: Brier, Brier skill, 80% coverage,
                           CSI at 50%, reliability table
  calibration.json         what the forecast service needs (primary method): the validation ratios
                           for today / tomorrow and the hourly band ratios
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
import pandas as pd

from fmllm.aq import new_sensor_ids
from fmllm.calibration import evaluate_all, fit_all, fit_hourly_band, hourly_coverage, load_ensemble
from fmllm.data import load_hourly
# pylint: enable=wrong-import-position

RESULTS = REPO / "kaggle" / "experiments" / "15_aq_ensemble_calibration" / "results"


def main():
    """Fit on validation, score test / post-test (all and new sensors), save the outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS), help="folder holding ensemble_{val,test,post}.npz")
    args = ap.parse_args()
    folder = Path(args.results)
    ens = {p: load_ensemble(folder / f"ensemble_{p}.npz") for p in ("val", "test", "post")}
    fitted = fit_all(ens["val"])
    low_r, high_r = fit_hourly_band(ens["val"])
    new = {str(s) for s in new_sensor_ids(load_hourly(REPO / "data" / "raw" / "openaq_accra" / "pm25_hourly.csv"))}

    frames = []
    for name, a, sensors in (("test (Harmattan)", ens["test"], None), ("post Mar-Oct 2026", ens["post"], None),
                             ("post, new sensors", ens["post"], new)):
        df = evaluate_all(a, fitted, sensors)
        df.insert(0, "checked_on", name)
        frames.append(df)
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(folder / "calibration_scores.csv", index=False)

    pd.set_option("display.width", 200)
    print("fitted on validation:", {p: {"scale_factor": f["scale_factor"], "n_ratios": len(f["ratios"]),
                                        "base_rate": round(f["base_rate"], 3)} for p, f in fitted.items()})
    print(scores.drop(columns=["reliability"]).to_string(index=False))
    for name, a in (("test", ens["test"]), ("post", ens["post"])):
        print(f"hourly 80% band coverage on {name}: {hourly_coverage(a, low_r, high_r):.3f}")

    calib = {"method": "ratio", "fitted_on": "validation Oct-Nov 2025", "limit": 35.0,
             "ratios": {p: fitted[p]["ratios"].round(4).tolist() for p in fitted},
             "hourly_band": {"low_ratio": low_r.round(4).tolist(), "high_ratio": high_r.round(4).tolist()}}
    with open(folder / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(calib, f)
    print(f"saved {folder / 'calibration_scores.csv'} and calibration.json")


if __name__ == "__main__":
    main()
