"""
Recalibrating Method B on live data (experiment 18, runs locally on CPU).

Experiment 17 found that in November 2026 (the clean season) the probabilities learned on
Oct-Nov 2025 overstate the risk: "39%" happened ~20% of the time. Method B's ratios
(actual / forecast of the 24-hour average) are re-learned from the live month, and the
test is kept honest by splitting it in time:

  fit    forecasts issued 28 Oct - 10 Nov 2026   (ratios and hourly band learned here)
  test   forecasts issued 13 Nov 2026 onwards    (2-day buffer: no forecast window is in both)

Both calibrations, old (validation Oct-Nov 2025) and new (fit half), are scored on the same
test forecasts. Days are judged with >= 18 real hours (the pre-declared rule) and, because
sensors reported only ~55% of hours in November, also with >= 12.

Writes results/:
  live_forecasts.npz        every live forecast: point (N, 48), truth (N, 48), sensor, issue_time
  recalibration_scores.csv  old vs new on the test half: Brier skill, coverage, reliability
  calibration_nov2026.json  ratios + hourly band re-learned on the WHOLE live month, for the service

    python recalibrate_live.py        # needs data/raw/openaq_accra_live (experiment 17 --fetch)
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
import numpy as np
import pandas as pd
import torch

from fmllm.aq import build_period
from fmllm.calibration import (LIMIT, PERIODS, RatioCalibration, fit_hourly_band, fit_ratios, hourly_coverage,
                               outcomes, ratio_method, score_probabilities)
from fmllm.forecast import forecast
from fmllm.model import LlamaBackbone, load_checkpoint
# pylint: enable=wrong-import-position

LIVE_START = "2026-10-28"
FIT_END = pd.Timestamp("2026-11-11")      # fit: issued before this
TEST_START = pd.Timestamp("2026-11-13")   # test: issued from this (48-hour windows never straddle both)
LIVE_CSV = REPO / "data" / "raw" / "openaq_accra_live" / "pm25_hourly.csv"
RUN2 = REPO / "kaggle" / "experiments" / "12_aq_first_train" / "results_run2"
OLD = REPO / "kaggle" / "experiments" / "15_aq_ensemble_calibration" / "results" / "calibration.json"
OUT = HERE / "results"


def live_forecasts(model, trained):  # pylint: disable=too-many-locals
    """
    Run 2's plain 48-hour forecast for every live window (issue hours 00/06/12/18), whatever
    its share of real hours: outcomes are judged per period afterwards.

    Args:
        model: run 2.
        trained: ids of the sensors the model was trained on.

    Returns:
        Dict of arrays: point (N, 48) and truth (N, 48) in µg/m³ (truth NaN where no reading),
        sensor (N,), issue_time (N,), in_training (N,).
    """
    ds = build_period(LIVE_CSV, model.config, LIVE_START, min_real_fraction=0.0).test
    points, truths, sensors, issued = [], [], [], []
    i0 = 0
    for hist, _, js, starts in torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False):
        pred = forecast(model, hist, 48).float().numpy()[:, 0]
        for k, (j, s) in enumerate(zip(js.tolist(), starts.tolist())):
            mean, std = ds.scale.get(ds.sensors[j], (0.0, 1.0))
            points.append(pred[k] * std + mean)
            f0 = s + ds.seq_len + ds.gap
            truths.append(ds.truth[f0: f0 + 48, j])
            sensors.append(int(ds.sensors[j]))
            issued.append(ds.issue_time(i0 + k))
        i0 += len(js)
    sensors = np.array(sensors)
    return {"point": np.array(points), "truth": np.array(truths), "sensor": sensors,
            "issue_time": np.array(issued, dtype="datetime64[ns]"), "in_training": np.isin(sensors, list(trained))}


def subset(a, mask):
    """
    Args:
        a: dict of per-forecast arrays.
        mask: (N,) bool.

    Returns:
        The same dict restricted to the masked forecasts.
    """
    return {k: v[mask] for k, v in a.items()}


def main():  # pylint: disable=too-many-locals
    """Forecast the live month, fit on the first half, compare old and new on the second, write the files."""
    OUT.mkdir(parents=True, exist_ok=True)
    trained = set(json.loads((RUN2 / "training_sensors.json").read_text())["training_sensors"])
    model = load_checkpoint(RUN2 / "aq_pm25_best.pt", LlamaBackbone("cpu"), "cpu")
    a = live_forecasts(model, trained)
    np.savez_compressed(OUT / "live_forecasts.npz", **a)
    when = pd.to_datetime(a["issue_time"])
    fit, test = subset(a, when < FIT_END), subset(a, when >= TEST_START)
    print(f"{len(a['point'])} live forecasts: fit {len(fit['point'])}, test {len(test['point'])}")

    old = RatioCalibration.load(OLD)
    rows = []
    for min_real in (18, 12):
        new_ratios = {p: fit_ratios(fit, p, min_real) for p in PERIODS}
        fit_y = outcomes(fit, min_real)
        for group, g in (("all", test), ("trained", subset(test, test["in_training"])),
                         ("new", subset(test, ~test["in_training"]))):
            y = outcomes(g, min_real)
            for p in PERIODS:
                seen = fit_y[p][np.isfinite(fit_y[p])]
                base_rate = float((seen >= LIMIT).mean())          # "always say the usual rate", from the fit half
                for name, ratios in (("old (Oct-Nov 2025)", old.ratios[p]), ("new (live fit half)", new_ratios[p])):
                    prob, low, high = ratio_method(g, p, ratios)
                    sc = score_probabilities(prob, low, high, y[p], base_rate)
                    sc["reliability"] = "; ".join(f"{r['bin']}: said {r['mean_prob']:.2f}, got {r['observed']:.2f} "
                                                  f"(n={r['n']})" for r in sc["reliability"])
                    rows.append({"min_real_hours": min_real, "sensors": group, "period": p, "calibration": name, **sc})
    band_new = fit_hourly_band(fit)
    scores = pd.DataFrame(rows)
    scores.to_csv(OUT / "recalibration_scores.csv", index=False)
    cov = {"old": hourly_coverage(test, old.low_ratio, old.high_ratio),
           "new": hourly_coverage(test, *band_new)}
    print(scores[["min_real_hours", "sensors", "period", "calibration", "n", "base_rate_here", "brier_skill",
                  "coverage_80"]].to_string(index=False))
    print(f"hourly 80% band coverage on the test half: old {cov['old']:.3f}, new {cov['new']:.3f}")
    print("\nreliability, all sensors, today, >= 18 real hours:")
    for r in rows:
        if r["min_real_hours"] == 18 and r["sensors"] == "all" and r["period"] == "today":
            print(f"  {r['calibration']}: {r['reliability']}")

    # for the service: the same method re-learned on the whole live month (pre-declared 18-hour rule)
    low_r, high_r = fit_hourly_band(a)
    calib = {"method": "ratio", "fitted_on": f"live forecasts {LIVE_START} to {when.max():%Y-%m-%d}", "limit": LIMIT,
             "ratios": {p: np.round(fit_ratios(a, p), 4).tolist() for p in PERIODS},
             "hourly_band": {"low_ratio": low_r.round(4).tolist(), "high_ratio": high_r.round(4).tolist()},
             "test_half_hourly_coverage": cov}
    (OUT / "calibration_nov2026.json").write_text(json.dumps(calib), encoding="utf-8")
    print(f"saved {OUT / 'calibration_nov2026.json'}")


if __name__ == "__main__":
    main()
