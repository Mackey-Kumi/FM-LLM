"""
FM-LLM on Accra PM2.5: the first month live (experiment 17, runs locally on CPU).

The forecast service went live on 28 Oct 2026. This scores every forecast run 2 would
have issued since then (the service makes the same forecasts from the same data) against
what really happened, the same way as experiment 14:

  live_all       every sensor
  live_trained   the 24 sensors the model was trained on
  live_new       the sensors it never saw in training

with rule A (>= 75% of the 48 scored hours real, only real hours scored), issue times
00/06/12/18, the simple rules as yardstick and the day-level bootstrap. Plus:

  calibration_live.json   are Method B's probabilities still honest on live data?
  demo_weekend.csv        the forecasts issued at 12:00 on Fri 30 Oct (the demo), scored
                          (results/, judged wherever a day has >= 12 real hours)

Sensors reported far fewer hours in November (Clarity ~55% of hours, against ~74% in
Sep-Oct), so two scoring rules are reported, each in its own folder:
  results/ruleA/     the pre-declared rule (>= 75% real hours; a day needs >= 18 real hours)
  results/relaxed/   sensitivity check (>= 50%; a day needs >= 12 real hours)
Only real hours are ever scored; the rule only decides which forecasts are judged.

    python evaluate_live_month.py --fetch     # first download the readings up to now
    python evaluate_live_month.py             # score both rules (about 15 min on a laptop CPU)
"""

import argparse
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

from fmllm.aq import EVENTS, build_period
from fmllm.calibration import RatioCalibration, load_ensemble, outcomes, score_probabilities
from fmllm.evaluation import collect_records, print_summary, write_results
from fmllm.forecast import forecast
from fmllm.model import LlamaBackbone, load_checkpoint
from fmllm.payloads import level_of
# pylint: enable=wrong-import-position

LIVE_START = "2026-10-28"                       # the service went live
DEMO_ISSUE = pd.Timestamp("2026-10-30 12:00")   # forecasts on screen at the demo
HISTORY = REPO / "data" / "raw" / "openaq_accra"
LIVE_DATA = REPO / "data" / "raw" / "openaq_accra_live"
RUN2 = REPO / "kaggle" / "experiments" / "12_aq_first_train" / "results_run2"
EXP15 = REPO / "kaggle" / "experiments" / "15_aq_ensemble_calibration" / "results"
OUT = HERE / "results"


def fetch():
    """
    Download every sensor's readings since the history ends, with the live service's own
    fetch code, into data/raw/openaq_accra_live/ (the downloaded history is not touched).
    """
    from service.live import LiveService   # pylint: disable=import-outside-toplevel
    LIVE_DATA.mkdir(parents=True, exist_ok=True)
    svc = LiveService(data_dir=HISTORY, work_dir=LIVE_DATA, fetch=True)
    svc.long = pd.read_csv(HISTORY / "pm25_hourly.csv")
    svc.meta = pd.read_csv(HISTORY / "sensors.csv")
    n = svc.update_readings()          # writes LIVE_DATA/pm25_hourly.csv
    svc.meta.to_csv(LIVE_DATA / "sensors.csv", index=False)
    print(f"fetched {n} readings; data now ends {svc.raw.index.max()}")


def day_probabilities(model, ds, calibration, trained, min_real_hours=18):  # pylint: disable=too-many-locals
    """
    Method B on every forecast: chance that each 24-hour period's average reaches 35, its
    80% range, and what really happened.

    Args:
        model: run 2.
        ds: the live SensorWindows.
        calibration: RatioCalibration (calibration.json of experiment 15).
        trained: ids of the sensors the model was trained on.
        min_real_hours: real hours a 24-hour period needs before its average is judged.

    Returns:
        DataFrame, one row per (forecast, period): sensor, issue_time, period, in_training,
        mean_forecast, prob, low, high, actual (NaN if fewer than min_real_hours real hours).
    """
    rows = []
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    i0 = 0
    for hist, _, js, starts in loader:
        pred = forecast(model, hist, 48).float().numpy()[:, 0]
        for k, (j, s) in enumerate(zip(js.tolist(), starts.tolist())):
            sensor = ds.sensors[j]
            mean, std = ds.scale.get(sensor, (0.0, 1.0))
            point = pred[k] * std + mean
            f0 = s + ds.seq_len + ds.gap
            truth = ds.truth[f0: f0 + 48, j]
            for period, sl in (("today", slice(0, 24)), ("tomorrow", slice(24, 48))):
                d = calibration.day(float(point[sl].mean()), period)
                real = truth[sl]
                rows.append({"sensor": str(sensor), "issue_time": ds.issue_time(i0 + k), "period": period,
                             "in_training": int(sensor) in trained, "mean_forecast": float(point[sl].mean()),
                             "prob": d["prob_exceed"], "low": d["low"], "high": d["high"],
                             "actual": (float(np.nanmean(real)) if np.isfinite(real).sum() >= min_real_hours
                                        else np.nan)})
        i0 += len(js)
    return pd.DataFrame(rows)


def main():
    """Fetch (optional), then score the live month: tables, bootstrap, calibration, demo weekend."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="download readings up to now first")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    if args.fetch:
        fetch()
    csv = LIVE_DATA / "pm25_hourly.csv"
    trained = set(json.loads((RUN2 / "training_sensors.json").read_text())["training_sensors"])
    torch.manual_seed(0)
    model = load_checkpoint(RUN2 / "aq_pm25_best.pt", LlamaBackbone("cpu"), "cpu")
    for rule, frac, hours in (("ruleA", 0.75, 18), ("relaxed", 0.5, 12)):
        score(model, csv, trained, rule, frac, hours, args.n_boot)
    demo_weekend(model, csv, trained)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def score(model, csv, trained, rule, min_fraction, min_hours, n_boot):
    """
    Score the live month under one rule and write results/<rule>/.

    Args:
        model: run 2.
        csv: the live pm25_hourly.csv.
        trained: ids of the sensors the model was trained on.
        rule: folder name ("ruleA" or "relaxed").
        min_fraction: share of the 48 scored hours that must be real.
        min_hours: real hours a 24-hour period needs to be judged.
        n_boot: bootstrap resamples.
    """
    period = build_period(csv, model.config, LIVE_START, min_real_fraction=min_fraction)
    ds = period.test
    print(f"[{rule}] live period {LIVE_START} -> {period.raw.index.max()}: {len(ds)} forecasts at "
          f"{len({j for j, _ in ds.items})} sensors", flush=True)

    # 1. accuracy, three groups (as experiment 14)
    rows = collect_records(model, ds, torch.device("cpu"), EVENTS, min_real_hours=min_hours)
    is_trained = rows["sensor"].astype(int).isin(trained)
    groups = {"live_all": rows, "live_trained": rows[is_trained], "live_new": rows[~is_trained]}
    out = OUT / rule
    for name, df in groups.items():
        print(f"\n{'=' * 25} {name} ({df['sensor'].nunique()} sensors, {int((df['method'] == 'fm_llm').sum())} "
              f"forecasts) {'=' * 25}")
        (out / name).mkdir(parents=True, exist_ok=True)
        summary, per_sensor, boot = write_results(df, EVENTS, out / name, n_boot=n_boot,
                                                  extra={"group": name, "model": "run 2", "rule": rule,
                                                         "min_real_fraction": min_fraction, "min_real_hours": min_hours,
                                                         "period": [LIVE_START, str(period.raw.index.max())]})
        print_summary(summary, per_sensor, boot, EVENTS)

    # 2. Method B probabilities on live data
    calibration = RatioCalibration.load(EXP15 / "calibration.json")
    base = outcomes(load_ensemble(EXP15 / "ensemble_val.npz"))
    probs = day_probabilities(model, ds, calibration, trained, min_hours)
    probs.to_csv(out / "live_probabilities.csv", index=False)
    cal = {}
    for group, sel in (("all", probs), ("trained", probs[probs.in_training]), ("new", probs[~probs.in_training])):
        for p in ("today", "tomorrow"):
            g = sel[sel.period == p]
            seen = base[p][np.isfinite(base[p])]                 # validation outcomes with enough real hours
            base_rate = float((seen >= calibration.limit).mean())
            cal[f"{group}/{p}"] = score_probabilities(g.prob.to_numpy(), g.low.to_numpy(), g.high.to_numpy(),
                                                      g.actual.to_numpy(), base_rate)
    (out / "calibration_live.json").write_text(json.dumps(cal, indent=1))
    print("\nMethod B on live data:", json.dumps({k: {x: v[x] for x in ("n", "base_rate_here", "brier_skill",
                                                                         "coverage_80")} for k, v in cal.items()},
                                                  indent=1))



def demo_weekend(model, csv, trained, min_hours=12):
    """
    Score the forecasts that were on screen at the demo: every sensor's forecast issued at
    12:00 on Fri 30 Oct, whatever its share of real hours (30 Oct itself had few readings);
    each 24-hour period is judged where it has at least `min_hours` real hours.

    Args:
        model: run 2.
        csv: the live pm25_hourly.csv.
        trained: ids of the sensors the model was trained on.
        min_hours: real hours a 24-hour period needs to be judged.

    Writes:
        results/demo_weekend.csv: per sensor and period, forecast vs what happened.
    """
    calibration = RatioCalibration.load(EXP15 / "calibration.json")
    period = build_period(csv, model.config, str(DEMO_ISSUE), str(DEMO_ISSUE + pd.Timedelta(hours=97)),
                          min_real_fraction=0.0)
    probs = day_probabilities(model, period.test, calibration, trained, min_hours)
    demo = probs[probs.issue_time == DEMO_ISSUE].copy()
    demo["level"] = demo.mean_forecast.map(level_of)
    demo["actual_level"] = demo.actual.map(lambda x: level_of(x) if np.isfinite(x) else None)
    names = dict(zip(pd.read_csv(LIVE_DATA / "sensors.csv").sensor_id.astype(str),
                     pd.read_csv(LIVE_DATA / "sensors.csv").name))
    demo.insert(1, "name", demo.sensor.map(names))
    OUT.mkdir(parents=True, exist_ok=True)
    demo.to_csv(OUT / "demo_weekend.csv", index=False)
    scored = demo[demo.actual_level.notna()]
    print(f"\ndemo weekend: {len(scored)} sensor-days scored, level right at "
          f"{int((scored.level == scored.actual_level).sum())}, mean |error| of the daily average "
          f"{(scored.mean_forecast - scored.actual).abs().mean():.1f} µg/m³")


if __name__ == "__main__":
    main()
