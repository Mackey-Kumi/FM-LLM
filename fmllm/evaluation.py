"""
Test-set evaluation: one row per (forecast, method), the summary tables, and a
bootstrap for confidence intervals.

  collect_records  run the model and the simple rules on every test forecast -> DataFrame
  summarize        accuracy and warning scores, overall and per sensor (from that DataFrame)
  bootstrap        95% intervals for every score, and for FM-LLM minus each baseline
  write_results    save the forecast table, the summary and the bootstrap

Why the bootstrap resamples whole DAYS: all sensors share the same weather, so
a dusty Harmattan day affects every sensor at once and the forecasts issued on
the same day are not independent. Resampling single forecasts would make the
intervals look narrower (more certain) than they really are.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .events import event_occurs, score, score_by
from .forecast import BASELINES, baseline_forecast, forecast

MODEL = "fm_llm"
SCOPES = ("all", "daily")   # all = every test forecast (4 a day); daily = the 00:00 forecast only


@torch.no_grad()
def collect_records(model, ds, device, events: Dict, horizons: Sequence[int] = (24, 48),
                    max_batches: Optional[int] = None) -> pd.DataFrame:
    """
    Forecast every test window with the model and each simple rule, and record how each did.

    Args:
        model: trained FMLLM (eval mode is set here). Its output is on the dataset's scaled
            scale; it is converted back to µg/m³ with the dataset's per-sensor scale.
        ds: SensorWindows of the test period.
        device: torch device the model is on.
        events: {name: (stat, hours, limit)}, e.g. fmllm.aq.EVENTS.
        horizons: hours ahead to score the error at, e.g. (24, 48).
        max_batches: stop after this many batches (smoke runs). None = all.

    Returns:
        DataFrame, one row per (forecast, method), with columns
            sensor, issue_time, day (issue date), daily (issued at 00:00), method,
            mae_{h}h for each horizon (µg/m³),
            and for each event e: e__event (did it happen), e__warned (did this method
            forecast it), e__first_hour (first hour reaching the limit, "max" events only).
    """
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    horizon = max(horizons)
    rows = []
    i0 = 0  # index of the batch's first sample, to look up issue times
    for b, (hist, _, js, starts) in enumerate(loader):  # truth comes unscaled from ds.values below
        if max_batches and b >= max_batches:
            break
        pred = forecast(model, hist.to(device), horizon).float().cpu().numpy()[:, 0]
        for k, (j, s) in enumerate(zip(js.tolist(), starts.tolist())):
            sensor = ds.sensors[j]
            issued = ds.issue_time(i0 + k)
            mean, std = ds.scale.get(sensor, (0.0, 1.0))
            raw_hist = ds.values[s: s + ds.seq_len, j]
            truth = ds.values[s + ds.seq_len: s + ds.seq_len + horizon, j]
            methods = {MODEL: pred[k] * std + mean}
            methods.update({m: baseline_forecast(raw_hist[:, None], m, horizon)[:, 0] for m in BASELINES})
            happened = {e: event_occurs(truth, *spec) for e, spec in events.items()}
            for m, p in methods.items():
                row = {"sensor": str(sensor), "issue_time": issued, "day": issued.normalize(),
                       "daily": issued.hour == 0, "method": m}
                row.update({f"mae_{h}h": float(np.abs(p[:h] - truth[:h]).mean()) for h in horizons})
                for e, spec in events.items():
                    row[f"{e}__event"] = happened[e]["exceeds"]
                    row[f"{e}__warned"] = event_occurs(p, *spec)["exceeds"]
                    row[f"{e}__first_hour"] = happened[e]["first_hour"]
                rows.append(row)
        i0 += len(js)
    return pd.DataFrame(rows)


def _event_records(df: pd.DataFrame, event: str):
    """
    Args:
        df: rows of one method (from collect_records).
        event: event name.

    Returns:
        List of dicts in the format fmllm.events.score expects.
    """
    return [{"location": s, "event": bool(ev), "warned": bool(w), "first_hour": fh}
            for s, ev, w, fh in zip(df["sensor"], df[f"{event}__event"], df[f"{event}__warned"],
                                    df[f"{event}__first_hour"])]


def summarize(df: pd.DataFrame, events: Dict):
    """
    The tables for the thesis: accuracy and warning scores, overall and per sensor.

    Args:
        df: output of collect_records.
        events: the events that were recorded (names must match df's columns).

    Returns:
        summary: dict with
            mae_overall: {mae_{h}h: {method: mean error}},
            warnings: {event: {scope: {n, base_rate, scores: {method: score()},
                                       scores_by_sensor: {method: score_by()}}}},
            n_forecasts: number of test forecasts (per method).
        per_sensor: DataFrame, rows = sensors, columns = (mae_{h}h, method).
    """
    mae_cols = [c for c in df.columns if c.startswith("mae_")]
    overall = df.groupby("method")[mae_cols].mean()
    per_sensor = df.groupby(["sensor", "method"])[mae_cols].mean().unstack("method")
    methods = list(overall.index)
    warnings = {}
    for e in events:
        warnings[e] = {}
        for scope in SCOPES:
            sel = df if scope == "all" else df[df["daily"]]
            ref = sel[sel["method"] == methods[0]]
            warnings[e][scope] = {
                "n": len(ref),
                "base_rate": float(ref[f"{e}__event"].mean()) if len(ref) else None,
                "scores": {m: score(_event_records(sel[sel["method"] == m], e)) for m in methods},
                "scores_by_sensor": {m: score_by(_event_records(sel[sel["method"] == m], e)) for m in methods},
            }
    summary = {"mae_overall": overall.to_dict(), "warnings": warnings,
               "n_forecasts": int((df["method"] == methods[0]).sum())}
    return summary, per_sensor


def bootstrap(df: pd.DataFrame, events: Dict, n_boot: int = 2000, level: float = 0.95,
              seed: int = 0, reference: str = MODEL) -> pd.DataFrame:
    """
    Confidence intervals by resampling whole test days with replacement (see the module docstring).

    For every metric and method we report the estimate and its interval, and the
    PAIRED difference reference - method (computed on the same resampled days), with
    its interval and how often the reference came out better. If the difference's
    interval excludes 0, the gap is unlikely to be luck.

    Metrics:
        mae_{h}h (lower is better): all forecasts.
        for each event, in both scopes: csi and hit_rate (higher is better),
            far = false-alarm ratio (lower is better).

    Args:
        df: output of collect_records.
        events: the events recorded in df.
        n_boot: number of resamples.
        level: confidence level, e.g. 0.95.
        seed: random seed (the same seed gives the same intervals).
        reference: the method the others are compared with.

    Returns:
        DataFrame with columns metric, scope, method, estimate, ci_low, ci_high,
        diff (reference - method), diff_low, diff_high, ref_better (share of
        resamples in which the reference was better). For the reference's own rows the
        diff columns are NaN.
    """
    rng = np.random.default_rng(seed)
    days = np.array(sorted(df["day"].unique()))
    day_idx = {d: i for i, d in enumerate(days)}
    picks = rng.integers(0, len(days), size=(n_boot, len(days)))   # each row = one resampled test season
    counts = np.stack([np.bincount(p, minlength=len(days)) for p in picks])  # (n_boot, n_days)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    methods = sorted(df["method"].unique())

    def per_day(sub: pd.DataFrame, col: str) -> np.ndarray:
        """Sum of `col` per test day, as an array aligned with `days` (0 for days with no rows)."""
        out = np.zeros(len(days))
        sums = sub.groupby("day")[col].sum()
        out[[day_idx[d] for d in sums.index]] = sums.to_numpy(dtype=float)
        return out

    def ratio(num, den):
        """Elementwise num / den, NaN where den is 0."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    # metric -> (scope, better, function(sub_df) -> (point estimate, bootstrap array))
    specs = {}
    for col in [c for c in df.columns if c.startswith("mae_")]:
        def mae(sub, col=col):
            """Mean error of `col` in `sub`: (point estimate, one value per resample)."""
            s, n = per_day(sub, col), per_day(sub.assign(one=1), "one")
            return ratio(s.sum(), n.sum()), ratio(counts @ s, counts @ n)
        specs[f"{col}@all"] = ("all", "lower", mae)
    for e in events:
        def parts(sub, e=e):
            """Per-day hits, misses and false alarms of event `e` in `sub`."""
            ev, w = sub[f"{e}__event"].astype(bool), sub[f"{e}__warned"].astype(bool)
            return (per_day(sub.assign(x=ev & w), "x"), per_day(sub.assign(x=ev & ~w), "x"),
                    per_day(sub.assign(x=~ev & w), "x"))

        def csi(sub, parts=parts):
            """CSI = hits / (hits + misses + false alarms): (point estimate, one value per resample)."""
            h, m, f = parts(sub)
            return ratio(h.sum(), h.sum() + m.sum() + f.sum()), ratio(counts @ h, counts @ (h + m + f))

        def hit_rate(sub, parts=parts):
            """Hit rate = hits / (hits + misses): (point estimate, one value per resample)."""
            h, m, _ = parts(sub)
            return ratio(h.sum(), h.sum() + m.sum()), ratio(counts @ h, counts @ (h + m))

        def far(sub, parts=parts):
            """False-alarm ratio = false alarms / (hits + false alarms): (point, one value per resample)."""
            h, _, f = parts(sub)
            return ratio(f.sum(), h.sum() + f.sum()), ratio(counts @ f, counts @ (h + f))

        for scope in SCOPES:   # key = "metric@scope"
            specs[f"{e}:csi@{scope}"] = (scope, "higher", csi)
            specs[f"{e}:hit_rate@{scope}"] = (scope, "higher", hit_rate)
            specs[f"{e}:far@{scope}"] = (scope, "lower", far)

    rows = []
    for key, (scope, better, fn) in specs.items():
        metric = key.split("@")[0]
        sel = df if scope == "all" else df[df["daily"]]
        results = {m: fn(sel[sel["method"] == m]) for m in methods}
        ref_point, ref_boot = results[reference]
        for m in methods:
            point, boot = results[m]
            row = {"metric": metric, "scope": scope, "method": m, "estimate": float(point),
                   "ci_low": float(np.nanquantile(boot, lo_q)), "ci_high": float(np.nanquantile(boot, hi_q)),
                   "diff": np.nan, "diff_low": np.nan, "diff_high": np.nan, "ref_better": np.nan}
            if m != reference:
                d = ref_boot - boot
                row.update(diff=float(ref_point - point), diff_low=float(np.nanquantile(d, lo_q)),
                           diff_high=float(np.nanquantile(d, hi_q)),
                           ref_better=float(np.nanmean(d < 0) if better == "lower" else np.nanmean(d > 0)))
            rows.append(row)
    return pd.DataFrame(rows)


def write_results(df: pd.DataFrame, events: Dict, out_dir, extra: Optional[Dict] = None, n_boot: int = 2000):
    """
    Save everything an experiment reports about its test period.

    Args:
        df: output of collect_records.
        events: the events recorded in df.
        out_dir: folder to write into.
        extra: more fields for test_results.json (e.g. the split dates).
        n_boot: bootstrap resamples.

    Writes:
        test_forecasts.csv: df itself (one row per forecast and method; the raw material
            for any later analysis).
        test_mae_per_sensor.csv, test_results.json: the summary tables.
        test_bootstrap.csv: bootstrap().

    Returns:
        (summary, per_sensor, boot): the same tables as Python objects.
    """
    out_dir = Path(out_dir)
    summary, per_sensor = summarize(df, events)
    boot = bootstrap(df, events, n_boot=n_boot)
    df.to_csv(out_dir / "test_forecasts.csv", index=False)
    per_sensor.to_csv(out_dir / "test_mae_per_sensor.csv")
    boot.to_csv(out_dir / "test_bootstrap.csv", index=False)
    summary.update({"events": {e: {"stat": st, "hours": h, "limit": lim} for e, (st, h, lim) in events.items()},
                    **(extra or {})})
    with open(out_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    return summary, per_sensor, boot


def print_summary(summary: Dict, per_sensor: pd.DataFrame, boot: pd.DataFrame, events: Dict):
    """
    Print the main tables to the log.

    Args:
        summary, per_sensor, boot: from write_results.
        events: the events recorded.
    """
    mae = pd.DataFrame(summary["mae_overall"])
    print("\nTEST: MAE in µg/m³, averaged over forecasts:")
    print(mae.round(2).to_string())
    for col in mae.columns:
        wins = (per_sensor[col].idxmin(axis=1) == MODEL).sum()
        print(f"  {col}: FM-LLM is the best method at {wins}/{len(per_sensor)} sensors")
    for e, (st, h, lim) in events.items():
        for scope in SCOPES:
            w = summary["warnings"][e][scope]
            br = f"{w['base_rate']:.0%}" if w["base_rate"] is not None else "n/a"
            print(f"\nWarnings: {e} ({st} of next {h} h >= {lim}), {scope}: n={w['n']}, happened in {br}")
            for m, sc in w["scores"].items():
                print(f"  {m:14s} hits={sc['hits']:4d} misses={sc['misses']:4d} false_alarms={sc['false_alarms']:4d}"
                      f"  hit_rate={sc['hit_rate']}  FAR={sc['false_alarm_ratio']}  CSI={sc['csi']}")
    print("\nBOOTSTRAP (resampling test days; diff = FM-LLM minus method, 95% interval):")
    show = boot[boot["method"] != MODEL].copy()
    for c in ("estimate", "diff", "diff_low", "diff_high", "ref_better"):
        show[c] = show[c].round(3)
    print(show[["metric", "scope", "method", "estimate", "diff", "diff_low", "diff_high", "ref_better"]]
          .to_string(index=False))
