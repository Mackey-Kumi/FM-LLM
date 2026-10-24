"""
Calibrating the likelihood of a bad day ("today's average will reach 35 µg/m³").

Fitted on the validation period only, then checked on the test and post-test periods,
from the ensemble files that experiment 15 writes (fmllm.evaluation.collect_ensemble).

Methods (the primary one was chosen BEFORE looking at test results):
  raw       the dropout ensemble as it is: share of members >= 35. Known to be too
            confident (its spread is too narrow).
  scaled    Method A: stretch the ensemble around its median by one factor, fitted so
            that the 80% range covers 80% of validation outcomes.
  ratio     Method B (PRIMARY): learn the model's real errors on validation, as ratios
            actual / forecast of the 24-hour average. A new forecast F becomes the set of
            outcomes F x ratio_j; the probability is the share of them >= 35. Ratios suit
            PM2.5 because errors grow with the level.

Measures:
  Brier score      mean (probability - outcome)^2; lower is better, 0 is perfect.
  Brier skill      1 - Brier / Brier of always saying the validation base rate; > 0 = useful.
  reliability      in bins of stated probability, how often the event really happened.
  coverage         share of outcomes inside the stated 80% range (should be ~0.80).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

LIMIT = 35.0       # Ghana's 24-hour PM2.5 standard (GS 1236:2019)
MIN_REAL = 18      # an outlook period is judged only with >= 18 real hours (as in fmllm.evaluation)
DAY = 24
PERIODS = ("today", "tomorrow")


def load_ensemble(path) -> Dict[str, np.ndarray]:
    """
    Args:
        path: ensemble_<period>.npz from experiment 15.

    Returns:
        Its arrays as a dict (see fmllm.evaluation.collect_ensemble).
    """
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def outcomes(a: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    The real 24-hour averages for each outlook period, where enough hours were measured.

    Args:
        a: ensemble arrays.

    Returns:
        {"today": (N,), "tomorrow": (N,)} in µg/m³; NaN where fewer than MIN_REAL real hours.
    """
    out = {}
    for k, name in enumerate(PERIODS):
        block = a["truth"][:, k * DAY:(k + 1) * DAY]
        n_real = np.isfinite(block).sum(axis=1)
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=1)
        out[name] = np.where(n_real >= MIN_REAL, mean, np.nan)
    return out


def point_means(a: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Args:
        a: ensemble arrays.

    Returns:
        {"today": (N,), "tomorrow": (N,)}: the plain forecast's 24-hour averages.
    """
    return {name: a["point"][:, k * DAY:(k + 1) * DAY].mean(axis=1) for k, name in enumerate(PERIODS)}


# ---------------------------------------------------------------------------
# Methods: each returns (probability of >= LIMIT, 10th percentile, 90th percentile) per forecast
# ---------------------------------------------------------------------------
def raw_ensemble(a: Dict[str, np.ndarray], period: str):
    """
    Args:
        a: ensemble arrays.
        period: "today" or "tomorrow".

    Returns:
        (prob, low, high): (N,) arrays from the ensemble members' 24-hour averages.
    """
    m = a["day_means"][:, :, PERIODS.index(period)]
    return (m >= LIMIT).mean(axis=1), np.quantile(m, 0.1, axis=1), np.quantile(m, 0.9, axis=1)


def fit_scale(a_val: Dict[str, np.ndarray], period: str, target: float = 0.8) -> float:
    """
    Method A: find the stretch factor that makes the ensemble's 80% range cover 80% of
    validation outcomes.

    Args:
        a_val: validation ensemble arrays.
        period: "today" or "tomorrow".
        target: coverage to reach.

    Returns:
        The factor (1 = no change; > 1 = wider).
    """
    y = outcomes(a_val)[period]
    ok = np.isfinite(y)
    for factor in np.arange(1.0, 50.0, 0.1):
        _, low, high = scaled_ensemble(a_val, period, factor)
        if np.mean((y[ok] >= low[ok]) & (y[ok] <= high[ok])) >= target:
            return float(round(factor, 1))
    return 50.0


def scaled_ensemble(a: Dict[str, np.ndarray], period: str, factor: float):
    """
    Args:
        a: ensemble arrays.
        period: "today" or "tomorrow".
        factor: stretch factor from fit_scale.

    Returns:
        (prob, low, high) of the stretched members.
    """
    m = a["day_means"][:, :, PERIODS.index(period)]
    centre = np.median(m, axis=1, keepdims=True)
    stretched = centre + factor * (m - centre)
    return (stretched >= LIMIT).mean(axis=1), np.quantile(stretched, 0.1, axis=1), np.quantile(stretched, 0.9, axis=1)


def fit_ratios(a_val: Dict[str, np.ndarray], period: str) -> np.ndarray:
    """
    Method B: the model's real errors on validation, as actual / forecast 24-hour averages.

    Args:
        a_val: validation ensemble arrays.
        period: "today" or "tomorrow".

    Returns:
        (M,) sorted ratios (M = validation forecasts with a judged outcome).
    """
    y, f = outcomes(a_val)[period], point_means(a_val)[period]
    ok = np.isfinite(y) & (f > 0) & (y > 0)
    return np.sort(y[ok] / f[ok])


def ratio_method(a: Dict[str, np.ndarray], period: str, ratios: np.ndarray):
    """
    Args:
        a: ensemble arrays (only the plain forecast is used).
        period: "today" or "tomorrow".
        ratios: from fit_ratios.

    Returns:
        (prob, low, high): P(forecast x ratio >= LIMIT) and the 10th / 90th percentiles of
        forecast x ratio.
    """
    f = point_means(a)[period]
    # P(f * r >= LIMIT) = share of ratios >= LIMIT / f  (ratios are sorted)
    prob = 1.0 - np.searchsorted(ratios, LIMIT / np.maximum(f, 1e-6), side="left") / len(ratios)
    return prob, f * np.quantile(ratios, 0.1), f * np.quantile(ratios, 0.9)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_probabilities(prob: np.ndarray, low: np.ndarray, high: np.ndarray, y: np.ndarray,
                        base_rate: float, bins=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)) -> Dict:
    """
    Args:
        prob: (N,) stated probabilities of y >= LIMIT.
        low, high: (N,) stated 80% range.
        y: (N,) real 24-hour averages (NaN = not judged; those rows are skipped).
        base_rate: event frequency on validation (the "always say this" reference forecast).
        bins: probability bins for the reliability table.

    Returns:
        dict with n, base_rate_here, brier, brier_reference, brier_skill, coverage_80,
        csi_at_50 (warn when prob >= 0.5), and reliability: list of
        {bin, n, mean_prob, observed} per non-empty bin.
    """
    ok = np.isfinite(y)
    p, lo, hi, yy = prob[ok], low[ok], high[ok], y[ok]
    event = (yy >= LIMIT).astype(float)
    brier = float(np.mean((p - event) ** 2))
    ref = float(np.mean((base_rate - event) ** 2))
    warned = p >= 0.5
    hits, misses = int((warned & (event == 1)).sum()), int((~warned & (event == 1)).sum())
    fas = int((warned & (event == 0)).sum())
    rel = []
    for lo_b, hi_b in zip(bins[:-1], bins[1:]):
        m = (p >= lo_b) & (p < hi_b)
        if m.any():
            rel.append({"bin": f"{lo_b:.1f}-{min(hi_b, 1):.1f}", "n": int(m.sum()),
                        "mean_prob": round(float(p[m].mean()), 3), "observed": round(float(event[m].mean()), 3)})
    return {"n": int(ok.sum()), "base_rate_here": round(float(event.mean()), 3), "brier": round(brier, 4),
            "brier_reference": round(ref, 4), "brier_skill": round(1 - brier / ref, 3) if ref > 0 else None,
            "coverage_80": round(float(np.mean((yy >= lo) & (yy <= hi))), 3),
            "csi_at_50": round(hits / (hits + misses + fas), 3) if hits + misses + fas else None,
            "reliability": rel}


def fit_all(a_val: Dict[str, np.ndarray]) -> Dict:
    """
    Fit every method on validation.

    Args:
        a_val: validation ensemble arrays.

    Returns:
        {period: {"scale_factor", "ratios", "base_rate"}} for "today" and "tomorrow".
    """
    fitted = {}
    for period in PERIODS:
        y = outcomes(a_val)[period]
        fitted[period] = {"scale_factor": fit_scale(a_val, period), "ratios": fit_ratios(a_val, period),
                          "base_rate": float(np.nanmean(y >= LIMIT)) if np.isfinite(y).any() else 0.0}
    return fitted


def evaluate_all(a: Dict[str, np.ndarray], fitted: Dict, sensors: Optional[set] = None) -> pd.DataFrame:
    """
    Score every method on one period's ensemble arrays.

    Args:
        a: ensemble arrays of the period to check.
        fitted: fit_all() result (from validation).
        sensors: only score these sensor ids (e.g. the new ones). None = all.

    Returns:
        One row per (period, method): the score_probabilities() fields (reliability as text).
    """
    keep = np.ones(len(a["point"]), bool) if sensors is None else np.isin(a["sensor"], list(sensors))
    sub = {k: (v[keep] if len(v) == len(keep) else v) for k, v in a.items()}
    rows = []
    for period in PERIODS:
        y, f = outcomes(sub)[period], fitted[period]
        methods = {"raw": raw_ensemble(sub, period),
                   "scaled": scaled_ensemble(sub, period, f["scale_factor"]),
                   "ratio (primary)": ratio_method(sub, period, f["ratios"])}
        for name, (p, lo, hi) in methods.items():
            s = score_probabilities(p, lo, hi, y, f["base_rate"])
            s["reliability"] = "; ".join(f"{r['bin']}: said {r['mean_prob']:.2f}, got {r['observed']:.2f} (n={r['n']})"
                                         for r in s["reliability"])
            rows.append({"period": period, "method": name, **s})
    return pd.DataFrame(rows)


def fit_hourly_band(a_val: Dict[str, np.ndarray], low_q: float = 0.1, high_q: float = 0.9):
    """
    Hourly ranges for the app's chart: on validation, the 10th / 90th percentile of
    actual / forecast at each forecast hour (real readings only).

    Args:
        a_val: validation ensemble arrays.
        low_q, high_q: percentiles of the band.

    Returns:
        (low_ratio, high_ratio): (horizon,) arrays. The band at hour h is
        forecast_h x low_ratio[h] .. forecast_h x high_ratio[h].
    """
    ratio = np.where((a_val["point"] > 0) & np.isfinite(a_val["truth"]), a_val["truth"] / a_val["point"], np.nan)
    return np.nanquantile(ratio, low_q, axis=0), np.nanquantile(ratio, high_q, axis=0)


def hourly_coverage(a: Dict[str, np.ndarray], low_ratio: np.ndarray, high_ratio: np.ndarray) -> float:
    """
    Args:
        a: ensemble arrays of a period to check.
        low_ratio, high_ratio: from fit_hourly_band.

    Returns:
        Share of real hourly readings inside the band (should be ~0.80).
    """
    low, high, t = a["point"] * low_ratio, a["point"] * high_ratio, a["truth"]
    ok = np.isfinite(t)
    return float(np.mean((t[ok] >= low[ok]) & (t[ok] <= high[ok])))
