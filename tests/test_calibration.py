import numpy as np
import pytest

from fmllm.calibration import (fit_all, fit_ratios, fit_scale, outcomes, ratio_method, raw_ensemble,
                               score_probabilities)


def fake(n=400, seed=0, bias=1.0):
    """Forecasts whose real outcome = forecast x a random ratio (median `bias`); ensemble far too narrow."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(10, 60, size=n)
    actual = f * bias * np.exp(rng.normal(0, 0.25, size=n))
    truth = np.repeat(actual[:, None], 48, axis=1)
    point = np.repeat(f[:, None], 48, axis=1)
    members = f[:, None] + rng.normal(0, 0.5, size=(n, 30))           # tiny spread: overconfident
    day_means = np.stack([members, members], axis=2)
    return {"point": point, "truth": truth, "day_means": day_means, "sensor": np.array(["s"] * n),
            "median": point, "q10": point, "q90": point}


def test_outcomes_need_enough_real_hours():
    a = fake(3)
    a["truth"][0, :10] = np.nan            # 14 real hours today: not judged
    y = outcomes(a)["today"]
    assert np.isnan(y[0]) and np.isfinite(y[1])


def test_raw_ensemble_is_overconfident_and_ratio_method_is_calibrated():
    val, test = fake(seed=0), fake(seed=1)
    fitted = fit_all(val)
    y = outcomes(test)["today"]
    raw = score_probabilities(*raw_ensemble(test, "today"), y, fitted["today"]["base_rate"])
    rat = score_probabilities(*ratio_method(test, "today", fitted["today"]["ratios"]), y, fitted["today"]["base_rate"])
    assert raw["coverage_80"] < 0.3                       # narrow ensemble misses most outcomes
    assert abs(rat["coverage_80"] - 0.8) < 0.08           # learned errors give honest ranges
    assert rat["brier"] < raw["brier"]


def test_scale_factor_widens_a_narrow_ensemble():
    assert fit_scale(fake(seed=0), "today") > 5


def test_ratio_probability_matches_direct_count():
    ratios = fit_ratios(fake(seed=0), "today")
    a = fake(5, seed=3)
    p, _, _ = ratio_method(a, "today", ratios)
    f = a["point"][:, :24].mean(axis=1)
    direct = [(f_i * ratios >= 35).mean() for f_i in f]
    assert np.allclose(p, direct)


def test_brier_skill_zero_for_climatology():
    y = np.array([10.0, 40.0, 50.0, 20.0])
    base = 0.5
    s = score_probabilities(np.full(4, base), np.zeros(4), np.full(4, 100.0), y, base)
    assert s["brier_skill"] == pytest.approx(0.0) and s["coverage_80"] == 1.0


def test_hourly_band_covers_about_80_percent():
    from fmllm.calibration import fit_hourly_band, hourly_coverage
    lo, hi = fit_hourly_band(fake(seed=0))
    assert lo.shape == (48,) and (lo < hi).all()
    assert abs(hourly_coverage(fake(seed=1), lo, hi) - 0.8) < 0.08


def test_ratio_calibration_object_matches_the_functions(tmp_path):
    import json
    from fmllm.calibration import RatioCalibration, fit_hourly_band
    val = fake(seed=0)
    fitted = fit_all(val)
    lo, hi = fit_hourly_band(val)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"ratios": {p: fitted[p]["ratios"].tolist() for p in fitted},
                                "hourly_band": {"low_ratio": lo.tolist(), "high_ratio": hi.tolist()}, "limit": 35.0}))
    cal = RatioCalibration.load(path)
    test = fake(5, seed=4)
    p, low, high = ratio_method(test, "today", fitted["today"]["ratios"])
    f = test["point"][:, :24].mean(axis=1)
    for k in range(5):
        d = cal.day(f[k], "today")
        assert d["prob_exceed"] == pytest.approx(p[k]) and d["low"] == pytest.approx(low[k], rel=1e-6)
    band_lo, band_hi = cal.hourly_band(test["point"][0])
    assert np.allclose(band_lo, test["point"][0] * lo)
