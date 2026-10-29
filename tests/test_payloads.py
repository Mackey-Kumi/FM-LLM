import json

import numpy as np
import pandas as pd

from fmllm.forecast import summarize_ensemble
from fmllm.payloads import (SensorForecast, SensorInput, day_outlook, forecast_detail_payload, level_of,
                            prepare_inputs, replay_payload, worst_hours)


def test_levels_follow_the_contract():
    assert level_of(10) == "good" and level_of(15) == "moderate" and level_of(34.9) == "moderate"
    assert level_of(35) == "unhealthy" and level_of(55.5) == "very_unhealthy" and level_of(float("nan")) is None


def test_worst_hours_finds_the_peak_window():
    med = np.full(48, 10.0)
    med[29:33] = 50.0
    w = worst_hours(med, pd.Timestamp("2026-10-30 13:00"))
    assert w["start"] == "2026-10-31T18:00:00Z" and w["end"] == "2026-10-31T21:00:00Z" and w["pm25_peak"] == 50.0


def fake_forecast():
    rng = np.random.default_rng(0)
    samples = 30 + rng.normal(0, 3, size=(30, 48))
    inp = SensorInput(7, pd.Timestamp("2026-10-30 12:00"), np.zeros(672, dtype="float32"),
                      pd.Series([20.0, np.nan], index=pd.date_range("2026-10-30 11:00", periods=2, freq="h")))
    return SensorForecast(inp, samples, {"today": summarize_ensemble(samples[:, :24], 35),
                                         "tomorrow": summarize_ensemble(samples[:, 24:], 35),
                                         "all": summarize_ensemble(samples, 35)})


def test_detail_payload_matches_contract_and_is_json():
    p = forecast_detail_payload(fake_forecast(), "Test Market", pd.Timestamp("2026-10-30 13:05"))
    assert set(p) == {"id", "name", "issued_at", "forecast_start", "recent", "hourly", "daily", "worst_hours", "message"}
    assert p["forecast_start"] == "2026-10-30T13:00:00Z" and len(p["hourly"]) == 48
    assert p["recent"][1]["pm25"] is None                         # missing reading -> null
    assert {"pm25", "low", "high"} <= set(p["hourly"][0]) and p["hourly"][0]["low"] <= p["hourly"][0]["high"]
    assert p["daily"][0]["level"] == "moderate" and 0 <= p["daily"][0]["prob_unhealthy"] <= 1
    json.dumps(p)                                                  # serialisable as-is


def test_prepare_inputs_uses_each_sensors_own_last_reading():
    idx = pd.date_range("2026-09-01", periods=24 * 40, freq="h")
    raw = pd.DataFrame({1: np.full(len(idx), 20.0), 2: np.full(len(idx), 25.0), 3: np.full(len(idx), 30.0)}, index=idx)
    raw.iloc[-3:, 0] = np.nan            # sensor 1 reports 3 h late: still forecast, from its own last reading
    raw.iloc[-24 * 3:, 1] = np.nan       # sensor 2 silent for 3 days: skipped
    raw.iloc[100:130, 2] = np.nan        # sensor 3: 30-h gap, but it's older than 28 days... check both cases
    as_of = idx[-1] + pd.Timedelta(hours=1)
    inputs, skipped = prepare_inputs(raw, raw, as_of)
    ids = {i.sensor_id: i for i in inputs}
    assert ids[1].last_reading_time == idx[-4] and len(ids[1].history) == 672
    assert skipped[2] == "no recent reading"
    assert 3 in ids                       # the gap is outside its last 28 days


def test_replay_reports_the_real_outcome():
    f = fake_forecast()
    idx = pd.date_range("2026-10-30 13:00", periods=24, freq="h")
    raw = pd.DataFrame({7: np.full(24, 40.0)}, index=idx)
    p = replay_payload([f], {7: "Test Market"}, pd.Timestamp("2026-10-30 13:00"), raw)
    today = p["sensors"][0]["today"]
    assert p["mode"] == "replay" and today["actual_mean_24h"] == 40.0 and today["actual_level"] == "unhealthy"
    assert day_outlook(f.summary["today"])["mean_24h"] == today["mean_24h"]


def test_in_training_is_marked_only_when_the_training_set_is_given():
    f = fake_forecast()
    idx = pd.date_range("2026-10-30 13:00", periods=24, freq="h")
    raw = pd.DataFrame({7: np.full(24, 40.0)}, index=idx)
    issued = pd.Timestamp("2026-10-30 13:00")
    assert "in_training" not in replay_payload([f], {}, issued, raw)["sensors"][0]
    assert replay_payload([f], {}, issued, raw, trained={7})["sensors"][0]["in_training"] is True
    assert replay_payload([f], {}, issued, raw, trained={8})["sensors"][0]["in_training"] is False
    assert forecast_detail_payload(f, "", issued, trained={7})["in_training"] is True


def test_worst_hours_ignores_hours_before_issue():
    med = np.full(48, 10.0)
    med[0:4] = 60.0          # the past: between the last reading and the issue time
    med[20:24] = 40.0
    start = pd.Timestamp("2026-10-30 10:00")
    w = worst_hours(med, start, not_before=start + pd.Timedelta(hours=3))
    assert w["start"] == "2026-10-31T06:00:00Z"


def test_run_forecasts_with_ratio_calibration_uses_one_plain_forecast():
    import torch
    from fmllm.calibration import RatioCalibration
    from fmllm.model import FMLLM, ModelConfig, TinyBackbone
    from fmllm.payloads import run_forecasts
    torch.manual_seed(0)
    config = ModelConfig(mlp_dim=16, llm_dim=32, hidden_dim=16)
    model = FMLLM(config, TinyBackbone(config.llm_dim))
    inp = SensorInput(1, pd.Timestamp("2026-10-30 12:00"), np.full(672, 30.0, dtype="float32") +
                      np.sin(np.arange(672, dtype="float32")), pd.Series(dtype=float))
    cal = RatioCalibration({"today": [0.8, 1.0, 1.5], "tomorrow": [0.7, 1.0, 1.6]}, np.full(48, 0.8), np.full(48, 1.3))
    f = run_forecasts(model, [inp], {1: (30.0, 5.0)}, calibration=cal)[0]
    assert f.samples.shape == (1, 48)                                   # no ensemble
    t = f.summary["today"]
    assert np.allclose(t["low"], t["median"] * 0.8) and 0 <= t["prob_exceed"] <= 1
    # same forecast twice -> identical (dropout stays off)
    g = run_forecasts(model, [inp], {1: (30.0, 5.0)}, calibration=cal)[0]
    assert np.allclose(f.summary["all"]["median"], g.summary["all"]["median"])
