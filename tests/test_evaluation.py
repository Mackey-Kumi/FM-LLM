import numpy as np
import pandas as pd
import pytest
import torch

from fmllm.data import SensorWindows
from fmllm.evaluation import bootstrap, collect_records, summarize
from fmllm.events import score
from fmllm.model import FMLLM, ModelConfig, TinyBackbone

EVENTS = {"high": ("mean", 24, 30.0)}


def fake_records(n_days=30, seed=0):
    """A small collect_records-style table: 2 sensors x n_days daily forecasts x 2 methods."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2026-01-01", periods=n_days, freq="D"):
        for sensor in ("A", "B"):
            event = bool(rng.random() < 0.4)
            for method, err, hit_prob in (("fm_llm", 5.0, 0.9), ("persistence", 9.0, 0.6)):
                warned = event if rng.random() < hit_prob else not event
                rows.append({"sensor": sensor, "issue_time": d, "day": d, "daily": True, "method": method,
                             "mae_24h": err + rng.normal(0, 1), "high__event": event, "high__warned": warned,
                             "high__first_hour": None})
    return pd.DataFrame(rows)


def test_bootstrap_estimates_match_direct_computation_and_sit_inside_their_interval():
    df = fake_records()
    boot = bootstrap(df, EVENTS, n_boot=500)
    mae = boot[(boot.metric == "mae_24h") & (boot.method == "fm_llm")].iloc[0]
    assert mae.estimate == pytest.approx(df[df.method == "fm_llm"].mae_24h.mean())
    assert mae.ci_low <= mae.estimate <= mae.ci_high
    csi = boot[(boot.metric == "high:csi") & (boot.scope == "all") & (boot.method == "fm_llm")].iloc[0]
    recs = [{"event": e, "warned": w} for e, w in df[df.method == "fm_llm"][["high__event", "high__warned"]].values]
    assert csi.estimate == pytest.approx(score(recs)["csi"])


def test_clearly_better_method_has_interval_excluding_zero():
    boot = bootstrap(fake_records(n_days=60), EVENTS, n_boot=500)
    mae = boot[(boot.metric == "mae_24h") & (boot.method == "persistence")].iloc[0]
    assert mae["diff"] < 0 and mae.diff_high < 0      # fm_llm's error is lower on every resample
    assert mae.ref_better == 1.0


def test_identical_methods_differ_by_exactly_zero():
    df = fake_records()
    twin = df[df.method == "fm_llm"].assign(method="copy")
    boot = bootstrap(pd.concat([df[df.method == "fm_llm"], twin]), EVENTS, n_boot=200)
    rows = boot[boot.method == "copy"]
    assert np.allclose(rows["diff"], 0) and np.allclose(rows.diff_low, 0) and np.allclose(rows.diff_high, 0)
    assert (rows.ref_better == 0).all()


def test_bootstrap_is_reproducible_with_a_seed():
    df = fake_records()
    pd.testing.assert_frame_equal(bootstrap(df, EVENTS, n_boot=100, seed=1), bootstrap(df, EVENTS, n_boot=100, seed=1))


def test_summarize_matches_score():
    df = fake_records()
    summary, per_sensor = summarize(df, EVENTS)
    recs = [{"event": e, "warned": w} for e, w in df[df.method == "persistence"][["high__event", "high__warned"]].values]
    assert summary["warnings"]["high"]["daily"]["scores"]["persistence"]["hits"] == score(recs)["hits"]
    assert set(per_sensor.index) == {"A", "B"}


def test_collect_records_one_row_per_forecast_and_method():
    torch.manual_seed(0)
    idx = pd.date_range("2026-01-01", periods=24 * 12, freq="h")
    wide = pd.DataFrame({"s1": 20 + 10 * np.sin(np.arange(len(idx)) / 4)}, index=idx)
    config = ModelConfig(patch_len=48, seq_len=96, mlp_dim=16, llm_dim=32, hidden_dim=16)
    model = FMLLM(config, TinyBackbone(config.llm_dim))
    ds = SensorWindows(wide, seq_len=96, pred_len=48, issue_hours=(0, 12))
    df = collect_records(model, ds, "cpu", EVENTS, horizons=(24, 48))
    assert len(df) == 4 * len(ds)                            # fm_llm + 3 simple rules
    assert {"mae_24h", "mae_48h", "high__event", "high__warned"} <= set(df.columns)
    assert df[df.method == "persistence"].daily.sum() == sum(ds.issue_time(i).hour == 0 for i in range(len(ds)))
