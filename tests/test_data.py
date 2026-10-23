import numpy as np
import pandas as pd
import pytest

from fmllm.data import SensorWindows, drop_zero_runs, fill_short_gaps, load_hourly, train_scale, window_starts


def test_load_hourly_pivots_reindexes_and_drops_bad_values(tmp_path):
    p = tmp_path / "h.csv"
    pd.DataFrame({
        "sensor_id": [1, 1, 1, 2, 2],
        "hour": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z",
                 "2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
        "pm25": [10.0, -5.0, 30.0, 20.0, 25.0],
        "coverage": [100, 100, 100, 100, 20],
    }).to_csv(p, index=False)
    w = load_hourly(p, min_coverage=50)
    assert list(w.index) == list(pd.date_range("2026-01-01", periods=4, freq="h"))
    assert w[1].tolist()[0] == 10 and np.isnan(w[1].tolist()[1]) and np.isnan(w[1].tolist()[2])  # negative + missing
    assert np.isnan(w[2].tolist()[1])  # low coverage


def test_fill_short_gaps_only():
    s = pd.DataFrame({"a": [0, np.nan, 2, np.nan, np.nan, np.nan, np.nan, 7.0]})
    out = fill_short_gaps(s, max_gap=3)["a"].tolist()
    assert out[1] == 1
    assert all(np.isnan(out[3:7]))  # 4-hour gap left alone


def test_window_starts_skip_gaps_and_respect_bounds():
    v = np.arange(20, dtype=float)
    v[10] = np.nan
    starts = window_starts(v, seq_len=4, pred_len=2)
    assert all(not (s <= 10 < s + 6) for s in starts)
    assert starts.min() == 0 and starts.max() == 14
    assert window_starts(v, 4, 2, lo=11).tolist() == list(range(11, 15))
    assert window_starts(v, 4, 2, stride=2).tolist() == [0, 2, 4, 12, 14]


def test_sensor_windows_split_and_scaling():
    idx = pd.date_range("2026-01-01", periods=48, freq="h")
    wide = pd.DataFrame({"s1": np.arange(48.0), "s2": np.arange(48.0) * 2}, index=idx)
    scale = train_scale(wide, "2026-01-02")
    ds = SensorWindows(wide, seq_len=8, pred_len=4, start="2026-01-02", scale=scale)
    for i in range(len(ds)):
        assert ds.issue_time(i) >= pd.Timestamp("2026-01-02")  # every forecast lies in this split
    hist, fut, j, s = ds[0]
    assert hist.shape == (1, 8) and fut.shape == (1, 4)
    mean, std = scale[ds.sensors[j]]
    assert hist[0, 0].item() == pytest.approx((wide.iloc[s, j] - mean) / std)
    assert {j for j, _ in ds.items} == {0, 1}


def test_future_must_be_real_readings():
    idx = pd.date_range("2026-01-01", periods=30, freq="h")
    raw = pd.DataFrame({"a": np.arange(30.0)}, index=idx)
    raw.iloc[5, 0] = np.nan    # inside some histories
    raw.iloc[20, 0] = np.nan   # inside some futures
    filled = fill_short_gaps(raw, max_gap=3)
    ds = SensorWindows(filled, seq_len=8, pred_len=4, observed=raw)
    for _, s in ds.items:
        assert not (s + 8 <= 20 < s + 12)  # no filled hour in any target
    assert any(s <= 5 < s + 8 for _, s in ds.items)  # filled hours are fine in the history
    assert len(SensorWindows(filled, 8, 4)) > len(ds)


def test_drop_zero_runs_removes_long_runs_only():
    w = pd.DataFrame({"a": [5.0, 0, 7, 0, 0, 0, 0, 9, 0, 0]})
    out = drop_zero_runs(w, min_run=3)["a"].tolist()
    assert out[1] == 0                      # isolated zero kept
    assert all(np.isnan(out[3:7]))          # 4-hour run removed
    assert out[8:] == [0, 0]                # 2-hour run kept
    assert out[7] == 9


def test_issue_hours_filter():
    idx = pd.date_range("2026-01-01", periods=72, freq="h")
    wide = pd.DataFrame({"a": np.arange(72.0)}, index=idx)
    ds = SensorWindows(wide, seq_len=8, pred_len=4, issue_hours=(0, 12))
    assert len(ds) > 0
    assert {ds.issue_time(i).hour for i in range(len(ds))} == {0, 12}


def test_min_real_fraction_accepts_scattered_gaps_and_truth_keeps_them_missing():
    idx = pd.date_range("2026-01-01", periods=40, freq="h")
    raw = pd.DataFrame({"a": np.arange(40.0)}, index=idx)
    raw.iloc[[30, 33], 0] = np.nan                      # 2 missing hours inside some futures
    filled = fill_short_gaps(raw, max_gap=3)
    strict = SensorWindows(filled, seq_len=8, pred_len=8, observed=raw)
    loose = SensorWindows(filled, seq_len=8, pred_len=8, observed=raw, min_real_fraction=0.75)
    assert len(loose) > len(strict)                      # 6 of 8 real is accepted at 75%
    assert np.isnan(loose.truth[30, 0]) and loose.values[30, 0] == 30.0   # model reads the fill, scoring sees the gap
