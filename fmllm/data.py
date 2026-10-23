"""
Hourly sensor data -> training windows.

Each sensor is one series (channel independence), so a sample is one sensor's
`seq_len` hours of history plus the `pred_len` hours that follow. Short gaps
are filled by linear interpolation; windows that still contain a gap are
skipped rather than guessed at.

Pipeline, in the order experiment 12 calls it:
    raw   = load_hourly(csv)                 # readings as a table, NaN where missing (incl. zero-runs)
    wide  = fill_short_gaps(raw, max_gap)    # short holes filled
    scale = train_scale(wide, train_end)     # per-sensor mean/std, training period only
    ds    = SensorWindows(wide, ..., observed=raw)   # (history, future) samples for one split
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

PM25_VALID_RANGE = (0.0, 1000.0)  # µg/m³; outside this is a sensor fault, not air


def drop_zero_runs(wide: pd.DataFrame, min_run: int = 3) -> pd.DataFrame:
    """
    Treat runs of exact-zero readings as missing. Some sensors keep reporting
    0 µg/m³ while broken (sensor 10330993: a 1,648-hour run, through the
    2025-26 Harmattan). Real air is never exactly 0 for hours on end; 93% of all
    zeros in the Accra data come in runs of 6 h or more.

    Args:
        wide: hourly table (rows = hours, columns = sensors), NaN = missing.
        min_run: runs of at least this many consecutive exact zeros are removed.
            Shorter runs (isolated zeros) are kept.

    Returns:
        A copy of `wide` with those runs set to NaN.
    """
    out = wide.copy()
    for col in out:
        zero = out[col].eq(0)
        run_id = (zero != zero.shift()).cumsum()
        run_len = zero.groupby(run_id).transform("sum")
        out.loc[zero & (run_len >= min_run), col] = np.nan
    return out


def load_hourly(csv_path, min_coverage: float = 0.0, valid_range=PM25_VALID_RANGE,
                zero_run_hours: Optional[int] = 3) -> pd.DataFrame:
    """
    Read the CSV written by scripts/fetch_openaq.py into one table: one row per
    hour, one column per sensor.

    Note on `min_coverage`: it is off by default. OpenAQ's coverage compares the
    number of readings in an hour with the number it expected. Clarity sensors
    (most of Accra's network) report once an hour where OpenAQ expects four, so
    a perfectly normal Clarity hour shows 25%.

    Args:
        csv_path: path to pm25_hourly.csv. Long format, one row per (sensor, hour), with
            columns sensor_id, hour (UTC, ISO text), pm25 (µg/m³) and coverage (%).
        min_coverage: hours with coverage below this percentage are treated as missing.
            0 keeps everything.
        valid_range: (low, high) in µg/m³. Readings outside it are sensor faults and
            become missing.
        zero_run_hours: runs of this many or more consecutive exact-zero readings are
            sensor faults and become missing (see drop_zero_runs). None = keep zeros.

    Returns:
        DataFrame indexed by hour (continuous, every hour from the first reading to the
        last, UTC without timezone), with one column per sensor_id. Values are PM2.5 in
        µg/m³, NaN where there's no valid reading.
    """
    df = pd.read_csv(csv_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_localize(None)
    bad = ~df["pm25"].between(*valid_range)
    if "coverage" in df:
        bad |= df["coverage"].fillna(100) < min_coverage
    df.loc[bad, "pm25"] = np.nan
    wide = df.pivot_table(index="hour", columns="sensor_id", values="pm25", aggfunc="mean")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(full).rename_axis("hour")
    return drop_zero_runs(wide, zero_run_hours) if zero_run_hours else wide


def fill_short_gaps(wide: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """
    Fill short runs of missing hours by drawing a straight line between the
    readings on either side (linear interpolation). Longer runs stay missing,
    and so do gaps at the very start or end of a sensor's record.

    Args:
        wide: hourly table from load_hourly (rows = hours, columns = sensors, NaN = missing).
        max_gap: longest run of consecutive missing hours to fill. A run of
            max_gap + 1 or more hours is left as NaN.

    Returns:
        A copy of `wide` with the short gaps filled. The input is not modified.
    """
    out = wide.copy()
    for col in out:
        s = out[col]
        isna = s.isna()
        run_id = (isna != isna.shift()).cumsum()             # one id per run of NaN / non-NaN
        run_len = isna.groupby(run_id).transform("sum")      # length of the NaN run each hour is in
        filled = s.interpolate(method="linear", limit_area="inside")
        out[col] = s.where(~(isna & (run_len <= max_gap)), filled)
    return out


def window_starts(values: np.ndarray, seq_len: int, pred_len: int, stride: int = 1,
                  lo: int = 0, hi: Optional[int] = None) -> np.ndarray:
    """
    Find every position where a complete window (history + future, no NaN) fits.

    Args:
        values: one sensor's hourly series, shape (hours,). NaN = missing.
        seq_len: history length in hours (what the model reads), e.g. 672.
        pred_len: future length in hours (what it predicts), e.g. 96.
        stride: keep every `stride`-th valid start. 1 keeps them all; neighbouring
            windows overlap by all but one hour.
        lo: earliest allowed start index.
        hi: the window must end at or before this index (exclusive). None = end of series.

    Returns:
        Integer array of start indices s. Each satisfies lo <= s, s + seq_len + pred_len <= hi,
        and values[s : s + seq_len + pred_len] contains no NaN.
    """
    hi = len(values) if hi is None else hi
    total = seq_len + pred_len
    ok = ~np.isnan(values)
    # bad_before[i] = number of NaNs in values[:i], so the NaN count in values[a:b] is bad_before[b] - bad_before[a]
    bad_before = np.concatenate([[0], np.cumsum(~ok)])
    starts = np.arange(lo, max(lo, hi - total + 1))
    clean = (bad_before[starts + total] - bad_before[starts]) == 0
    return starts[clean][::stride]


class SensorWindows(torch.utils.data.Dataset):
    """
    The (history, future) training or evaluation samples for one time split.

    Each sample is one sensor's `seq_len` hours of history and the `pred_len`
    hours after it. The future must fall inside [start, end), the part of the
    timeline this split owns. The history may reach back before `start` into an
    earlier split (as in the ETTh1 experiments), but never forward in time.

    If `observed` is given, the first `real_future_hours` of every future (the
    part we score) must be real readings: filled hours may appear in the history
    and in the unscored tail of the future, never in what we score.

    Args:
        wide: gap-filled hourly table (rows = hours, columns = sensors), i.e. the output
            of fill_short_gaps. This is what the model reads.
        seq_len: history length in hours, e.g. 672 (28 days).
        pred_len: future length in hours, e.g. 96 (one model patch).
        start: first hour (inclusive, e.g. "2025-10-01") a future may begin at. None = no limit.
        end: hour (exclusive) the future must end before. None = end of data.
        stride: keep every `stride`-th valid window per sensor (1 = all).
        sensors: which sensor columns to use. None = all columns of `wide`.
        scale: {sensor_id: (mean, std)} from train_scale. Samples are returned as
            (value - mean) / std. None = no scaling.
        observed: the table BEFORE gap filling (output of load_hourly), or a boolean
            "is a real reading" table of the same shape. None = don't check.
        real_future_hours: how many hours at the start of the future must be real
            readings. None = the whole future (pred_len).
        issue_hours: keep only forecasts issued at these clock hours (UTC = Ghana time),
            e.g. (0, 6, 12, 18). Applied before `stride`. None = any hour.

    Attributes:
        items: list of (sensor column index, start index) for every sample.
        values: (hours, sensors) float32 array of the gap-filled data, unscaled (µg/m³).
        sensors: the sensor ids, in column order.
    """

    def __init__(self, wide: pd.DataFrame, seq_len: int, pred_len: int, start: Optional[str] = None,
                 end: Optional[str] = None, stride: int = 1, sensors: Optional[Sequence] = None,
                 scale: Optional[Dict] = None, observed: Optional[pd.DataFrame] = None,
                 real_future_hours: Optional[int] = None, issue_hours: Optional[Sequence[int]] = None):
        self.seq_len, self.pred_len = seq_len, pred_len
        self.sensors = list(sensors) if sensors is not None else list(wide.columns)
        self.values = wide[self.sensors].to_numpy(dtype="float32")
        self.index = wide.index
        self.scale = scale or {}
        lo = 0 if start is None else int(wide.index.searchsorted(pd.Timestamp(start)))
        hi = len(wide) if end is None else int(wide.index.searchsorted(pd.Timestamp(end)))

        real = None  # (hours, sensors) bool: True where there was an actual reading
        if observed is not None:
            obs = observed[self.sensors]
            real = obs.to_numpy() if (obs.dtypes == bool).all() else obs.notna().to_numpy()

        self.items: List[Tuple[int, int]] = []
        for j in range(len(self.sensors)):
            # a future [s + seq_len, s + seq_len + pred_len) starting at or after lo needs s >= lo - seq_len
            first = max(0, lo - seq_len)
            starts = window_starts(self.values[:, j], seq_len, pred_len, 1, first, hi)
            if real is not None and len(starts):
                fake_before = np.concatenate([[0], np.cumsum(~real[:, j])])  # same counting trick as window_starts
                fut = starts + seq_len
                n_real = pred_len if real_future_hours is None else real_future_hours
                starts = starts[(fake_before[fut + n_real] - fake_before[fut]) == 0]
            if issue_hours is not None and len(starts):
                starts = starts[np.isin(self.index[starts + seq_len].hour, list(issue_hours))]
            for s in starts[::stride]:
                self.items.append((j, int(s)))

    def __len__(self):
        """Returns: the number of samples."""
        return len(self.items)

    def _scaled(self, j, x):
        """
        Args:
            j: sensor column index.
            x: values of that sensor in µg/m³.

        Returns:
            (x - mean) / std with that sensor's training-period stats.
        """
        mean, std = self.scale.get(self.sensors[j], (0.0, 1.0))
        return (x - mean) / std

    def __getitem__(self, i):
        """
        Args:
            i: sample number, 0 <= i < len(self).

        Returns:
            history: float tensor (1, seq_len), scaled. The leading 1 is the channel dimension.
            future: float tensor (1, pred_len), scaled.
            j: sensor column index (look up the id with self.sensors[j]).
            s: start index of the window in self.values.
        """
        j, s = self.items[i]
        hist = self._scaled(j, self.values[s: s + self.seq_len, j])
        fut = self._scaled(j, self.values[s + self.seq_len: s + self.seq_len + self.pred_len, j])
        return torch.from_numpy(hist[None].copy()), torch.from_numpy(fut[None].copy()), j, s

    def issue_time(self, i) -> pd.Timestamp:
        """
        Args:
            i: sample number.

        Returns:
            The hour the forecast would be issued at: the first hour of the future window.
        """
        _, s = self.items[i]
        return self.index[s + self.seq_len]


def train_scale(wide: pd.DataFrame, train_end: str) -> Dict:
    """
    Per-sensor average and spread, computed from the training period only, so
    nothing about the validation or test periods leaks into the model.

    Args:
        wide: hourly table (rows = hours, columns = sensors).
        train_end: first hour NOT in training, e.g. "2025-10-01".

    Returns:
        {sensor_id: (mean, std)} in µg/m³. Sensors with fewer than 2 training readings
        get (0.0, 1.0), i.e. no scaling.
    """
    train = wide.loc[: pd.Timestamp(train_end) - pd.Timedelta(hours=1)]
    out = {}
    for col in wide:
        x = train[col].dropna()
        out[col] = (float(x.mean()), float(x.std()) or 1.0) if len(x) > 1 else (0.0, 1.0)
    return out
