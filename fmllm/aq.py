"""
The Accra PM2.5 experiment set-up, shared by every air-quality experiment
(12 = train, 13 = evaluate a saved model, ...), so they all use exactly the same
data rules, split and warning events.

Choices made here, and why (see the session logs for the evidence behind each):
  * Split by time, so nothing from the future leaks into training:
      train  up to 30 Sep 2025     (includes the Dec 2024 - Feb 2025 Harmattan)
      val    Oct - Nov 2025        (early stopping and all tuning decisions)
      test   Dec 2025 - Feb 2026   (the 2025-26 Harmattan; used only for final results)
  * History gaps up to MAX_GAP hours are interpolated; the scored first
    max(HORIZONS) hours of every future must be real readings.
  * Warning events follow how air-quality standards are defined: on the 24-hour
    average (main event: Ghana's standard, GS 1236:2019 Table 2, 35 µg/m³).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from .data import SensorWindows, fill_short_gaps, load_hourly, train_scale
from .model import ModelConfig

SPLIT = {"train_end": "2025-10-01", "val_end": "2025-12-01", "test_end": "2026-03-01"}
HORIZONS = (24, 48)                 # hours ahead we score
TEST_ISSUE_HOURS = (0, 6, 12, 18)   # test forecasts issued at fixed clock hours (Ghana time = UTC)
MAX_GAP = 12                        # interpolate history gaps up to this many hours
TRAIN_STRIDE, EVAL_STRIDE = 1, 6    # every training window; every 6th validation window

# name -> (statistic over the next `hours`, hours, limit in µg/m³)
EVENTS = {
    "ghana_24h_mean": ("mean", 24, 35.0),      # Ghana's 24-h PM2.5 standard (GS 1236:2019). Main event.
    "unhealthy_24h_mean": ("mean", 24, 55.5),  # US AQI "unhealthy" tier: rarer, more serious days.
    "any_hour_35.5": ("max", 24, 35.5),        # run-1 definition, kept only for comparison (93% base rate).
}


@dataclass
class AQSplits:
    """
    Everything an experiment needs from the data.

    Attributes:
        raw: hourly table before gap filling (from load_hourly), NaN = no reading.
        wide: the same table with short gaps filled (what the model reads).
        scale: {sensor_id: (mean, std)} from the training period.
        train, val, test: SensorWindows for each split.
    """
    raw: pd.DataFrame
    wide: pd.DataFrame
    scale: Dict
    train: SensorWindows
    val: SensorWindows
    test: SensorWindows


def build_splits(csv_path, config: ModelConfig, split: Optional[Dict] = None, max_gap: int = MAX_GAP) -> AQSplits:
    """
    Load the PM2.5 data and cut it into the train / validation / test windows.

    Args:
        csv_path: pm25_hourly.csv (from scripts/fetch_openaq.py).
        config: the model config. Its seq_len and patch_len set the window sizes
            (history = seq_len, future = patch_len).
        split: {"train_end", "val_end", "test_end"} dates. None = SPLIT.
        max_gap: longest history gap (hours) that is interpolated.

    Returns:
        AQSplits.
    """
    split = split or SPLIT
    raw = load_hourly(csv_path)
    wide = fill_short_gaps(raw, max_gap=max_gap)
    scale = train_scale(wide, split["train_end"])
    common = {"seq_len": config.seq_len, "pred_len": config.patch_len, "scale": scale, "observed": raw,
              "real_future_hours": max(HORIZONS)}
    return AQSplits(
        raw=raw, wide=wide, scale=scale,
        train=SensorWindows(wide, end=split["train_end"], stride=TRAIN_STRIDE, **common),
        val=SensorWindows(wide, start=split["train_end"], end=split["val_end"], stride=EVAL_STRIDE, **common),
        test=SensorWindows(wide, start=split["val_end"], end=split["test_end"],
                           issue_hours=TEST_ISSUE_HOURS, **common),
    )


# Held-out period AFTER the test set: never used for training, tuning or testing.
POST_TEST = {"start": "2026-03-01", "end": None}   # None = up to the end of the downloaded data
MIN_REAL_FRACTION = 0.75   # "Option A": >= 75% of the scored hours real; only real hours are scored


def new_sensor_ids(raw: pd.DataFrame, train_end: Optional[str] = None):
    """
    Sensors the model never saw in training: no readings before the end of the training period.

    Args:
        raw: hourly readings (columns = sensor ids).
        train_end: first hour after training. None = SPLIT["train_end"].

    Returns:
        Set of sensor ids.
    """
    before = raw.loc[: pd.Timestamp(train_end or SPLIT["train_end"]) - pd.Timedelta(hours=1)]
    return {c for c in raw.columns if before[c].notna().sum() == 0}


def build_period(csv_path, config: ModelConfig, start: str, end: Optional[str] = None,
                 max_gap: int = MAX_GAP, min_real_fraction: float = MIN_REAL_FRACTION) -> AQSplits:
    """
    Forecast windows for any period, with the training data rules (gap filling,
    scaling from the training period) and the Option A scoring rule: at least
    `min_real_fraction` of the scored hours must be real, and only those are scored.
    Used for the post-test evaluation, and to re-score the test period the same way.

    Args:
        csv_path: pm25_hourly.csv.
        config: model config (window sizes).
        start: first hour a scored future may begin, e.g. "2026-03-01".
        end: hour the futures must end before. None = end of the data.
        max_gap: longest history gap (hours) that is interpolated.
        min_real_fraction: share of the scored hours that must be real (1.0 = the strict
            rule of experiments 12-13).

    Returns:
        AQSplits whose `test` holds this period's windows (train and val are empty
        placeholders: the same dataset, not used).
    """
    raw = load_hourly(csv_path)
    wide = fill_short_gaps(raw, max_gap=max_gap)
    scale = train_scale(wide, SPLIT["train_end"])
    period = SensorWindows(wide, seq_len=config.seq_len, pred_len=config.patch_len, start=start, end=end,
                           issue_hours=TEST_ISSUE_HOURS, scale=scale, observed=raw,
                           real_future_hours=max(HORIZONS), min_real_fraction=min_real_fraction)
    return AQSplits(raw=raw, wide=wide, scale=scale, train=period, val=period, test=period)
