"""
Build the API responses defined in docs/api_contract.md.

Used twice, so both produce exactly the same format:
  * scripts/make_api_examples.py writes example files for the app developer;
  * the forecast service answers the live API from the same functions.

The work splits in two:
  1. prepare_inputs / run_forecasts: pick the sensors that can be forecast, cut each
     one's last 28 days (ending at ITS OWN latest reading: sensors report with a
     delay) and run the model ensemble.
  2. *_payload functions: pure dict-building from those results (tested without a model).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from .forecast import forecast_ensemble, summarize_ensemble

HORIZON = 48            # hours forecast per sensor
DAY = 24                # "today" = hours 0-24 from the forecast start, "tomorrow" = hours 24-48
UNHEALTHY = 35.0        # Ghana's 24-hour PM2.5 standard (GS 1236:2019)
MAX_LAG_H = 24          # a sensor silent for longer than this isn't forecast
RECENT_H = 48           # hours of real readings returned with each forecast
LEVELS = ((15.0, "good"), (35.0, "moderate"), (55.5, "unhealthy"), (float("inf"), "very_unhealthy"))


def level_of(mean_24h: Optional[float]) -> Optional[str]:
    """
    Args:
        mean_24h: 24-hour average PM2.5 in µg/m³ (None/NaN allowed).

    Returns:
        "good" (< 15), "moderate" (15-35), "unhealthy" (35-55.5), "very_unhealthy" (>= 55.5),
        or None if the value is missing.
    """
    if mean_24h is None or np.isnan(mean_24h):
        return None
    return next(name for upper, name in LEVELS if mean_24h < upper)


def iso(ts) -> str:
    """
    Args:
        ts: a timestamp in UTC (timezone-naive).

    Returns:
        ISO 8601 text with a Z suffix, e.g. "2026-10-30T14:00:00Z".
    """
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _r(x, digits=1):
    """
    Args:
        x: a number, NaN or None.
        digits: decimals to keep.

    Returns:
        The rounded float, or None for missing values (JSON null).
    """
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), digits)


@dataclass
class SensorInput:
    """
    One sensor ready to forecast.

    Attributes:
        sensor_id: OpenAQ sensor id.
        last_reading_time: hour of its latest real reading (the history ends here).
        history: (seq_len,) gap-filled µg/m³ values ending at last_reading_time.
        recent: the last RECENT_H hours of REAL readings (NaN = missing), ending at last_reading_time.
    """
    sensor_id: int
    last_reading_time: pd.Timestamp
    history: np.ndarray
    recent: pd.Series


@dataclass
class SensorForecast:
    """
    The ensemble result for one sensor.

    Attributes:
        inp: the SensorInput it was made from.
        samples: (n_samples, HORIZON) ensemble in µg/m³; hour 0 = the hour after the last reading.
        summary: summarize_ensemble() of the first 24 h ("today") and of hours 24-48 ("tomorrow").
    """
    inp: SensorInput
    samples: np.ndarray
    summary: Dict = field(default_factory=dict)

    @property
    def forecast_start(self) -> pd.Timestamp:
        """Returns: the first forecast hour (the hour after the last reading)."""
        return self.inp.last_reading_time + pd.Timedelta(hours=1)


def prepare_inputs(raw: pd.DataFrame, wide: pd.DataFrame, as_of: pd.Timestamp, seq_len: int = 672):
    """
    Choose the sensors that can be forecast at time `as_of` and cut their histories.

    A sensor qualifies when it has a real reading within MAX_LAG_H hours before `as_of`,
    and its seq_len hours ending at that reading have no gap the training rules couldn't
    fill (i.e. no NaN left in the gap-filled table).

    Args:
        raw: hourly readings before gap filling (fmllm.data.load_hourly).
        wide: the same after fill_short_gaps (what the model reads).
        as_of: "now" (live) or the replay issue time. Nothing after it is used.
        seq_len: history length the model needs.

    Returns:
        (inputs, skipped): list of SensorInput, and {sensor_id: reason} for the rest.
    """
    inputs, skipped = [], {}
    raw_now = raw.loc[: as_of - pd.Timedelta(hours=1)]
    for sid in raw.columns:
        last = raw_now[sid].last_valid_index()
        if last is None or as_of - last > pd.Timedelta(hours=MAX_LAG_H):
            skipped[sid] = "no recent reading"
            continue
        hist = wide[sid].loc[: last].iloc[-seq_len:]
        if len(hist) < seq_len or hist.isna().any():
            skipped[sid] = "gaps in the last 28 days"
            continue
        recent = raw[sid].loc[last - pd.Timedelta(hours=RECENT_H - 1): last]
        inputs.append(SensorInput(int(sid), last, hist.to_numpy(dtype="float32"), recent))
    return inputs, skipped


def run_forecasts(model, inputs: List[SensorInput], scale: Dict, n_samples: int = 30,
                  device="cpu") -> List[SensorForecast]:
    """
    Run the dropout ensemble for every prepared sensor in one batch.

    Args:
        model: trained FMLLM.
        inputs: from prepare_inputs.
        scale: {sensor_id: (mean, std)} used in training (sensors missing from it get no scaling).
        n_samples: ensemble size.
        device: where the model is.

    Returns:
        One SensorForecast per input, in µg/m³.
    """
    if not inputs:
        return []
    ms = np.array([scale.get(i.sensor_id, (0.0, 1.0)) for i in inputs], dtype="float32")   # (n, 2)
    x = np.stack([(i.history - m) / s for i, (m, s) in zip(inputs, ms)])[:, None, :]
    ens = forecast_ensemble(model, torch.from_numpy(x).to(device), HORIZON, n_samples=n_samples)
    ens = ens[:, :, 0].float().cpu().numpy() * ms[None, :, 1:2] + ms[None, :, 0:1]          # (samples, n, H)
    out = []
    for k, inp in enumerate(inputs):
        samples = ens[:, k]
        out.append(SensorForecast(inp, samples, {
            "today": summarize_ensemble(samples[:, :DAY], UNHEALTHY, hours=DAY),
            "tomorrow": summarize_ensemble(samples[:, DAY:2 * DAY], UNHEALTHY, hours=DAY),
            "all": summarize_ensemble(samples, UNHEALTHY, hours=DAY),
        }))
    return out


def day_outlook(summary: Dict) -> Dict:
    """
    Args:
        summary: summarize_ensemble() of one 24-hour period.

    Returns:
        {"mean_24h", "level", "prob_unhealthy"} for that period.
    """
    mean = float(np.mean(summary["median"]))
    return {"mean_24h": _r(mean), "level": level_of(mean), "prob_unhealthy": _r(summary["prob_exceed"], 2)}


def worst_hours(median: np.ndarray, start: pd.Timestamp, width: int = 4) -> Dict:
    """
    The `width`-hour stretch with the highest forecast average: the time to warn about.

    Args:
        median: (HORIZON,) central forecast in µg/m³.
        start: time of median[0].
        width: hours in the stretch.

    Returns:
        {"start", "end", "pm25_peak"}: ISO times (end = the last hour of the stretch) and the
        highest hourly value inside it.
    """
    sums = np.convolve(median, np.ones(width), mode="valid")
    i = int(np.argmax(sums))
    return {"start": iso(start + pd.Timedelta(hours=i)), "end": iso(start + pd.Timedelta(hours=i + width - 1)),
            "pm25_peak": _r(median[i: i + width].max())}


def message(today: Dict, tomorrow: Dict, worst: Dict) -> str:
    """
    A plain-language summary for the app.

    Args:
        today, tomorrow: day_outlook() results.
        worst: worst_hours() result.

    Returns:
        One or two short sentences.
    """
    words = {"good": "Good", "moderate": "Moderate", "unhealthy": "Unhealthy", "very_unhealthy": "Very unhealthy"}
    first = f"{words.get(today['level'], 'Unknown')} air expected over the next 24 hours"
    second = f"{words.get(tomorrow['level'], 'unknown').lower()} after that"
    text = f"{first}, {second}."
    if worst["pm25_peak"] is not None and worst["pm25_peak"] >= UNHEALTHY:
        s, e = pd.Timestamp(worst["start"][:-1]), pd.Timestamp(worst["end"][:-1])
        text += f" Worst around {s:%a %H:%M}–{e + pd.Timedelta(hours=1):%H:%M}, when levels may reach unhealthy."
    return text


def health_payload(issued_at, n_forecast: int, model_name: str, calibrated: bool) -> Dict:
    """
    Args:
        issued_at: when the forecasts were made.
        n_forecast: number of sensors forecast.
        model_name: model identifier.
        calibrated: whether the likelihoods have been calibrated.

    Returns:
        GET /v1/health response.
    """
    return {"status": "ok", "model": model_name, "last_update": iso(issued_at), "sensors_forecast": n_forecast,
            "likelihood_status": "calibrated" if calibrated else "uncalibrated"}


def sensors_payload(meta: pd.DataFrame, raw: pd.DataFrame, forecast_ids, as_of) -> List[Dict]:
    """
    Args:
        meta: sensors.csv (sensor_id, name, lat, lon, ...).
        raw: hourly readings (columns = sensor ids).
        forecast_ids: ids that have a forecast.
        as_of: nothing after this time is used.

    Returns:
        GET /v1/sensors response.
    """
    out = []
    raw_now = raw.loc[: as_of - pd.Timedelta(hours=1)]
    for row in meta.itertuples():
        col = raw_now[row.sensor_id] if row.sensor_id in raw_now else pd.Series(dtype=float)
        last = col.last_valid_index()
        out.append({"id": str(row.sensor_id), "name": row.name, "lat": round(row.lat, 5), "lon": round(row.lon, 5),
                    "last_reading": None if last is None else {"time": iso(last), "pm25": _r(col[last])},
                    "forecast_available": row.sensor_id in set(forecast_ids)})
    return out


def forecasts_payload(forecasts: List[SensorForecast], names: Dict, issued_at) -> Dict:
    """
    Args:
        forecasts: run_forecasts() results.
        names: {sensor_id: place name}.
        issued_at: when the forecasts were made.

    Returns:
        GET /v1/forecasts response.
    """
    return {"issued_at": iso(issued_at), "sensors": [
        {"id": str(f.inp.sensor_id), "name": names.get(f.inp.sensor_id, ""),
         "today": day_outlook(f.summary["today"]), "tomorrow": day_outlook(f.summary["tomorrow"])}
        for f in forecasts]}


def forecast_detail_payload(f: SensorForecast, name: str, issued_at) -> Dict:
    """
    Args:
        f: one sensor's run_forecasts() result.
        name: its place name.
        issued_at: when the forecast was made.

    Returns:
        GET /v1/forecasts/{sensor_id} response.
    """
    s, start = f.summary["all"], f.forecast_start
    today, tomorrow = day_outlook(f.summary["today"]), day_outlook(f.summary["tomorrow"])
    worst = worst_hours(s["median"], start)
    return {
        "id": str(f.inp.sensor_id), "name": name, "issued_at": iso(issued_at), "forecast_start": iso(start),
        "recent": [{"time": iso(t), "pm25": _r(v)} for t, v in f.inp.recent.items()],
        "hourly": [{"time": iso(start + pd.Timedelta(hours=h)), "pm25": _r(s["median"][h]),
                    "low": _r(s["low"][h]), "high": _r(s["high"][h])} for h in range(HORIZON)],
        "daily": [{"date": start.strftime("%Y-%m-%d"), **today},
                  {"date": (start + pd.Timedelta(hours=DAY)).strftime("%Y-%m-%d"), **tomorrow}],
        "worst_hours": worst,
        "message": message(today, tomorrow, worst),
    }


def replay_payload(forecasts: List[SensorForecast], names: Dict, issued_at, raw: pd.DataFrame,
                   min_real_hours: int = 18) -> Dict:
    """
    /v1/forecasts as it would have looked at `issued_at`, plus what really happened.

    Args:
        forecasts: run_forecasts() made with as_of = issued_at.
        names: {sensor_id: place name}.
        issued_at: the replayed issue time.
        raw: hourly readings, including the hours after issued_at (the outcome).
        min_real_hours: the actual 24-h average is only reported if at least this many
            of the 24 hours have real readings.

    Returns:
        GET /v1/replay response.
    """
    base = forecasts_payload(forecasts, names, issued_at)
    for entry, f in zip(base["sensors"], forecasts):
        truth = raw[f.inp.sensor_id].loc[f.forecast_start: f.forecast_start + pd.Timedelta(hours=DAY - 1)]
        actual = float(truth.mean()) if truth.notna().sum() >= min_real_hours else float("nan")
        entry["today"].update({"actual_mean_24h": _r(actual), "actual_level": level_of(actual)})
    base["mode"] = "replay"
    return base
