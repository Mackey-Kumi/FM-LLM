"""
The live forecast service: keeps Accra's PM2.5 readings up to date and re-forecasts every hour.

One LiveService object does everything the API needs:
  1. starts from the downloaded history (data/raw/openaq_accra, or SERVICE_DATA_DIR),
  2. every hour fetches each sensor's new hours from OpenAQ (scripts/fetch_openaq.py),
  3. cleans them exactly as in training (fmllm.data.load_hourly / fill_short_gaps),
  4. forecasts every sensor that qualifies with Method B (fmllm.payloads.run_forecasts),
  5. stores the finished responses, so API requests only read from memory.

Replay days (the demo mode) are forecast on request and kept; the three demo dates and
the whole Harmattan test period are worked out in the background after the first live run.
"""

import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

# pylint: disable=wrong-import-position
import fetch_openaq

from fmllm.aq import MAX_GAP, SPLIT
from fmllm.calibration import RatioCalibration
from fmllm.data import fill_short_gaps, load_hourly, train_scale
from fmllm.model import LlamaBackbone, load_checkpoint
from fmllm.payloads import (forecast_detail_payload, forecasts_payload, health_payload, prepare_inputs, replay_payload,
                            run_forecasts, sensors_payload)
# pylint: enable=wrong-import-position

DATA_DIR = Path(os.environ.get("SERVICE_DATA_DIR", REPO / "data" / "raw" / "openaq_accra"))
CHECKPOINT = Path(os.environ.get("SERVICE_CHECKPOINT", REPO / "kaggle" / "experiments" / "12_aq_first_train" /
                                 "results_run2" / "aq_pm25_best.pt"))
CALIBRATION = Path(os.environ.get("SERVICE_CALIBRATION", REPO / "kaggle" / "experiments" /
                                  "15_aq_ensemble_calibration" / "results" / "calibration.json"))
TRAINING_SENSORS = Path(os.environ.get("SERVICE_TRAINING_SENSORS", CHECKPOINT.parent / "training_sensors.json"))
WORK_DIR = Path(os.environ.get("SERVICE_WORK_DIR", REPO / "outputs" / "service"))
MODEL_NAME = "fm-llm-aq-run2"
DEMO_DATES = ("2026-01-29", "2026-02-26", "2026-02-16",     # Harmattan test: worst, borderline, clean
              "2026-09-05", "2026-08-25")                   # sensors never trained on (docs/api_contract.md)
HARMATTAN = ("2025-12-01", "2026-02-28")
REFRESH_MINUTE = 5            # new forecasts at hh:05
OVERLAP_H = 6                 # re-fetch the last few hours too: sensors report late and OpenAQ fills them in
NEW_SENSOR_DAYS = 35          # history fetched for a sensor first seen live (28 days needed + slack)
DISCOVER_EVERY_H = 24         # how often the sensor list is re-read from OpenAQ


def utc_now() -> pd.Timestamp:
    """Returns: the current time in UTC, without timezone (the convention of every table here)."""
    return pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)


class LiveService:
    """
    Holds the readings, the model and the latest responses. All public methods are safe to
    call from the web server while the hourly loop runs in its own thread.

    Args:
        data_dir: folder with pm25_hourly.csv and sensors.csv (the starting history).
        checkpoint: trained FM-LLM checkpoint.
        calibration: calibration.json (Method B).
        work_dir: writable folder for the updated readings.
        fetch: whether to fetch new hours from OpenAQ (False = serve the starting history only).
    """

    def __init__(self, data_dir=DATA_DIR, checkpoint=CHECKPOINT, calibration=CALIBRATION, work_dir=WORK_DIR,
                 fetch: bool = True):
        self.data_dir, self.checkpoint, self.calibration_path = Path(data_dir), Path(checkpoint), Path(calibration)
        self.work_dir = Path(work_dir)
        self.fetch = fetch
        self.status = "starting"
        self.error: Optional[str] = None
        self.live: Optional[Dict] = None          # {"health", "sensors", "forecasts", "details", "issued_at"}
        self.replays: Dict[str, Dict] = {}
        self._model_lock = threading.Lock()       # one forecast at a time (live loop vs replay requests)
        self._data_lock = threading.Lock()
        self._last_discovery = None
        self._started = False
        self.model = self.calibration = self.scale = None
        self.trained = None                       # ids of the sensors the model was trained on
        self.long = self.meta = self.raw = self.wide = None

    # ------------------------------------------------------------------ data
    def load(self):
        """Load the starting history, the model and the calibration (slow: Llama is downloaded once)."""
        t0 = time.time()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.long = pd.read_csv(self.data_dir / "pm25_hourly.csv")
        self.meta = pd.read_csv(self.data_dir / "sensors.csv")
        self._rebuild()
        self.scale = train_scale(self.wide, SPLIT["train_end"])     # the training scaling, fixed for good
        self.model = load_checkpoint(str(self.checkpoint), LlamaBackbone("cpu"), "cpu")
        self.calibration = RatioCalibration.load(str(self.calibration_path))
        if TRAINING_SENSORS.exists():
            self.trained = set(json.loads(TRAINING_SENSORS.read_text(encoding="utf-8"))["training_sensors"])
        print(f"[service] history + model loaded in {time.time() - t0:.0f}s; data ends {self.raw.index.max()}",
              flush=True)

    def _rebuild(self):
        """Turn the long table of readings into the cleaned hourly tables, exactly as in training."""
        path = self.work_dir / "pm25_hourly.csv"
        self.long.to_csv(path, index=False)
        raw = load_hourly(path)
        wide = fill_short_gaps(raw, max_gap=MAX_GAP)
        with self._data_lock:
            self.raw, self.wide = raw, wide

    def _discover(self, key):
        """
        Add sensors that have started reporting since the starting history was downloaded.

        Args:
            key: OpenAQ API key.
        """
        since = (utc_now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        found = pd.DataFrame(fetch_openaq.pm25_sensors(key, 25000, since))
        new = found[~found.sensor_id.isin(self.meta.sensor_id)]
        if len(new):
            self.meta = pd.concat([self.meta, new], ignore_index=True)
            print(f"[service] {len(new)} new sensor(s): {', '.join(new.name)}", flush=True)
        self._last_discovery = utc_now()

    def update_readings(self) -> int:
        """
        Fetch every sensor's hours since its last stored reading (minus OVERLAP_H) from OpenAQ.
        A sensor that fails is skipped this hour; the others still update.

        Returns:
            Number of new or corrected hourly readings.
        """
        key = fetch_openaq.api_key()
        if self._last_discovery is None or utc_now() - self._last_discovery > pd.Timedelta(hours=DISCOVER_EVERY_H):
            try:
                self._discover(key)
            except Exception:   # pylint: disable=broad-except  # the sensor list is optional; keep the old one
                traceback.print_exc()
        now = utc_now()
        last = pd.to_datetime(self.long.groupby("sensor_id")["hour"].max(), utc=True).dt.tz_localize(None)
        rows = []
        for sid in self.meta.sensor_id:
            start = last[sid] - pd.Timedelta(hours=OVERLAP_H) if sid in last else now - pd.Timedelta(days=NEW_SENSOR_DAYS)
            if now - start > pd.Timedelta(days=NEW_SENSOR_DAYS):     # silent for weeks: only recent hours matter
                start = now - pd.Timedelta(days=NEW_SENSOR_DAYS)
            try:
                rows += [(sid, hour, value, cov) for hour, value, cov in fetch_openaq.sensor_hours(
                    key, sid, start.strftime("%Y-%m-%dT%H:00:00"), now.strftime("%Y-%m-%dT%H:00:00"))]
            except Exception as e:   # pylint: disable=broad-except  # one bad sensor must not stop the hour
                print(f"[service] sensor {sid}: {e}", flush=True)
        if rows:
            new = pd.DataFrame(rows, columns=["sensor_id", "hour", "pm25", "coverage"])
            self.long = (pd.concat([self.long, new], ignore_index=True)
                         .drop_duplicates(["sensor_id", "hour"], keep="last"))
            self._rebuild()
        return len(rows)

    # ------------------------------------------------------------------ forecasts
    def _forecast(self, as_of: pd.Timestamp):
        """
        Args:
            as_of: issue time; nothing after it is used.

        Returns:
            (forecasts, raw) as of that time.
        """
        with self._data_lock:
            raw, wide = self.raw, self.wide
        inputs, _ = prepare_inputs(raw, wide, as_of)
        with self._model_lock:
            return run_forecasts(self.model, inputs, self.scale, calibration=self.calibration), raw

    def refresh(self):
        """Forecast every sensor now and swap in the new responses."""
        t0 = time.time()
        issued = utc_now().floor("min")
        as_of = issued.floor("h") + pd.Timedelta(hours=1)    # every reading up to the current hour counts
        forecasts, raw = self._forecast(as_of)
        names = dict(zip(self.meta.sensor_id, self.meta.name.astype(str).str.strip()))
        ids = [f.inp.sensor_id for f in forecasts]
        self.live = {
            "issued_at": issued,
            "health": health_payload(issued, len(forecasts), MODEL_NAME, calibrated=True),
            "sensors": sensors_payload(self.meta, raw, ids, as_of, self.trained),
            "forecasts": forecasts_payload(forecasts, names, issued, self.trained),
            "details": {str(f.inp.sensor_id): forecast_detail_payload(f, names.get(f.inp.sensor_id, ""), issued,
                                                                      self.trained)
                        for f in forecasts},
        }
        self.status = "ok"
        print(f"[service] {issued}: {len(forecasts)} sensors forecast in {time.time() - t0:.0f}s", flush=True)

    def replay(self, date: str) -> Dict:
        """
        /v1/replay for one day, worked out once and kept.

        Args:
            date: "YYYY-MM-DD".

        Returns:
            The replay response (forecasts issued at 00:00 that day, plus what happened).

        Raises:
            ValueError: bad date, or not enough data before (28 days) or after (24 h) it.
        """
        if date in self.replays:
            return self.replays[date]
        when = pd.Timestamp(date)
        if when.strftime("%Y-%m-%d") != date:
            raise ValueError("date must be YYYY-MM-DD")
        with self._data_lock:
            first, end = self.raw.index.min(), self.raw.index.max()
        if when < first + pd.Timedelta(days=29) or when + pd.Timedelta(hours=24) > end:
            raise ValueError(f"replay needs 28 days of data before the date and 24 hours after it: choose between "
                             f"{(first + pd.Timedelta(days=29)):%Y-%m-%d} and {(end - pd.Timedelta(hours=24)):%Y-%m-%d}")
        forecasts, raw = self._forecast(when)
        names = dict(zip(self.meta.sensor_id, self.meta.name.astype(str).str.strip()))
        self.replays[date] = replay_payload(forecasts, names, when, raw, trained=self.trained)
        return self.replays[date]

    def _warm_replays(self):
        """Work out the demo dates, then every Harmattan day, so replay requests answer instantly."""
        days = list(DEMO_DATES) + [d.strftime("%Y-%m-%d") for d in pd.date_range(*HARMATTAN)]
        for d in dict.fromkeys(days):
            try:
                self.replay(d)
            except ValueError:
                pass

    # ------------------------------------------------------------------ loop
    def run_forever(self):
        """Load, then update + forecast every hour at hh:05. Errors keep the last good forecasts."""
        try:
            self.load()
        except Exception as e:   # pylint: disable=broad-except  # report it through /v1/health
            traceback.print_exc()
            self.status, self.error = "error", f"start-up failed: {e}"
            return
        warmed = False
        while True:
            try:
                if self.fetch:
                    print(f"[service] fetched {self.update_readings()} hourly readings", flush=True)
                self.refresh()
                self.error = None
            except Exception as e:   # pylint: disable=broad-except  # keep serving the last forecasts
                traceback.print_exc()
                self.error = f"last update failed: {e}"
            if not warmed:
                warmed = True
                threading.Thread(target=self._warm_replays, daemon=True).start()
            now = datetime.now(timezone.utc)
            wake = (now + timedelta(hours=1)).replace(minute=REFRESH_MINUTE, second=0, microsecond=0)
            if now.minute < REFRESH_MINUTE:
                wake -= timedelta(hours=1)
            time.sleep(max(60.0, (wake - now).total_seconds()))

    def start(self):
        """Run the loop in a background thread (returns at once). Calling it again does nothing."""
        if self._started:
            return
        self._started = True
        threading.Thread(target=self.run_forever, daemon=True).start()

    def detail(self, sensor_id: str) -> Optional[Dict]:
        """
        Args:
            sensor_id: sensor id as text.

        Returns:
            Its /v1/forecasts/{id} response, or None if it has no forecast.
        """
        return (self.live or {}).get("details", {}).get(sensor_id)

    def known(self, sensor_id: str) -> bool:
        """
        Args:
            sensor_id: sensor id as text.

        Returns:
            Whether the sensor exists in the sensor list.
        """
        return self.meta is not None and sensor_id in set(self.meta.sensor_id.astype(str))
