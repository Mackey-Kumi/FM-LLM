"""
Write example responses for every endpoint in docs/api_contract.md, made by the real
model on real data (on CPU), so the app can be built against realistic numbers.

    python scripts/make_api_examples.py
    python scripts/make_api_examples.py --checkpoint path/to/aq_pm25_best.pt --samples 30

Writes docs/api_examples/:
  health.json, sensors.json, forecasts.json, forecast_<sensor_id>.json,
  replay_<date>.json for three highlight days, README.md (what each file is).

"Live" examples are issued at the end of the downloaded data (data/raw/openaq_accra),
so they're as current as the last fetch_openaq.py run. The live service makes the same
responses with the same code (fmllm/payloads.py) from fresh data every hour.
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# pylint: disable=wrong-import-position
import pandas as pd

from fmllm.aq import MAX_GAP, SPLIT
from fmllm.data import fill_short_gaps, load_hourly, train_scale
from fmllm.model import LlamaBackbone, load_checkpoint
from fmllm.payloads import (UNHEALTHY, forecast_detail_payload, forecasts_payload, health_payload,
                            prepare_inputs, replay_payload, run_forecasts, sensors_payload)
# pylint: enable=wrong-import-position

DATA = REPO / "data" / "raw" / "openaq_accra"
OUT = REPO / "docs" / "api_examples"
CHECKPOINT = REPO / "kaggle" / "experiments" / "12_aq_first_train" / "results_run2" / "aq_pm25_best.pt"
MODEL_NAME = "fm-llm-aq-run2"
DETAIL_SENSOR = 10331004   # Graphic Road: a long-running sensor, used for the detail example if available


def write(name, payload):
    """
    Args:
        name: file name inside docs/api_examples/.
        payload: JSON-serialisable response.
    """
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print(f"  wrote {name}")


def highlight_days(raw: pd.DataFrame):
    """
    Pick three replay dates in the Harmattan test period from the real data: the cleanest
    day, the day closest to Ghana's 35 µg/m³ standard, and the worst day (by the median
    of all sensors' daily averages, counting sensors with >= 18 real hours that day).

    Args:
        raw: hourly readings.

    Returns:
        {"clean": date, "borderline": date, "bad": date} as "YYYY-MM-DD" strings, plus their
        network daily values under "values".
    """
    period = raw.loc[SPLIT["val_end"]: pd.Timestamp(SPLIT["test_end"]) - pd.Timedelta(hours=1)]
    daily = period.resample("D").mean().where(period.resample("D").count() >= 18)
    network = daily.median(axis=1).dropna()
    picks = {"clean": network.idxmin(), "borderline": (network - UNHEALTHY).abs().idxmin(), "bad": network.idxmax()}
    return {k: v.strftime("%Y-%m-%d") for k, v in picks.items()} | {
        "values": {k: round(float(network[v]), 1) for k, v in picks.items()}}


def main():
    """Build and write every example file (see the module docstring)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CHECKPOINT))
    ap.add_argument("--samples", type=int, default=30, help="ensemble size")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    raw = load_hourly(DATA / "pm25_hourly.csv")
    wide = fill_short_gaps(raw, max_gap=MAX_GAP)
    scale = train_scale(wide, SPLIT["train_end"])
    meta = pd.read_csv(DATA / "sensors.csv")
    names = dict(zip(meta.sensor_id, meta.name.str.strip()))
    model = load_checkpoint(args.checkpoint, LlamaBackbone("cpu"), "cpu")
    print(f"data + model loaded in {time.time() - t0:.0f}s")

    # live examples, issued just after the last downloaded hour
    as_of = raw.index.max() + pd.Timedelta(hours=1)
    inputs, skipped = prepare_inputs(raw, wide, as_of)
    t1 = time.time()
    forecasts = run_forecasts(model, inputs, scale, n_samples=args.samples)
    print(f"live: {len(forecasts)} sensors forecast, {len(skipped)} skipped, {time.time() - t1:.0f}s")
    issued = as_of
    write("health.json", health_payload(issued, len(forecasts), MODEL_NAME, calibrated=False))
    write("sensors.json", sensors_payload(meta, raw, [f.inp.sensor_id for f in forecasts], as_of))
    write("forecasts.json", forecasts_payload(forecasts, names, issued))
    detail = next((f for f in forecasts if f.inp.sensor_id == DETAIL_SENSOR), forecasts[0])
    write(f"forecast_{detail.inp.sensor_id}.json", forecast_detail_payload(detail, names.get(detail.inp.sensor_id, ""),
                                                                            issued))

    # replay examples for three highlight days
    days = highlight_days(raw)
    for kind in ("clean", "borderline", "bad"):
        when = pd.Timestamp(days[kind])
        r_inputs, _ = prepare_inputs(raw, wide, when)
        r_forecasts = run_forecasts(model, r_inputs, scale, n_samples=args.samples)
        write(f"replay_{days[kind]}.json", replay_payload(r_forecasts, names, when, raw))

    readme = f"""# Example API responses

Made by the real model (`{MODEL_NAME}`) on real OpenAQ data, in exactly the format of `../api_contract.md`.
Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by `scripts/make_api_examples.py`.

| File | Endpoint |
|---|---|
| `health.json` | `GET /v1/health` |
| `sensors.json` | `GET /v1/sensors` |
| `forecasts.json` | `GET /v1/forecasts` |
| `forecast_{detail.inp.sensor_id}.json` | `GET /v1/forecasts/{detail.inp.sensor_id}` ({names.get(detail.inp.sensor_id, "")}) |
| `replay_{days['clean']}.json` | `GET /v1/replay?date={days['clean']}`: clean day (network average {days['values']['clean']} µg/m³) |
| `replay_{days['borderline']}.json` | `GET /v1/replay?date={days['borderline']}`: borderline day ({days['values']['borderline']} µg/m³) |
| `replay_{days['bad']}.json` | `GET /v1/replay?date={days['bad']}`: worst Harmattan day ({days['values']['bad']} µg/m³) |

Notes:
- The "live" files are issued at {iso_time(issued)}, the end of the last data download. The real service is current to the hour.
- `likelihood_status` is `"uncalibrated"`: the probabilities are real model output but not yet checked, so treat them as placeholders.
- The three replay files are the suggested demo dates (clean / borderline / bad).
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s; highlight days: {days}")


def iso_time(ts):
    """
    Args:
        ts: a timestamp.

    Returns:
        "YYYY-MM-DD HH:MM UTC" text for the README.
    """
    return f"{pd.Timestamp(ts):%Y-%m-%d %H:%M} UTC"


if __name__ == "__main__":
    main()
