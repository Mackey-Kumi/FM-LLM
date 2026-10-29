# Forecast API contract (v0.1, 23 Oct 2026)

The forecast service runs FM-LLM on Accra's live PM2.5 sensors every hour and serves 48-hour forecasts per sensor. The app only reads from this API.

## Basics
- JSON over HTTPS, GET only, no auth. CORS open (call it from the browser).
- Base URL: `https://mackey333-accra-air-forecast.hf.space` (live, e.g. `/v1/health`). The example responses in `api_examples/` have the same format, for building offline.
- Times: ISO 8601 UTC, which is also Ghana time, e.g. `2026-10-30T14:00:00Z`
- PM2.5 in µg/m³, 1 decimal. Probabilities 0–1, 2 decimals. `null` = not available.
- New forecasts every hour at ~:05. Polling every 15 min is enough.
- Errors: normal HTTP status + `{"error": "..."}`, e.g. 404 for an unknown sensor.

## Levels (24-hour average PM2.5)
| `level` | µg/m³ | Basis |
|---|---|---|
| `good` | < 15 | WHO daily guideline |
| `moderate` | 15 – 35 | — |
| `unhealthy` | 35 – 55.5 | above Ghana's legal standard (GS 1236:2019) |
| `very_unhealthy` | ≥ 55.5 | US air-quality index "unhealthy" |

## Endpoints

### GET /v1/health
```json
{"status": "ok", "model": "fm-llm-aq-run2", "last_update": "2026-10-30T13:05:00Z",
 "sensors_forecast": 56, "likelihood_status": "uncalibrated"}
```
`likelihood_status` is `"calibrated"`: the probabilities and ranges are checked against real outcomes (when it says 60%, it happened ~60% of the time). If it ever says `"uncalibrated"`, hide the probabilities or mark them "experimental".

### GET /v1/sensors (map)
```json
[{"id": "10331004", "name": "Graphic Road", "lat": 5.5585, "lon": -0.2002,
  "last_reading": {"time": "2026-10-30T12:00:00Z", "pm25": 21.4},
  "forecast_available": true, "in_training": true}]
```
- `name` = sensor location (market, school, junction). Use it as the area label.
- `forecast_available: false` = too many recent gaps. Show it grey.
- `in_training`: `true` = one of the 24 sensors the model was trained on (data before 1 Oct 2025); `false` = the model never saw this sensor's data. Also on every entry of `/v1/forecasts`, `/v1/forecasts/{id}` and `/v1/replay`.

### GET /v1/forecasts (all sensors, one line each)
```json
{"issued_at": "2026-10-30T13:05:00Z",
 "sensors": [{"id": "10331004", "name": "Graphic Road",
              "today":    {"mean_24h": 24.8, "level": "moderate", "prob_unhealthy": 0.08},
              "tomorrow": {"mean_24h": 31.2, "level": "moderate", "prob_unhealthy": 0.31},
              "in_training": true}]}
```
- `today` = the next 24 hours from the forecast start; `tomorrow` = the 24 hours after that (rolling, not calendar days).
- `prob_unhealthy` = chance that period's average reaches 35.

### GET /v1/forecasts/{sensor_id} (detail screen)
```json
{"id": "10331004", "name": "Graphic Road", "issued_at": "2026-10-30T13:05:00Z",
 "forecast_start": "2026-10-30T13:00:00Z",
 "recent":   [{"time": "2026-10-28T13:00:00Z", "pm25": 18.2}],
 "hourly":   [{"time": "2026-10-30T13:00:00Z", "pm25": 22.1, "low": 17.5, "high": 27.9}],
 "daily":    [{"date": "2026-10-30", "mean_24h": 24.8, "level": "moderate", "prob_unhealthy": 0.08},
              {"date": "2026-10-31", "mean_24h": 31.2, "level": "moderate", "prob_unhealthy": 0.31}],
 "worst_hours": {"start": "2026-10-31T05:00:00Z", "end": "2026-10-31T09:00:00Z", "pm25_peak": 41.0},
 "message": "Moderate air today. Tomorrow morning (05:00–09:00) may reach unhealthy levels near this sensor."}
```
- `recent`: last 48 h of real readings (`null` = missing hour).
- `hourly`: next 48 h. `pm25` = forecast line; `low`/`high` = likely range (shaded band).
- `forecast_start`: sensors report ~2 h late, so the forecast starts the hour after the last reading.
- `worst_hours`: the window to warn about (usually early morning).
- `message`: ready-made sentence; use as-is or build your own.

### GET /v1/replay?date=YYYY-MM-DD (demo mode)
Shows any past day as if it were live: the same shape as `/v1/forecasts`, issued at 00:00 that day using only data up to then, plus what actually happened (`actual_*`, `null` if the sensor had fewer than 18 real hours). Valid dates: 2023-08-13 to yesterday (the 400 error lists them). Fair dates are **1 Dec 2025 onwards**: the model trained on everything before 1 Oct 2025.
```json
{"issued_at": "2026-01-14T00:00:00Z", "mode": "replay",
 "sensors": [{"id": "10331004", "name": "Graphic Road",
              "today": {"mean_24h": 41.3, "level": "unhealthy", "prob_unhealthy": 0.74,
                        "actual_mean_24h": 44.0, "actual_level": "unhealthy"},
              "tomorrow": {"mean_24h": 39.8, "level": "unhealthy", "prob_unhealthy": 0.65},
              "in_training": true}]}
```
Demo dates (pre-computed, answer in under a second):
- Harmattan test, sensors the model trained on: **2026-01-29** worst day, **2026-02-26** borderline, **2026-02-16** clean.
- Sensors the model never trained on (`in_training: false`): **2026-09-05** (37 such sensors), **2026-08-25** (13, one unhealthy).

## Decided
- Hosting: Hugging Face Space (free; the model runs on CPU). Backup: my laptop with cached forecasts. The Space sleeps after 48 h without visitors and takes a few minutes to wake; I'll wake it before the demo.
- After any restart, `/v1/health` returns 503 with `"status": "starting"` for about 2 minutes, until the first forecasts are ready. Show a "loading" state for any 503.
- No push notifications for the demo. Warnings show when the app is opened.
- Changes to this contract: we agree first, then bump to /v2.
