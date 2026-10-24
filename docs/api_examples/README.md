# Example API responses

Made by the real model (`fm-llm-aq-run2`) on real OpenAQ data, in exactly the format of `../api_contract.md`.
Generated 2026-10-24 00:45 by `scripts/make_api_examples.py`.

| File | Endpoint |
|---|---|
| `health.json` | `GET /v1/health` |
| `sensors.json` | `GET /v1/sensors` |
| `forecasts.json` | `GET /v1/forecasts` |
| `forecast_10331004.json` | `GET /v1/forecasts/10331004` (Graphic Road) |
| `replay_2026-02-16.json` | `GET /v1/replay?date=2026-02-16`: clean day (network average 19.5 µg/m³) |
| `replay_2026-02-26.json` | `GET /v1/replay?date=2026-02-26`: borderline day (35.3 µg/m³) |
| `replay_2026-01-29.json` | `GET /v1/replay?date=2026-01-29`: worst Harmattan day (44.5 µg/m³) |

Notes:
- The "live" files are issued at 2026-10-22 21:00 UTC, the end of the last data download. The real service is current to the hour.
- Probabilities and ranges are calibrated (Method B: the model's real errors on validation; see experiment 15). `likelihood_status` is `"calibrated"`.
- The three replay files are the suggested demo dates (clean / borderline / bad).
