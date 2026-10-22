# Handoff: where we stopped (22 Oct 2026)

## Done
- Local tag `gridforecast-demo` (not pushed). Removed: `app/`, its tests, `test_engine.py`, experiment 11.
- Shared package `fmllm/`:
  - `model.py`: identical to experiment 09; loads the ETTh1 checkpoints.
  - `losses.py`, `forecast.py` (with the baselines), `events.py`.
  - `data.py`: gap filling, time split; the scored future must be real readings.
- 40 CPU tests passing.
- Data: `scripts/fetch_openaq.py` downloaded all 65 Accra PM2.5 sensors (450k hourly rows) to `data/raw/openaq_accra/`.
- Experiment 12 (`kaggle/experiments/12_aq_first_train/`): ETTh1 recipe on Accra PM2.5. It passes the smoke test on CPU (`--smoke`).
- `scripts/build_kaggle_bundle.py` builds the `fm-llm-accra-aq` Kaggle dataset (fmllm + data). Not uploaded.
- READMEs updated for phase 2.

## Data decisions made
- **No coverage filter.** Clarity sensors report one reading an hour against OpenAQ's expected four, so a normal hour shows 25% "coverage".
- **History:** gaps up to 12 h are interpolated. **Scored future (first 48 h):** real readings only.
- **Split:** train to Sep 2025 (24 sensors, about 27k windows), validate Oct–Nov 2025 (12 sensors), test Dec 2025 – Feb 2026 Harmattan (18 sensors).
- **Not used yet:** the 45 sensors installed from April 2026 onward. They're a later "unseen locations" test.

## Running
- Kaggle dataset `mackeykumi/fm-llm-accra-aq` uploaded. Experiment 12 (version 2) was pushed on 22 Oct and is training.
- Next step: pull its output (`kaggle kernels output mackeykumi/fm-llm-accra-pm2-5-first-train-t4x2 -p ./output -o`) and go through the results.

## Still open
- `WARN_LIMIT` = 35.5 µg/m³ (the US AQI "unhealthy for sensitive groups" threshold) is a placeholder.
- Sensor bias correction isn't done yet.
