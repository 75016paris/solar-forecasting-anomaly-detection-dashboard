# TODO

## Portfolio/publication checklist

- [x] Generate name-redacted `data/inverter_plants.csv`.
- [x] Restore name-redacted `data/inverter_five_minutes_generation_logs.csv`.
- [x] Restore name-redacted reference source CSVs for source-agreement diagnostics.
- [x] Restore name-redacted `open_data/gazipur_weather.csv`.
- [x] Regenerate `clean/*` from local data.
- [x] Regenerate `models/*` from local data.
- [x] Re-run compile, data-loading and Streamlit smoke checks.
- [x] Update `README.md` from local cleanup wording to public demo wording.
- [x] Add dashboard screenshots for portfolio and GitHub README.

## App/code cleanup

- [x] Add a cleaned local data pipeline before model tuning (`src/data_cleaning.py`, `build_clean_dataset.py`).
- [x] Add a conservative daily source-agreement filter for model training/evaluation.
- [x] Label model metrics as reliable-days performance and flag plants with source-data issues.
- [x] Run bounded tuning experiment for reliable plants; keep no-grid models because tuning did not improve R².
- [ ] Rename `app_solar_monitoring_enhanced.py` to a cleaner public entrypoint, e.g. `app.py` or `dashboard.py`.
- [ ] Consider moving hard-coded plant configuration into `src/config.py`.
- [ ] Add a small smoke-test script for loading all plants and checking key dashboard assumptions.
- [ ] Review the upload/retraining flow: keep for local operator use or remove for a simpler portfolio demo.

## Future product/data ideas

- Explore usage of weather-station-shaped local data.
- Expand to more plants only if the demo scope needs it.
- Explore replacement/imputation of missing data without skewing model evaluation.
- Decide whether Plant E should get a separate billing-meter daily fallback model.
- Track score evolution when new data is added and models are retrained.
- Explore different models/horizons, e.g. fast Ridge/RandomForest for short-term prediction and neural approaches only if they clearly improve results.
