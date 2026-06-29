# Solar Forecasting & Anomaly Detection Dashboard

This is a Streamlit dashboard prototype for photovoltaic production forecasting, anomaly detection, data-quality analysis and model comparison.

This portfolio version uses project-shaped CSV values with company/client names, plant labels and plant IDs replaced by neutral demo identifiers. The underlying generation/weather magnitudes are preserved so the dashboard remains realistic.

> Privacy note: this is **name-redacted operational-style data**, not fully synthetic data. It is included only to demonstrate the forecasting/anomaly-detection workflow.

## Screenshots

Screenshots below use the name-redacted local demo data included in this repository.

![Solar forecasting dashboard executive summary](docs/screenshots/executive-summary.png)

![Solar dashboard forecast outlook and production trend](docs/screenshots/forecast-outlook.png)

![Solar dashboard recent production performance](docs/screenshots/production-performance.png)

## What it demonstrates

- Python data workflow with pandas and NumPy.
- Feature engineering for solar production forecasting.
- scikit-learn model training, comparison and saved model loading.
- pvlib solar-position features.
- Streamlit + Plotly dashboarding.
- Data completeness checks and anomaly-oriented monitoring.
- Practical refactor from exploratory work into reusable `src/` modules.

## Dashboard scope

The dashboard focuses on five name-redacted photovoltaic plants:

- Plant A
- Plant B
- Plant C
- Plant D
- Plant E

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app_solar_monitoring_enhanced.py
```

Then open the local Streamlit URL, usually:

```text
http://localhost:8501
```

An OpenWeather API key is optional for the live forecast tab. Without it, the historical monitoring, model and data-quality tabs still run from local CSV/model artifacts.

Create a local `.env` if needed:

```text
ow_key=YOUR_OPENWEATHER_KEY
ow_email=YOUR_EMAIL_OPTIONAL
```

Do not commit `.env` files.

## Rebuild cleaned data and models

Build cleaned pipeline outputs:

```bash
python3 build_clean_dataset.py
```

Train all plant models:

```bash
python3 train_all_models.py
```

Local restoration helpers that reference private source paths are intentionally kept outside the public repo.

## Verification

Useful smoke checks:

```bash
python3 -m compileall app_solar_monitoring_enhanced.py src train_model.py train_all_models.py build_clean_dataset.py
python3 build_clean_dataset.py
streamlit run app_solar_monitoring_enhanced.py --server.headless true --server.port 8505
curl http://localhost:8505/_stcore/health
```

Expected health result:

```text
ok
```

## Repository structure

```text
app_solar_monitoring_enhanced.py  Streamlit dashboard
build_clean_dataset.py            generate cleaned clean/*.csv files
src/                              modular data/model/forecasting utilities
data/                             name-redacted plant metadata + generation/reference data
open_data/                        local weather history used by the app
models/                           saved model artifacts and metrics
docs/screenshots/                 dashboard screenshots
train_model.py                    train one default plant model
train_all_models.py               train all configured plant models
DATA_SCHEMA.md                    data schema notes
```

## Data and privacy status

Current local state:

- `data/` and `open_data/` use project-shaped values with visible company/client names, plant labels and plant IDs removed;
- `clean/` is generated from that local data;
- `models/` are regenerated from that local data;
- `.env`, raw private data folders, notebooks and old history are not part of this mirror.

## Status

Prototype/portfolio demo. The project demonstrates data cleaning, feature engineering, dashboarding, model comparison and anomaly-monitoring thinking; it is not presented as a production solar monitoring platform.

## License

Portfolio/source-available material. Do not reuse operational data without permission.
