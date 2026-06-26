"""
Streamlit dashboard for photovoltaic production forecasting and anomaly review.

The app loads plant generation data, weather history, trained scikit-learn models,
and reusable helpers from `src/` to present monitoring, forecasting, model and
data-quality views.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import html
import os
from dotenv import load_dotenv

# Import from the modular data/model architecture
from src import (
    data_loader,
    feature_engineering,
    forecasting,
    visualization,
    data_quality,
    model_loader,
    metrics,
)
from src.config import PLANT_CONFIG, DATA_PATHS, ML_CONFIG

# Load environment variables
load_dotenv()

# Anonymized demo plant configuration
DEMO_PLANTS = {
    'Plant A': {'plant_id': 900001, 'capacity_kwp': 269.28},
    'Plant B': {'plant_id': 900002, 'capacity_kwp': 522.72},
    'Plant C': {'plant_id': 900003, 'capacity_kwp': 227.04},
    'Plant D': {'plant_id': 900004, 'capacity_kwp': 525.36},
    'Plant E': {'plant_id': 900005, 'capacity_kwp': 285.12}
}

QUALITY_WARNING_THRESHOLDS = {
    'min_data_availability_pct': 60,
    'min_test_r2': 0.0,
    'max_test_wape_pct': 70,
}

def _infer_weather_snapshot_today() -> pd.Timestamp:
    """Use the last timestamp in the local Gazipur weather CSV as the snapshot date."""
    fallback = pd.Timestamp('2025-10-29', tz=PLANT_CONFIG['timezone'])
    try:
        weather_path = Path(DATA_PATHS['weather'])
        if not weather_path.exists():
            return fallback
        weather_dates = pd.read_csv(weather_path, usecols=['dt_iso'])
        timestamps = pd.to_datetime(
            weather_dates['dt_iso'].str.replace(' UTC', '', regex=False),
            utc=True,
            errors='coerce'
        ).dt.tz_convert(PLANT_CONFIG['timezone'])
        latest = timestamps.max()
        return latest.floor('D') if pd.notna(latest) else fallback
    except Exception:
        return fallback


# This project uses a fixed name-redacted historical snapshot. Treat the
# weather snapshot end date as "today" so freshness labels describe the local
# data rather than the real current date.
DATA_SNAPSHOT_TODAY = _infer_weather_snapshot_today()

# Page configuration
st.set_page_config(
    page_title="Solar Forecasting & Anomaly Detection",
    page_icon="☀️",
    layout="wide"
)

# Load user preferences from session state
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 10
if 'selected_plant' not in st.session_state:
    st.session_state.selected_plant = 'Plant A'
if 'alert_email' not in st.session_state:
    st.session_state.alert_email = ''

PLANT_SELECTOR_KEYS = [
    'executive_plant_selector',
    'quality_detail_plant_selector',
    'overview_plant_selector',
    'performance_plant_selector',
    'model_plant_selector',
    'forecast_plant_selector',
    'alerts_plant_selector',
]


def set_selected_plant(plant_name: str):
    """Keep all tab-level plant controls aligned without navigating away from the current tab."""
    st.query_params.clear()
    st.session_state.selected_plant = plant_name
    for key in PLANT_SELECTOR_KEYS:
        st.session_state[key] = plant_name


def _style_completeness_cell(value):
    """Light, readable table styling for data completeness."""
    if pd.isna(value):
        return ''
    if value >= 85:
        return 'background-color: #d8f3dc; color: #1b4332;'
    if value >= 60:
        return 'background-color: #fff3bf; color: #5c4400;'
    return 'background-color: #ffd6d6; color: #6b0000;'


def _style_r2_cell(value):
    """Light, readable table styling for R² where higher is better."""
    if pd.isna(value):
        return ''
    if value >= 0.5:
        return 'background-color: #d8f3dc; color: #1b4332;'
    if value >= 0:
        return 'background-color: #fff3bf; color: #5c4400;'
    return 'background-color: #ffd6d6; color: #6b0000;'


def _style_wape_cell(value):
    """Light, readable table styling for WAPE where lower is better."""
    if pd.isna(value):
        return ''
    if value <= 50:
        return 'background-color: #d8f3dc; color: #1b4332;'
    if value <= 70:
        return 'background-color: #fff3bf; color: #5c4400;'
    return 'background-color: #ffd6d6; color: #6b0000;'


def _quality_overview_cache_token() -> tuple:
    """Cache token that changes when generated quality inputs change."""
    paths = [
        DATA_PATHS['data_quality_summary'],
        DATA_PATHS['generation_5m'],
        'data/inverter_daily_generation_logs.csv',
        'data/plants_billing_meter_logs.csv',
    ]
    return tuple(_file_mtime(path) for path in paths)


def _model_summary_cache_token() -> tuple:
    """Cache token that changes when model artifacts are regenerated."""
    paths = []
    for plant_name in DEMO_PLANTS:
        model_dir = model_loader.get_plant_directory(plant_name)
        paths.extend([
            model_dir / 'model_comparison.csv',
            model_dir / 'feature_columns.pkl',
            model_dir / 'plant_info.pkl',
        ])
    return tuple(_file_mtime(str(path)) for path in paths)


def _get_plant_quality_card_data() -> dict:
    """Return compact quality metadata for plant selector cards."""
    try:
        overview_df = build_all_plants_quality_overview(_quality_overview_cache_token())
    except Exception:
        overview_df = pd.DataFrame()

    quality = {}
    for plant_name in DEMO_PLANTS:
        row = overview_df[overview_df['Plant'] == plant_name] if not overview_df.empty else pd.DataFrame()
        if row.empty:
            # Last-resort fallback from the generated clean summary prevents stale
            # Streamlit cache entries from rendering Unknown/n/a cards after a data rebuild.
            try:
                summary = pd.read_csv(DATA_PATHS['data_quality_summary'])
                row = summary[summary['plant_name'] == plant_name]
            except Exception:
                row = pd.DataFrame()

            if row.empty:
                quality[plant_name] = {
                    'band': 'Needs rebuild',
                    'completeness': 'n/a',
                    'warnings': 'Run build_clean_dataset.py',
                }
                continue

            row = row.iloc[0]
            completeness = row['usable_daylight_modeling_hours_pct']
            band = '🟢 Good' if completeness >= 85 else '🟡 Usable' if completeness >= 60 else '🔴 Poor'
            quality[plant_name] = {
                'band': band,
                'completeness': f"{completeness:.1f}%" if pd.notna(completeness) else 'n/a',
                'warnings': row.get('warnings', 'None'),
            }
            continue

        row = row.iloc[0]
        completeness = row['Usable daylight data (%)']
        quality[plant_name] = {
            'band': row['Data quality band'],
            'completeness': f"{completeness:.1f}%" if pd.notna(completeness) else 'n/a',
            'warnings': row['Warnings'],
        }
    return quality


def render_plant_selector(selector_key: str, label: str = '🏭 Select plant') -> str:
    """Render compact clickable plant cards that preserve the current tab on rerun."""
    plant_names = list(DEMO_PLANTS.keys())
    current = st.session_state.get('selected_plant', plant_names[0])
    if current not in plant_names:
        current = plant_names[0]
        st.session_state.selected_plant = current

    st.markdown(f"**{label}**")
    st.markdown('''
        <style>
            div[data-testid="stButton"] > button {
                min-height: 4.1rem;
                padding: 0.35rem 0.45rem;
                line-height: 1.08;
                white-space: pre-line;
                text-align: left;
                font-size: 0.74rem;
                border-radius: 0.45rem;
            }
            div[data-testid="stButton"] > button:disabled {
                background: #f1f3f5;
                border-color: #6c757d;
                color: #222;
                opacity: 1;
                box-shadow: inset 0 0 0 1px #6c757d;
            }
        </style>
    ''', unsafe_allow_html=True)

    quality = _get_plant_quality_card_data()
    cols = st.columns(len(plant_names), gap='small')
    for col, plant_name in zip(cols, plant_names):
        plant_quality = quality[plant_name]
        selected = plant_name == current
        warning_badge = ' ⚠️' if plant_quality['warnings'] != 'None' else ''
        label_text = (
            f"{plant_name}{warning_badge}\n"
            f"{DEMO_PLANTS[plant_name]['capacity_kwp']:.0f} kWp\n"
            f"{plant_quality['band']}\n"
            f"Completeness: {plant_quality['completeness']}"
        )

        with col:
            st.button(
                label_text,
                key=f"{selector_key}_{plant_name}",
                disabled=selected,
                use_container_width=True,
                on_click=set_selected_plant,
                args=(plant_name,),
            )

    return st.session_state.selected_plant

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 1.9rem;
        font-weight: 700;
        color: #FF6B35;
        text-align: center;
        margin: 0.15rem 0 0.35rem 0;
        line-height: 1.15;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B35;
    }
    .alert-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e2e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 6px 10px;
        font-size: 0.9rem;
    }
    div[data-testid="stMetric"] {
        min-width: 0;
        container-type: inline-size;
    }
    div[data-testid="stMetricValue"] {
        min-width: 0;
        max-width: 100%;
        overflow: hidden;
    }
    div[data-testid="stMetricValue"] p {
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        font-size: clamp(0.95rem, 7cqw, 2.25rem) !important;
        line-height: 1.2;
    }
    </style>
""", unsafe_allow_html=True)


# load_plant_model() moved to src/model_loader.py



@st.cache_data(ttl=300)  # Reduced to 5 minutes for data quality issues
def load_raw_5min_data(plant_name='Plant A'):
    """Load raw 5-minute generation data without hourly aggregation for completeness analysis"""
    try:
        # Load plants data
        df_plants = pd.read_csv(DATA_PATHS['inverter_plants'])
        df_plants['plant_address'] = df_plants['plant_address'].fillna('Solar Region')

        # Load 5-minute generation data
        df = pd.read_csv(DATA_PATHS['generation_5m'])

        # Merge with plant info
        df['generation_date'] = pd.to_datetime(df['generation_date'])
        df = df.merge(df_plants[['plant_id', 'plant_name']], on='plant_id', how='left')
        df = df[df['plant_name'] == plant_name].copy()

        # Clean generation values
        df['generation_amount'] = df['generation_amount'].astype(str).str.replace(',', '').astype(float)
        df['generation_kwh'] = df['generation_amount'] / 1000

        # Handle timezone - NO DATE FILTERING for quality audit
        # Standardized approach: localize to UTC if naive, then convert to local timezone
        if df['generation_date'].dt.tz is None:
            df['generation_date'] = df['generation_date'].dt.tz_localize('UTC')
        else:
            df['generation_date'] = df['generation_date'].dt.tz_convert('UTC')

        # Convert to Asia/Dhaka timezone
        df['generation_date'] = df['generation_date'].dt.tz_convert('Asia/Dhaka')

        # Data Quality Checks (before any modifications)
        original_count = len(df)
        date_min = df['generation_date'].min()
        date_max = df['generation_date'].max()

        # Detect duplicates (but keep them for audit)
        duplicates_count = df.duplicated(subset='generation_date', keep=False).sum()

        # Sort by date
        df = df.sort_values('generation_date')

        # Remove duplicates (keep last occurrence)
        df_clean = df.drop_duplicates(subset='generation_date', keep='last')
        removed_duplicates = len(df) - len(df_clean)

        # Debug info
        print(f"\n📊 Data Loading Summary for {plant_name}:")
        print(f"   Date range: {date_min} to {date_max}")
        print(f"   Total records: {original_count:,}")
        print(f"   Duplicate timestamps: {duplicates_count:,}")
        print(f"   After deduplication: {len(df_clean):,}")

        # Set index
        df_clean = df_clean.set_index('generation_date')

        # Return both raw and clean dataframes for quality analysis
        df = df_clean

        return df
    except Exception as e:
        st.error(f"Error loading 5-minute data: {e}")
        return None

# audit_data_quality() moved to src/data_quality.py


# analyze_data_completeness() moved to src/data_quality.py



@st.cache_data(ttl=3600)
def load_data(plant_name='Plant A'):
    """Load the processed hourly data with predictions for a specific plant"""
    try:
        # Load generation data for the plant
        df = data_loader.load_generation_data(plant_name=plant_name)

        # Ensure index is datetime
        if 'generation_date' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            df = df.set_index('generation_date')

        # Load plant-specific model
        model, feature_columns = model_loader.load_plant_model(plant_name)

        if model is not None and feature_columns is not None:
            # Generate predictions using plant-specific model
            try:
                # Load weather and merge
                df_weather = data_loader.load_weather_data()
                df_temp = df.reset_index()
                df_temp = df_temp.merge(df_weather, on='generation_date', how='left')
                df_temp = data_loader.interpolate_weather(df_temp)

                # Get plant capacity
                plant_capacity = DEMO_PLANTS[plant_name]['capacity_kwp']

                # Create features with correct plant capacity
                df_temp = feature_engineering.create_all_features(df_temp, capacity_kwp=plant_capacity)

                # Prepare features for prediction (validate columns exist)
                missing_cols = set(feature_columns) - set(df_temp.columns)
                if missing_cols:
                    st.error(f"⚠️ Missing features for {plant_name}: {missing_cols}")
                    raise ValueError(f"Missing required features: {missing_cols}")

                X = df_temp[feature_columns]
                X = X.ffill().bfill().fillna(0)

                # Generate predictions
                predictions = model.predict(X)
                predictions = np.maximum(predictions, 0)

                # Add predictions to dataframe
                df_temp['ml_predicted_kwh'] = predictions
                df = df_temp.set_index('generation_date')

            except Exception as e:
                st.warning(f"Could not generate predictions for {plant_name}: {e}")
                df['ml_predicted_kwh'] = df['generation_kwh']
        else:
            # Model not found, use dummy predictions
            st.warning(f"⚠️ Model not found for {plant_name}. Using actual values as predictions.")
            df['ml_predicted_kwh'] = df['generation_kwh']

        # Add clearsky estimate if not present (calculate properly using solar position)
        if 'clearsky_expected_kwh' not in df.columns:
            try:
                plant_capacity = DEMO_PLANTS[plant_name]['capacity_kwp']
                df_temp = df.reset_index()
                df_temp = feature_engineering.add_solar_position(df_temp, capacity_kwp=plant_capacity)
                df['clearsky_expected_kwh'] = df_temp.set_index('generation_date')['clearsky_expected_kwh']
            except Exception as e:
                st.warning(f"Could not calculate clear-sky for {plant_name}: {e}")
                df['clearsky_expected_kwh'] = df['generation_kwh'] * 1.2  # Fallback only if calculation fails

        return df
    except FileNotFoundError:
        st.warning(f"⚠️ Data not found for {plant_name}. Please check data files.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_data(ttl=3600)
def calculate_daylight_data_availability(plant_name: str):
    """Calculate completeness only for daylight/modeling hours."""
    hourly_df = data_loader.load_generation_data(plant_name=plant_name)
    if hourly_df.empty:
        return {'availability_pct': np.nan, 'available_hours': 0, 'total_hours': 0}

    from pvlib.location import Location

    location = Location(
        latitude=PLANT_CONFIG['latitude'],
        longitude=PLANT_CONFIG['longitude'],
        tz=PLANT_CONFIG['timezone']
    )
    times = pd.DatetimeIndex(hourly_df['generation_date'])
    solpos = location.get_solarposition(times)
    daylight_mask = solpos['elevation'].values > ML_CONFIG.get('min_sun_elevation', 5)
    daylight_df = hourly_df[daylight_mask]

    total_hours = len(daylight_df)
    available_hours = int(daylight_df['data_available'].sum()) if total_hours else 0
    availability_pct = available_hours / total_hours * 100 if total_hours else np.nan

    return {
        'availability_pct': availability_pct,
        'available_hours': available_hours,
        'total_hours': total_hours,
    }


def get_quality_summary(plant_name: str, model_cache_token: tuple | None = None):
    """Summarize daylight data availability and model reliability for visible dashboard warnings."""
    summary = {
        'data_availability_pct': None,
        'available_hours': None,
        'total_hours': None,
        'test_r2': None,
        'test_wape_pct': None,
        'issues': []
    }

    try:
        availability = calculate_daylight_data_availability(plant_name)
        summary['total_hours'] = availability['total_hours']
        summary['available_hours'] = availability['available_hours']
        summary['data_availability_pct'] = availability['availability_pct']

        if summary['data_availability_pct'] < QUALITY_WARNING_THRESHOLDS['min_data_availability_pct']:
            summary['issues'].append(
                f"Only {summary['data_availability_pct']:.1f}% of daylight/modeling hours have enough 5-minute records."
            )
    except Exception as e:
        summary['issues'].append(f"Could not calculate daylight data availability: {e}")

    try:
        comparison_df = model_loader.load_all_models_comparison(plant_name)
        if comparison_df is not None and not comparison_df.empty:
            best_model = comparison_df.iloc[0]
            summary['test_r2'] = float(best_model['Test R²'])
            if 'Test WAPE (%)' in comparison_df.columns:
                summary['test_wape_pct'] = float(best_model['Test WAPE (%)'])

            if summary['test_r2'] < QUALITY_WARNING_THRESHOLDS['min_test_r2']:
                summary['issues'].append(f"Model test R² is {summary['test_r2']:.3f}, worse than a simple mean baseline.")

            if (
                summary['test_wape_pct'] is not None
                and summary['test_wape_pct'] > QUALITY_WARNING_THRESHOLDS['max_test_wape_pct']
            ):
                summary['issues'].append(f"Model WAPE is high at {summary['test_wape_pct']:.1f}%.")
    except Exception as e:
        summary['issues'].append(f"Could not load model reliability metrics: {e}")

    return summary


def build_all_plants_model_summary(model_cache_token: tuple | None = None) -> pd.DataFrame:
    """Build one best-model summary row per plant."""
    rows = []
    for plant_name in DEMO_PLANTS:
        comparison_df = model_loader.load_all_models_comparison(plant_name)
        if comparison_df is None or comparison_df.empty:
            rows.append({
                'Plant': plant_name,
                'Best model': 'n/a',
                'R²': np.nan,
                'WAPE (%)': np.nan,
                'MAE (kWh)': np.nan,
                'RMSE (kWh)': np.nan,
                'Training Time (s)': np.nan,
                'Reliability': 'Unknown',
            })
            continue

        best_model = comparison_df.iloc[0]
        wape_col = 'Test WAPE (%)' if 'Test WAPE (%)' in comparison_df.columns else 'Test MAPE (%)'
        r2 = float(best_model['Test R²'])
        wape = float(best_model[wape_col]) if pd.notna(best_model[wape_col]) else np.nan
        has_warning = (
            r2 < QUALITY_WARNING_THRESHOLDS['min_test_r2']
            or (pd.notna(wape) and wape > QUALITY_WARNING_THRESHOLDS['max_test_wape_pct'])
        )
        rows.append({
            'Plant': plant_name,
            'Best model': best_model['Model'],
            'R²': r2,
            'WAPE (%)': wape,
            'MAE (kWh)': best_model['Test MAE (kWh)'],
            'RMSE (kWh)': best_model['Test RMSE (kWh)'],
            'Training Time (s)': best_model.get('Training Time (s)', np.nan),
            'Reliability': '⚠️ Review' if has_warning else '✅ Usable',
        })
    return pd.DataFrame(rows)


def render_quality_warning(plant_name: str):
    """Render a visible warning when data/model quality is not sufficient."""
    summary = get_quality_summary(plant_name, _model_summary_cache_token())

    if not summary['issues']:
        return

    st.error(f"⚠️ Data quality / unreliable model warning for {plant_name}")
    st.markdown(
        "This plant's forecasts and anomaly alerts should be reviewed carefully "
        "until data coverage/model quality is improved."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if summary['data_availability_pct'] is not None:
            st.metric("Usable daylight data", f"{summary['data_availability_pct']:.1f}%")
    with col2:
        if summary['test_r2'] is not None:
            st.metric("Model R²", f"{summary['test_r2']:.3f}")
    with col3:
        if summary['test_wape_pct'] is not None:
            st.metric("Model WAPE", f"{summary['test_wape_pct']:.1f}%")

    with st.expander("Why this warning appears", expanded=True):
        for issue in summary['issues']:
            st.write(f"- {issue}")
        st.caption(
            "Thresholds: usable daylight/modeling hours < 60%, model R² < 0, or WAPE > 70%. "
            "These limits are intentionally conservative so only clearly unreliable cases are flagged."
        )


# load_all_models_comparison() moved to src/model_loader.py


# load_model_and_features() moved to src/model_loader.py


# load_multiple_models() moved to src/model_loader.py



@st.cache_data
def load_full_featured_data(plant_name='Plant A'):
    """Load data with all features for feature analysis"""
    try:
        # Load raw data
        df = data_loader.load_generation_data(plant_name=plant_name)

        if 'generation_date' in df.columns:
            df = df.set_index('generation_date')

        # Load and merge weather data
        df_weather = data_loader.load_weather_data()
        df = df.reset_index()
        df = df.merge(df_weather, on='generation_date', how='left')
        df = data_loader.interpolate_weather(df)

        # Get plant capacity
        plant_capacity = DEMO_PLANTS[plant_name]['capacity_kwp']

        # Create all features with correct plant capacity
        df = feature_engineering.create_all_features(df, capacity_kwp=plant_capacity)
        df = df.set_index('generation_date')

        return df
    except Exception as e:
        st.warning(f"Could not load full featured data: {e}")
        return None

# calculate_daily_metrics() moved to src/metrics.py


# plot_calendar_heatmap() moved to src/visualization.py


# plot_time_of_day_analysis() moved to src/visualization.py


# plot_weather_correlation() moved to src/visualization.py


# plot_monthly_comparison() moved to src/visualization.py

# plot_weekly_production_pattern() moved to src/visualization.py

# plot_monthly_production_pattern() moved to src/visualization.py

# calculate_capacity_factor() moved to src/metrics.py



# fetch_weather_forecast() moved to src/forecasting.py


# create_forecast_features_optimized() moved to src/forecasting.py (renamed to create_forecast_features)


# plot_forecast_daily() moved to src/visualization.py (renamed to plot_forecast_daily_plotly)
# plot_forecast_hourly() moved to src/visualization.py (renamed to plot_forecast_hourly_plotly)

# plot_feature_importance_proxy() moved to src/visualization.py

# plot_top_features_importance() moved to src/visualization.py

# plot_model_comparison_metrics() moved to src/visualization.py


SOURCE_DATA_DIRS = [Path('data')]

SOURCE_SPECS = [
    {
        'label': '5-minute inverter generation',
        'filename': 'inverter_five_minutes_generation_logs.csv',
        'datetime_col': 'generation_date',
        'bucket_freq': '5min',
        'bucket_label': '5-minute records',
        'used_for_model': True,
        'base_use': '✅ Cleaned hourly model target source',
        'notes': 'Raw source used by build_clean_dataset.py. Models use clean/hourly_model_targets.csv, where incomplete hours are NaN/excluded rather than treated as zero.'
    },
    {
        'label': 'Hourly inverter generation file',
        'filename': 'inverter_hourly_generation_logs.csv',
        'datetime_col': 'generation_date',
        'bucket_freq': '1h',
        'bucket_label': 'hourly buckets',
        'used_for_model': False,
        'base_use': '🔎 Confirmation / inverter cross-check',
        'notes': 'Potential hourly fallback candidate, but not currently used for training. This remains a validation source rather than a gap-filling source.'
    },
    {
        'label': 'Daily inverter generation',
        'filename': 'inverter_daily_generation_logs.csv',
        'datetime_col': 'generation_date',
        'bucket_freq': '1D',
        'bucket_label': 'daily records',
        'used_for_model': False,
        'base_use': '🔎 Aggregate confirmation',
        'notes': 'Aggregate validation source only for the current hourly model. It cannot directly fill hourly gaps without assumptions or a separate daily model.'
    },
    {
        'label': 'Hourly meter readings',
        'filename': 'plants_meter_data_logs_hourly.csv',
        'datetime_col': 'date',
        'bucket_freq': '1h',
        'bucket_label': 'hourly meter buckets',
        'used_for_model': False,
        'base_use': '🔎 Meter cross-check',
        'notes': 'Potential aggregate/hourly fallback candidate after deriving production from cumulative meter-reading differences. Not currently used for training.'
    },
    {
        'label': 'Billing meter readings',
        'filename': 'plants_billing_meter_logs.csv',
        'datetime_col': 'date',
        'bucket_freq': '1D',
        'bucket_label': 'daily meter buckets',
        'used_for_model': False,
        'base_use': '🔎 Aggregate meter check',
        'notes': 'Best aggregate fallback candidate when inverter 5-minute data is incomplete. Better suited for validation or a separate daily model than direct hourly gap-filling.'
    },
]


def _file_mtime(path: str) -> float:
    """Return file mtime so Streamlit cache invalidates when local CSVs change."""
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


def _load_weather_for_outlook() -> tuple[pd.DataFrame, list[str]]:
    """Load local weather columns in the same shape as OpenWeather forecast rows."""
    weather = pd.read_csv(DATA_PATHS['weather'])
    weather['timestamp'] = pd.to_datetime(
        weather['dt_iso'].str.replace(' UTC', '', regex=False), utc=True, errors='coerce'
    ).dt.tz_convert(PLANT_CONFIG['timezone'])
    weather = weather.set_index('timestamp').sort_index()

    required_cols = ['temp', 'pressure', 'humidity', 'clouds_all', 'wind_speed']
    optional_defaults = {
        'feels_like': weather['temp'] if 'temp' in weather.columns else 25,
        'wind_deg': 0,
        'visibility': 10000,
        'rain_1h': 0,
    }
    for col, default in optional_defaults.items():
        if col not in weather.columns:
            weather[col] = default
        else:
            weather[col] = weather[col].fillna(default)

    available_cols = [col for col in required_cols + list(optional_defaults.keys()) if col in weather.columns]
    return weather[available_cols], required_cols


@st.cache_data(ttl=3600)
def build_historical_weather_replay(weather_mtime: float) -> pd.DataFrame:
    """Build a 5-day model replay from the latest weather rows in the local CSV."""
    weather, required_cols = _load_weather_for_outlook()
    weather = weather[weather.index <= DATA_SNAPSHOT_TODAY]
    return weather.resample('3h').mean(numeric_only=True).dropna(subset=required_cols).tail(40)


@st.cache_data(ttl=3600)
def build_historical_weather_forecast_scenario(weather_mtime: float, start_iso: str) -> pd.DataFrame:
    """Use weather after the production cutoff as a historical 5-day forecast scenario."""
    weather, required_cols = _load_weather_for_outlook()
    start = pd.Timestamp(start_iso).tz_convert(PLANT_CONFIG['timezone'])
    scenario = weather[weather.index >= start]
    return scenario.resample('3h').mean(numeric_only=True).dropna(subset=required_cols).head(40)


def get_outlook_weather_input(openweather_api_key: str, historical_df: pd.DataFrame | None = None):
    """Return live forecast weather, historical forecast scenario, or replay fallback."""
    if openweather_api_key:
        weather_forecast = forecasting.fetch_weather_forecast(
            openweather_api_key,
            PLANT_CONFIG['latitude'],
            PLANT_CONFIG['longitude']
        )
        if weather_forecast is not None and not weather_forecast.empty:
            return weather_forecast, 'OpenWeather forecast', False

    if historical_df is not None and not historical_df.empty:
        last_production_time = pd.Timestamp(historical_df.index.max()).tz_convert(PLANT_CONFIG['timezone'])
        forecast_start = last_production_time.floor('D') + pd.Timedelta(days=1)
        scenario = build_historical_weather_forecast_scenario(
            _file_mtime(DATA_PATHS['weather']),
            forecast_start.isoformat()
        )
        if len(scenario) >= 8:
            label = 'Historical weather forecast scenario'
            if openweather_api_key:
                label += ' (live weather unavailable)'
            return scenario, label, False

    replay = build_historical_weather_replay(_file_mtime(DATA_PATHS['weather']))
    label = 'Latest historical weather replay'
    if openweather_api_key:
        label += ' (live weather unavailable)'
    return replay, label, True


def find_source_file(filename: str):
    """Find an optional source CSV in the project data directory."""
    for directory in SOURCE_DATA_DIRS:
        path = directory / filename
        if path.exists():
            return path
    return None


@st.cache_data(ttl=3600)
def summarize_source_quality(plant_name: str):
    """Compare data completeness across available source files for one plant."""
    plant_id = DEMO_PLANTS[plant_name]['plant_id']
    rows = []

    for spec in SOURCE_SPECS:
        path = find_source_file(spec['filename'])
        if path is None:
            rows.append({
                'Source': spec['label'],
                'File': spec['filename'],
                'Used for model': '✅' if spec['used_for_model'] else '',
                'Use in dashboard': spec['base_use'],
                'Granularity': spec['bucket_label'],
                'Status': 'Missing file',
                'Data period': None,
                'Expected records': None,
                'Collected records': None,
                'Completeness (%)': None,
                'Missing records': None,
                'Days since last data': None,
                'Freshness': 'Missing',
                'Notes': spec['notes'],
            })
            continue

        try:
            df_source = pd.read_csv(path)
            if 'plant_id' not in df_source.columns or spec['datetime_col'] not in df_source.columns:
                raise ValueError('required columns not found')

            df_source = df_source[df_source['plant_id'] == plant_id].copy()
            df_source['timestamp'] = pd.to_datetime(df_source[spec['datetime_col']], utc=True, errors='coerce')
            df_source = df_source.dropna(subset=['timestamp'])

            if df_source.empty:
                raise ValueError('no rows for selected plant')

            freq = spec['bucket_freq']
            if freq == '5min':
                buckets = df_source['timestamp'].dt.floor('5min')
            elif freq == '1h':
                buckets = df_source['timestamp'].dt.floor('h')
            else:
                buckets = df_source['timestamp'].dt.floor('D')

            collected = int(buckets.nunique())
            start = buckets.min()
            end = buckets.max()
            expected = len(pd.date_range(start=start, end=end, freq=freq))
            missing = max(expected - collected, 0)
            completeness = collected / expected * 100 if expected else 0
            days_since_last = (DATA_SNAPSHOT_TODAY.tz_convert('UTC') - end).days

            if days_since_last <= 7:
                freshness = '🟢 CURRENT'
            elif days_since_last <= 30:
                freshness = '🟡 RECENT'
            elif days_since_last <= 90:
                freshness = '🟠 STALE'
            else:
                freshness = '🔴 VERY STALE'

            minute_values = sorted(df_source['timestamp'].dt.minute.dropna().unique().tolist())
            extra_note = ''
            if spec['filename'] == 'inverter_hourly_generation_logs.csv' and len(minute_values) > 2:
                extra_note = ' Timestamp minutes vary; this file does not look like one clean hourly row per hour.'

            rows.append({
                'Source': spec['label'],
                'File': str(path),
                'Used for model': '✅' if spec['used_for_model'] else '',
                'Use in dashboard': spec['base_use'],
                'Granularity': spec['bucket_label'],
                'Status': 'Available',
                'Data period': f"{start.date()} → {end.date()}",
                'Expected records': expected,
                'Collected records': collected,
                'Completeness (%)': completeness,
                'Missing records': missing,
                'Days since last data': days_since_last,
                'Freshness': freshness,
                'Notes': spec['notes'] + extra_note,
            })
        except Exception as e:
            rows.append({
                'Source': spec['label'],
                'File': str(path),
                'Used for model': '✅' if spec['used_for_model'] else '',
                'Use in dashboard': spec['base_use'],
                'Granularity': spec['bucket_label'],
                'Status': f'Unavailable: {e}',
                'Data period': None,
                'Expected records': None,
                'Collected records': None,
                'Completeness (%)': None,
                'Missing records': None,
                'Days since last data': None,
                'Freshness': 'Unavailable',
                'Notes': spec['notes'],
            })

    source_df = pd.DataFrame(rows)
    model_rows = source_df[source_df['Used for model'] == '✅']
    if not model_rows.empty:
        model_completeness = model_rows.iloc[0]['Completeness (%)']
        model_is_partial = pd.notna(model_completeness) and model_completeness < 95
        model_is_weak = pd.notna(model_completeness) and model_completeness < 70

        if model_is_partial:
            source_df.loc[source_df['Used for model'] == '✅', 'Use in dashboard'] = '✅ Primary model source (partial)'
            source_df.loc[source_df['Used for model'] == '✅', 'Notes'] += (
                ' Some expected intervals are missing, so other CSVs are useful for validation or fallback planning.'
            )

        if model_is_weak:
            meter_candidates = source_df['Source'].isin(['Hourly meter readings', 'Billing meter readings'])
            source_df.loc[meter_candidates & (source_df['Status'] == 'Available'), 'Use in dashboard'] = (
                '⚠️ Fallback candidate + confirmation'
            )
        elif model_is_partial:
            meter_candidates = source_df['Source'].isin(['Hourly meter readings', 'Billing meter readings'])
            source_df.loc[meter_candidates & (source_df['Status'] == 'Available'), 'Use in dashboard'] = (
                '🔎 Confirmation / fallback candidate'
            )

    return source_df


@st.cache_data(ttl=3600)
def compare_daily_generation_sources(plant_name: str):
    """Compare daily production totals across inverter and meter source files."""
    plant_id = DEMO_PLANTS[plant_name]['plant_id']

    source_defs = [
        {
            'key': 'five_min_kwh',
            'label': '5-minute inverter',
            'filename': 'inverter_five_minutes_generation_logs.csv',
            'kind': 'generation_log',
            'amount_divisor': 1000,
            'date_shift_days': 0,
            'baseline': True,
        },
        {
            'key': 'hourly_kwh',
            'label': 'Hourly inverter',
            'filename': 'inverter_hourly_generation_logs.csv',
            'kind': 'generation_log',
            'amount_divisor': 1000,
            'date_shift_days': 0,
        },
        {
            'key': 'daily_inverter_kwh',
            'label': 'Daily inverter',
            'filename': 'inverter_daily_generation_logs.csv',
            'kind': 'generation_log',
            'amount_divisor': 1,
            # Daily inverter rows are timestamped around 00:00 local after the production day.
            'date_shift_days': -1,
        },
        {
            'key': 'hourly_meter_kwh',
            'label': 'Hourly meter',
            'filename': 'plants_meter_data_logs_hourly.csv',
            'kind': 'meter_reading',
            'date_col': 'date',
        },
        {
            'key': 'billing_meter_kwh',
            'label': 'Billing meter',
            'filename': 'plants_billing_meter_logs.csv',
            'kind': 'meter_reading',
            'date_col': 'date',
        },
    ]

    def shifted_local_date(timestamp_series: pd.Series, shift_days: int = 0) -> pd.Series:
        dates = timestamp_series.dt.tz_convert('Asia/Dhaka').dt.date
        if shift_days:
            dates = (pd.to_datetime(dates) + pd.Timedelta(days=shift_days)).dt.date
        return dates

    def daily_generation_log(path: Path, spec: dict) -> pd.DataFrame:
        df_source = pd.read_csv(path, usecols=['plant_id', 'generation_date', 'generation_amount'])
        df_source = df_source[df_source['plant_id'] == plant_id].copy()
        df_source['timestamp'] = pd.to_datetime(df_source['generation_date'], utc=True, errors='coerce')
        df_source['date'] = shifted_local_date(df_source['timestamp'], spec.get('date_shift_days', 0))
        df_source['generation_amount'] = pd.to_numeric(
            df_source['generation_amount'].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
        df_source = df_source.dropna(subset=['timestamp', 'generation_amount'])

        daily_total = (
            df_source.groupby('date')['generation_amount'].sum(min_count=1) / spec['amount_divisor']
        ).rename(spec['key'])
        daily_records = df_source.groupby('date').size().rename(f"{spec['key']}_records")
        return pd.concat([daily_total, daily_records], axis=1)

    def daily_meter_reading(path: Path, spec: dict) -> pd.DataFrame:
        df_source = pd.read_csv(path)
        required_cols = {'plant_id', 'meter_id', 'meter_reading', spec.get('date_col', 'date')}
        if not required_cols.issubset(df_source.columns):
            raise ValueError(f"missing columns: {sorted(required_cols - set(df_source.columns))}")

        df_source = df_source[df_source['plant_id'] == plant_id].copy()
        df_source['timestamp'] = pd.to_datetime(df_source[spec.get('date_col', 'date')], utc=True, errors='coerce')
        df_source['date'] = shifted_local_date(df_source['timestamp'], spec.get('date_shift_days', 0))
        df_source['meter_reading'] = pd.to_numeric(df_source['meter_reading'], errors='coerce')

        parts = []
        for _, meter_df in df_source.dropna(subset=['timestamp', 'meter_reading']).groupby('meter_id'):
            meter_df = meter_df.sort_values('timestamp').copy()
            meter_df[spec['key']] = meter_df['meter_reading'].diff() / 1000
            meter_df.loc[meter_df[spec['key']] < 0, spec['key']] = np.nan
            parts.append(meter_df[['date', spec['key']]])

        if not parts:
            return pd.DataFrame(columns=[spec['key'], f"{spec['key']}_records"])

        meter_daily_rows = pd.concat(parts)
        daily_total = meter_daily_rows.groupby('date')[spec['key']].sum(min_count=1).rename(spec['key'])
        daily_records = meter_daily_rows.groupby('date')[spec['key']].count().rename(f"{spec['key']}_records")
        return pd.concat([daily_total, daily_records], axis=1)

    try:
        daily_parts = []
        available_sources = []
        missing_sources = []

        for spec in source_defs:
            path = find_source_file(spec['filename'])
            if path is None:
                missing_sources.append({'label': spec['label'], 'filename': spec['filename']})
                continue

            if spec['kind'] == 'generation_log':
                daily_df = daily_generation_log(path, spec)
            else:
                daily_df = daily_meter_reading(path, spec)

            if daily_df.empty or spec['key'] not in daily_df.columns:
                missing_sources.append({'label': spec['label'], 'filename': str(path)})
                continue

            daily_parts.append(daily_df)
            available_sources.append({**spec, 'path': str(path)})

        if not daily_parts:
            return None

        daily_compare = pd.concat(daily_parts, axis=1).sort_index()
        if 'five_min_kwh' not in daily_compare.columns:
            return {'error': '5-minute inverter source is required as the comparison baseline.'}

        source_summaries = []
        match_tolerance_kwh = 0.01
        baseline = daily_compare['five_min_kwh']

        for source in available_sources:
            key = source['key']
            if key == 'five_min_kwh':
                continue

            comparison = daily_compare[['five_min_kwh', key]].dropna()
            if comparison.empty:
                continue

            difference = comparison[key] - comparison['five_min_kwh']
            five_total = comparison['five_min_kwh'].sum()
            source_total = comparison[key].sum()

            source_summaries.append({
                'Source': source['label'],
                'Overlap days': len(comparison),
                'Correlation': comparison['five_min_kwh'].corr(comparison[key]) if len(comparison) > 1 else np.nan,
                'Source / 5-min total': source_total / five_total if five_total else np.nan,
                'Daily MAE (kWh)': difference.abs().mean(),
                'Max abs diff (kWh)': difference.abs().max(),
                'Matching days (%)': (difference.abs() <= match_tolerance_kwh).mean() * 100,
                'File': source['path'],
            })

        if 'hourly_kwh' in daily_compare.columns:
            daily_compare['hourly_difference_kwh'] = daily_compare['hourly_kwh'] - baseline
            daily_compare['hourly_abs_difference_kwh'] = daily_compare['hourly_difference_kwh'].abs()
            daily_compare['hourly_pct_difference'] = np.where(
                baseline.abs() > 0,
                daily_compare['hourly_difference_kwh'] / baseline * 100,
                np.nan
            )

        return {
            'daily': daily_compare.reset_index(),
            'source_summaries': pd.DataFrame(source_summaries),
            'available_sources': available_sources,
            'missing_sources': missing_sources,
        }
    except Exception as e:
        return {'error': str(e)}


def _score_closeness_to_one(value: float) -> float:
    """Score a ratio by how close it is to 1.0."""
    if pd.isna(value):
        return np.nan
    return max(0, 100 - abs(value - 1) * 100)


def _score_correlation(value: float) -> float:
    """Convert correlation to a 0-100 score, ignoring negative correlation."""
    if pd.isna(value):
        return np.nan
    return max(0, min(100, value * 100))


def _get_source_summary_row(source_summaries: pd.DataFrame, source_name: str):
    """Return one source summary row by display name."""
    if source_summaries is None or source_summaries.empty:
        return None
    rows = source_summaries[source_summaries['Source'] == source_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _weighted_score(parts):
    """Return weighted average while ignoring missing component scores."""
    valid = [(score, weight) for score, weight in parts if not pd.isna(score)]
    if not valid:
        return np.nan
    total_weight = sum(weight for _, weight in valid)
    return sum(score * weight for score, weight in valid) / total_weight


def _style_source_ratio_cell(value):
    """Color source-agreement ratio cells by distance from 1.0."""
    if pd.isna(value):
        return 'background-color: #f1f3f5; color: #6c757d;'

    difference = abs(value - 1)
    if difference <= 0.05:
        return 'background-color: #d4edda; color: #155724; font-weight: 600;'
    if difference <= 0.20:
        return 'background-color: #fff3cd; color: #856404; font-weight: 600;'
    if difference <= 0.50:
        return 'background-color: #ffe0b2; color: #8a4b00; font-weight: 600;'
    return 'background-color: #f8d7da; color: #721c24; font-weight: 600;'


def _style_percentage_cell(value):
    """Color percentage completeness/match cells."""
    if pd.isna(value):
        return 'background-color: #f1f3f5; color: #6c757d;'
    if value >= 90:
        return 'background-color: #d4edda; color: #155724; font-weight: 600;'
    if value >= 70:
        return 'background-color: #fff3cd; color: #856404; font-weight: 600;'
    if value >= 50:
        return 'background-color: #ffe0b2; color: #8a4b00; font-weight: 600;'
    return 'background-color: #f8d7da; color: #721c24; font-weight: 600;'


def _format_hover_value(value, suffix='', decimals=1):
    """Format values for the plant hover recap."""
    if pd.isna(value):
        return 'n/a'
    return f"{value:.{decimals}f}{suffix}"


def _render_all_plants_quality_overview(overview_df: pd.DataFrame):
    """Render overview with CSV recap details on plant-name hover."""
    rows_html = []
    for _, row in overview_df.iterrows():
        tooltip_lines = [
            '<strong>CSV source recap</strong>',
            f"5-min usable coverage: {_format_hover_value(row['Usable daylight data (%)'], '%')}",
            f"Hourly inverter exact-match days: {_format_hover_value(row['Hourly inverter match (%)'], '%')}",
            f"Daily inverter / 5-min total: {_format_hover_value(row['Daily inverter ratio'], decimals=2)}",
            f"Hourly meter / 5-min total: {_format_hover_value(row['Hourly meter ratio'], decimals=2)}",
            f"Billing meter / 5-min total: {_format_hover_value(row['Billing meter ratio'], decimals=2)}",
            '<br><em>Example: 1.30 means that CSV reports 30% more energy than the 5-minute inverter source.</em>',
        ]
        tooltip_html = '<br>'.join(tooltip_lines)
        plant_html = (
            '<span class="plant-hover">'
            f"{html.escape(row['Plant'])} <span class=\"hover-hint\">ⓘ</span>"
            f'<span class="plant-tooltip">{tooltip_html}</span>'
            '</span>'
        )

        rows_html.append(
            '<tr>'
            f'<td>{plant_html}</td>'
            f'<td style="{_style_percentage_cell(row["Data quality score"])}">{_format_hover_value(row["Data quality score"])}</td>'
            f'<td>{html.escape(str(row["Data quality band"]))}</td>'
            f'<td style="{_style_percentage_cell(row["Usable daylight data (%)"])}">{_format_hover_value(row["Usable daylight data (%)"])}</td>'
            f'<td style="{_style_percentage_cell(row["Source agreement score"])}">{_format_hover_value(row["Source agreement score"])}</td>'
            f'<td>{html.escape(str(row["Warnings"]))}</td>'
            '</tr>'
        )

    table_html = f'''
    <style>
        .quality-overview-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
        .quality-overview-table th,
        .quality-overview-table td {{
            border: 1px solid #e6e9ef;
            padding: 0.55rem 0.65rem;
            text-align: left;
            vertical-align: top;
        }}
        .quality-overview-table th {{
            background: #f8f9fa;
            font-weight: 700;
        }}
        .plant-hover {{
            position: relative;
            display: inline-block;
            cursor: help;
            font-weight: 600;
        }}
        .hover-hint {{
            color: #6c757d;
            font-size: 0.8rem;
        }}
        .plant-tooltip {{
            visibility: hidden;
            opacity: 0;
            transition: opacity 0.15s ease;
            position: absolute;
            left: 0;
            top: 1.6rem;
            z-index: 999;
            min-width: 330px;
            max-width: 420px;
            background: #262730;
            color: white;
            border-radius: 0.4rem;
            padding: 0.75rem;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
            font-weight: 400;
            line-height: 1.35;
        }}
        .plant-hover:hover .plant-tooltip {{
            visibility: visible;
            opacity: 1;
        }}
    </style>
    <table class="quality-overview-table">
        <thead>
            <tr>
                <th>Plant</th>
                <th>Data quality score</th>
                <th>Data quality band</th>
                <th>Usable daylight data (%)</th>
                <th>Source agreement score</th>
                <th>Warnings</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
    '''
    st.markdown(table_html, unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def build_all_plants_quality_overview(cache_token: tuple | None = None) -> pd.DataFrame:
    """Build one quality overview row per configured plant."""
    rows = []

    for plant_name in DEMO_PLANTS:
        try:
            availability_summary = calculate_daylight_data_availability(plant_name)
            availability = availability_summary.get('availability_pct', np.nan)
        except Exception:
            availability = np.nan

        source_agreement = compare_daily_generation_sources(plant_name)
        source_summaries = (
            source_agreement.get('source_summaries')
            if source_agreement and 'error' not in source_agreement
            else pd.DataFrame()
        )

        hourly_row = _get_source_summary_row(source_summaries, 'Hourly inverter')
        daily_row = _get_source_summary_row(source_summaries, 'Daily inverter')
        hourly_meter_row = _get_source_summary_row(source_summaries, 'Hourly meter')
        billing_row = _get_source_summary_row(source_summaries, 'Billing meter')

        data_score = availability if pd.notna(availability) else np.nan

        hourly_score = hourly_row['Matching days (%)'] if hourly_row is not None else np.nan
        daily_score = np.nan
        if daily_row is not None:
            daily_score = _weighted_score([
                (_score_correlation(daily_row['Correlation']), 0.5),
                (_score_closeness_to_one(daily_row['Source / 5-min total']), 0.5),
            ])
        billing_score = np.nan
        if billing_row is not None:
            billing_score = _weighted_score([
                (_score_correlation(billing_row['Correlation']), 0.5),
                (_score_closeness_to_one(billing_row['Source / 5-min total']), 0.5),
            ])
        source_score = _weighted_score([(hourly_score, 0.20), (daily_score, 0.35), (billing_score, 0.45)])

        data_quality_score = _weighted_score([(data_score, 0.55), (source_score, 0.45)])

        if pd.isna(data_quality_score):
            label = 'Unknown'
        elif data_quality_score >= 80:
            label = '🟢 Good'
        elif data_quality_score >= 65:
            label = '🟡 Usable'
        elif data_quality_score >= 50:
            label = '🟠 Caution'
        else:
            label = '🔴 Poor'

        rows.append({
            'Plant': plant_name,
            'Data quality score': data_quality_score,
            'Data quality band': label,
            'Usable daylight data (%)': availability,
            'Source agreement score': source_score,
            'Hourly inverter match (%)': hourly_row['Matching days (%)'] if hourly_row is not None else np.nan,
            'Daily inverter ratio': daily_row['Source / 5-min total'] if daily_row is not None else np.nan,
            'Daily inverter corr': daily_row['Correlation'] if daily_row is not None else np.nan,
            'Hourly meter ratio': hourly_meter_row['Source / 5-min total'] if hourly_meter_row is not None else np.nan,
            'Hourly meter corr': hourly_meter_row['Correlation'] if hourly_meter_row is not None else np.nan,
            'Billing meter ratio': billing_row['Source / 5-min total'] if billing_row is not None else np.nan,
            'Billing meter corr': billing_row['Correlation'] if billing_row is not None else np.nan,
            'Warnings': '; '.join([
                warning for warning in [
                    'Low daylight coverage' if pd.notna(availability) and availability < 60 else None,
                    'Weak source agreement' if source_score is not None and not pd.isna(source_score) and source_score < 60 else None,
                    'Meter total differs from inverter' if billing_row is not None and (
                        billing_row['Source / 5-min total'] < 0.8 or billing_row['Source / 5-min total'] > 1.2
                    ) else None,
                ] if warning
            ]) or 'None',
        })

    return pd.DataFrame(rows).sort_values('Data quality score', ascending=False)


@st.cache_data(ttl=3600)
def build_hourly_training_source_diagnostics(plant_name: str) -> pd.DataFrame:
    """Build a candidate hourly target from 5-minute inverter data plus hourly inverter fallback."""
    plant_id = DEMO_PLANTS[plant_name]['plant_id']
    five_path = find_source_file('inverter_five_minutes_generation_logs.csv')
    hourly_path = find_source_file('inverter_hourly_generation_logs.csv')
    if five_path is None:
        return pd.DataFrame()

    five = pd.read_csv(five_path)
    five = five[five['plant_id'] == plant_id].copy()
    five['timestamp'] = pd.to_datetime(five['generation_date'], utc=True, errors='coerce').dt.tz_convert(PLANT_CONFIG['timezone'])
    five['generation_amount'] = pd.to_numeric(
        five['generation_amount'].astype(str).str.replace(',', '', regex=False),
        errors='coerce'
    )
    five = five.dropna(subset=['timestamp', 'generation_amount']).drop_duplicates(subset='timestamp', keep='last')
    five['generation_kwh'] = five['generation_amount'] / 1000

    five_hourly = (
        five.set_index('timestamp')
        .resample('1h')['generation_kwh']
        .agg([
            ('generation_kwh_5min_raw', lambda x: x.sum(min_count=1)),
            ('records_collected_5min', 'count'),
        ])
        .reset_index()
        .rename(columns={'timestamp': 'generation_date'})
    )
    five_hourly['expected_records_5min'] = 12
    five_hourly['data_completeness_pct'] = five_hourly['records_collected_5min'] / 12 * 100
    five_hourly['five_min_available'] = (
        five_hourly['data_completeness_pct'] >= ML_CONFIG.get('min_hourly_completeness_pct', 80)
    )
    five_hourly['generation_kwh_5min'] = five_hourly['generation_kwh_5min_raw'].where(
        five_hourly['five_min_available']
    )

    if hourly_path is not None:
        hourly = pd.read_csv(hourly_path)
        hourly = hourly[hourly['plant_id'] == plant_id].copy()
        hourly['timestamp'] = pd.to_datetime(hourly['generation_date'], utc=True, errors='coerce').dt.tz_convert(PLANT_CONFIG['timezone'])
        hourly['generation_amount'] = pd.to_numeric(
            hourly['generation_amount'].astype(str).str.replace(',', '', regex=False),
            errors='coerce'
        )
        hourly = hourly.dropna(subset=['timestamp', 'generation_amount'])
        hourly['hour'] = hourly['timestamp'].dt.floor('h')
        hourly['generation_kwh_hourly'] = hourly['generation_amount'] / 1000
        hourly_hourly = (
            hourly.groupby('hour')
            .agg(
                generation_kwh_hourly=('generation_kwh_hourly', 'sum'),
                hourly_records=('generation_kwh_hourly', 'count'),
            )
            .reset_index()
            .rename(columns={'hour': 'generation_date'})
        )
    else:
        hourly_hourly = pd.DataFrame(columns=['generation_date', 'generation_kwh_hourly', 'hourly_records'])

    diagnostics = five_hourly.merge(hourly_hourly, on='generation_date', how='outer').sort_values('generation_date')
    diagnostics['hourly_available'] = diagnostics['generation_kwh_hourly'].notna()
    diagnostics['generation_kwh_final'] = diagnostics['generation_kwh_5min']
    fallback_mask = diagnostics['generation_kwh_final'].isna() & diagnostics['hourly_available']
    diagnostics.loc[fallback_mask, 'generation_kwh_final'] = diagnostics.loc[fallback_mask, 'generation_kwh_hourly']

    diagnostics['target_source'] = 'missing'
    diagnostics.loc[diagnostics['generation_kwh_5min'].notna(), 'target_source'] = '5min_inverter'
    diagnostics.loc[fallback_mask, 'target_source'] = 'hourly_inverter_fallback'
    diagnostics['target_is_fallback'] = diagnostics['target_source'] == 'hourly_inverter_fallback'

    diagnostics['delta_kwh_hourly_minus_5min'] = (
        diagnostics['generation_kwh_hourly'] - diagnostics['generation_kwh_5min_raw']
    )
    diagnostics['delta_pct_hourly_vs_5min'] = np.where(
        diagnostics['generation_kwh_5min_raw'].abs() > 0.01,
        diagnostics['delta_kwh_hourly_minus_5min'] / diagnostics['generation_kwh_5min_raw'] * 100,
        np.where(diagnostics['delta_kwh_hourly_minus_5min'].abs() <= 0.01, 0, np.nan)
    )
    diagnostics['abs_delta_pct_hourly_vs_5min'] = diagnostics['delta_pct_hourly_vs_5min'].abs()

    both_present = diagnostics['five_min_available'] & diagnostics['hourly_available']
    diagnostics['target_quality_flag'] = 'missing'
    diagnostics.loc[diagnostics['five_min_available'], 'target_quality_flag'] = '5min_primary'
    diagnostics.loc[fallback_mask, 'target_quality_flag'] = 'hourly_fallback'
    diagnostics.loc[both_present & (diagnostics['abs_delta_pct_hourly_vs_5min'] <= 5), 'target_quality_flag'] = 'confirmed_5min'
    diagnostics.loc[
        both_present
        & (diagnostics['abs_delta_pct_hourly_vs_5min'] > 5)
        & (diagnostics['abs_delta_pct_hourly_vs_5min'] <= 20),
        'target_quality_flag'
    ] = 'moderate_difference'
    diagnostics.loc[both_present & (diagnostics['abs_delta_pct_hourly_vs_5min'] > 20), 'target_quality_flag'] = 'large_difference'

    diagnostics['target_quality_score'] = diagnostics['target_quality_flag'].map({
        'missing': 0,
        'large_difference': 1,
        'hourly_fallback': 2,
        'moderate_difference': 2,
        '5min_primary': 3,
        'confirmed_5min': 3,
    })

    return diagnostics


def create_training_source_heatmap(diagnostics: pd.DataFrame, days: int = 90):
    """Create hour-by-day heatmap for candidate training target confidence."""
    if diagnostics.empty:
        return None

    plot_df = diagnostics.copy()
    latest_date = plot_df['generation_date'].max()
    if pd.notna(latest_date):
        plot_df = plot_df[plot_df['generation_date'] >= latest_date - pd.Timedelta(days=days)]

    plot_df['Date'] = plot_df['generation_date'].dt.strftime('%Y-%m-%d')
    plot_df['Hour'] = plot_df['generation_date'].dt.hour
    plot_df['hover'] = plot_df.apply(
        lambda row: (
            f"<b>{row['generation_date'].strftime('%Y-%m-%d %H:%M')}</b><br>"
            f"Status: {row['target_quality_flag'].replace('_', ' ')}<br>"
            f"Training target: {row['target_source']}<br>"
            f"Final kWh: {_format_hover_value(row['generation_kwh_final'], ' kWh', 2)}<br>"
            f"5-min kWh: {_format_hover_value(row['generation_kwh_5min_raw'], ' kWh', 2)} "
            f"({_format_hover_value(row['data_completeness_pct'], '%')})<br>"
            f"Hourly inverter kWh: {_format_hover_value(row['generation_kwh_hourly'], ' kWh', 2)}<br>"
            f"Delta hourly vs 5-min: {_format_hover_value(row['delta_pct_hourly_vs_5min'], '%')}"
        ),
        axis=1
    )

    z = plot_df.pivot_table(index='Hour', columns='Date', values='target_quality_score', aggfunc='first').reindex(range(24))
    hover = plot_df.pivot_table(index='Hour', columns='Date', values='hover', aggfunc='first').reindex(range(24))

    fig = go.Figure(data=go.Heatmap(
        z=z.values,
        x=z.columns,
        y=z.index,
        text=hover.values,
        hovertemplate='%{text}<extra></extra>',
        colorscale=[
            [0.00, '#d73027'],
            [0.24, '#d73027'],
            [0.25, '#fdae61'],
            [0.49, '#fdae61'],
            [0.50, '#fee08b'],
            [0.74, '#fee08b'],
            [0.75, '#1a9850'],
            [1.00, '#1a9850'],
        ],
        zmin=0,
        zmax=3,
        colorbar=dict(
            title='Quality',
            tickmode='array',
            tickvals=[0, 1, 2, 3],
            ticktext=['Missing', 'Disagree', 'Fallback/usable', '5-min target']
        )
    ))
    fig.update_layout(
        title=f'Hourly training target confidence — last {days} days',
        xaxis_title='Date',
        yaxis_title='Hour of day',
        height=430,
        yaxis=dict(autorange='reversed')
    )
    return fig


def build_daily_control_comparison(plant_name: str, diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate hourly training target totals against daily/meter control sources."""
    if diagnostics.empty:
        return pd.DataFrame()

    final_daily = diagnostics.copy()
    final_daily['date'] = final_daily['generation_date'].dt.date
    final_daily = (
        final_daily.groupby('date')['generation_kwh_final']
        .sum(min_count=1)
        .rename('candidate_hourly_target_kwh')
    )

    source_agreement = compare_daily_generation_sources(plant_name)
    if not source_agreement or 'error' in source_agreement:
        return final_daily.reset_index()

    daily = source_agreement['daily'].copy()
    daily['date'] = pd.to_datetime(daily['date']).dt.date
    daily = daily.set_index('date')
    return pd.concat([final_daily, daily], axis=1).reset_index()


def _control_ratio_interpretation(ratio: float) -> str:
    """Explain candidate/control ratio in plain language."""
    if pd.isna(ratio):
        return 'n/a'
    difference_pct = (ratio - 1) * 100
    if abs(difference_pct) <= 5:
        return 'Close agreement'
    if difference_pct > 0:
        return f'Candidate is {difference_pct:.0f}% higher than control'
    return f'Candidate is {abs(difference_pct):.0f}% lower than control'


def summarize_daily_control_deltas(daily_compare: pd.DataFrame) -> pd.DataFrame:
    """Summarize deltas between candidate hourly target and control sources."""
    if daily_compare.empty or 'candidate_hourly_target_kwh' not in daily_compare.columns:
        return pd.DataFrame()

    control_cols = {
        'daily_inverter_kwh': 'Daily inverter control',
        'hourly_meter_kwh': 'Hourly meter control',
        'billing_meter_kwh': 'Billing meter control',
    }
    rows = []
    for col, label in control_cols.items():
        if col not in daily_compare.columns:
            continue
        comparison = daily_compare[['candidate_hourly_target_kwh', col]].dropna()
        if comparison.empty:
            continue
        delta = comparison['candidate_hourly_target_kwh'] - comparison[col]
        control_total = comparison[col].sum()
        ratio = comparison['candidate_hourly_target_kwh'].sum() / control_total if control_total else np.nan
        rows.append({
            'Control source': label,
            'Overlap days': len(comparison),
            'Candidate / control total': ratio,
            'Interpretation': _control_ratio_interpretation(ratio),
            'Mean delta (kWh)': delta.mean(),
            'Min delta (kWh)': delta.min(),
            'Max delta (kWh)': delta.max(),
            'MAE (kWh)': delta.abs().mean(),
        })
    return pd.DataFrame(rows)


def create_daily_control_line_chart(daily_compare: pd.DataFrame, days: int = 180):
    """Line chart comparing candidate hourly target daily totals with control sources."""
    if daily_compare.empty or 'candidate_hourly_target_kwh' not in daily_compare.columns:
        return None

    plot_df = daily_compare.copy()
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    latest_date = plot_df['date'].max()
    if pd.notna(latest_date):
        plot_df = plot_df[plot_df['date'] >= latest_date - pd.Timedelta(days=days)]

    fig = go.Figure()
    source_cols = [
        ('candidate_hourly_target_kwh', 'Candidate hourly target', '#1f77b4'),
        ('daily_inverter_kwh', 'Daily inverter control', '#2ca02c'),
        ('hourly_meter_kwh', 'Hourly meter control', '#ff7f0e'),
        ('billing_meter_kwh', 'Billing meter control', '#d62728'),
    ]
    for col, label, color in source_cols:
        if col in plot_df.columns:
            fig.add_trace(go.Scatter(
                x=plot_df['date'],
                y=plot_df[col],
                mode='lines',
                name=label,
                line=dict(color=color)
            ))
    fig.update_layout(
        title=f'Daily controls vs candidate hourly target — last {days} days',
        xaxis_title='Date',
        yaxis_title='Daily production (kWh)',
        height=420,
        hovermode='x unified'
    )
    return fig


def create_control_ratio_bar_chart(control_summary: pd.DataFrame):
    """Show candidate/control total ratios as an easier-to-read agreement chart."""
    if control_summary.empty or 'Candidate / control total' not in control_summary.columns:
        return None

    plot_df = control_summary.copy()
    plot_df['Difference from control (%)'] = (plot_df['Candidate / control total'] - 1) * 100
    colors = []
    for ratio in plot_df['Candidate / control total']:
        if pd.isna(ratio):
            colors.append('#adb5bd')
        elif abs(ratio - 1) <= 0.05:
            colors.append('#1a9850')
        elif abs(ratio - 1) <= 0.20:
            colors.append('#fee08b')
        elif abs(ratio - 1) <= 0.50:
            colors.append('#fdae61')
        else:
            colors.append('#d73027')

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df['Control source'],
        y=plot_df['Candidate / control total'],
        marker_color=colors,
        text=[_control_ratio_interpretation(value) for value in plot_df['Candidate / control total']],
        textposition='auto',
        customdata=plot_df[['Overlap days', 'Mean delta (kWh)', 'MAE (kWh)', 'Min delta (kWh)', 'Max delta (kWh)']],
        hovertemplate=(
            '<b>%{x}</b><br>'
            'Candidate/control total: %{y:.2f}<br>'
            'Overlap days: %{customdata[0]}<br>'
            'Mean delta: %{customdata[1]:.1f} kWh/day<br>'
            'MAE: %{customdata[2]:.1f} kWh/day<br>'
            'Min/Max delta: %{customdata[3]:.1f} / %{customdata[4]:.1f} kWh<extra></extra>'
        )
    ))
    fig.add_hline(y=1, line_dash='dash', line_color='#495057', annotation_text='Perfect agreement')
    fig.update_layout(
        title='Candidate hourly target vs daily/meter controls',
        yaxis_title='Candidate total ÷ control total',
        xaxis_title='Control source',
        height=390,
        yaxis=dict(range=[0, max(1.5, plot_df['Candidate / control total'].max(skipna=True) * 1.15)])
    )
    return fig


@st.cache_data(ttl=3600)
def compare_daily_source_agreement(plant_name: str):
    """Compare daily 5-minute totals against billing-meter daily totals."""
    plant_id = DEMO_PLANTS[plant_name]['plant_id']
    five_path = find_source_file('inverter_five_minutes_generation_logs.csv')
    meter_path = find_source_file('plants_billing_meter_logs.csv')
    if five_path is None or meter_path is None:
        return None

    try:
        five = pd.read_csv(five_path)
        five = five[five['plant_id'] == plant_id].copy()
        five['date'] = pd.to_datetime(five['generation_date'], utc=True, errors='coerce').dt.tz_convert('Asia/Dhaka').dt.date
        five['generation_amount'] = pd.to_numeric(
            five['generation_amount'].astype(str).str.replace(',', '', regex=False), errors='coerce'
        )
        five_daily = (five.groupby('date')['generation_amount'].sum() / 1000).rename('five_min_kwh')

        meter = pd.read_csv(meter_path)
        meter = meter[meter['plant_id'] == plant_id].copy()
        meter['timestamp'] = pd.to_datetime(meter['date'], utc=True, errors='coerce')
        meter['date_local'] = meter['timestamp'].dt.tz_convert('Asia/Dhaka').dt.date
        meter['meter_reading'] = pd.to_numeric(meter['meter_reading'], errors='coerce')

        parts = []
        for _, meter_df in meter.dropna(subset=['timestamp', 'meter_reading']).groupby('meter_id'):
            meter_df = meter_df.sort_values('timestamp').copy()
            meter_df['meter_kwh'] = meter_df['meter_reading'].diff() / 1000
            meter_df.loc[meter_df['meter_kwh'] < 0, 'meter_kwh'] = np.nan
            parts.append(meter_df[['date_local', 'meter_kwh']])

        if not parts:
            return None

        meter_daily = (
            pd.concat(parts)
            .groupby('date_local')['meter_kwh']
            .sum()
            .rename('meter_kwh')
        )

        comparison = pd.concat([five_daily, meter_daily], axis=1).dropna()
        if len(comparison) < 10:
            return None

        return {
            'overlap_days': len(comparison),
            'correlation': comparison['five_min_kwh'].corr(comparison['meter_kwh']),
            'sum_ratio': comparison['five_min_kwh'].sum() / comparison['meter_kwh'].sum(),
            'mae_kwh': (comparison['five_min_kwh'] - comparison['meter_kwh']).abs().mean(),
        }
    except Exception:
        return None


def create_annual_completeness_heatmap(daily_stats: pd.DataFrame, start_date, end_date, title: str):
    """Create a calendar-style heatmap of daily 5-minute data completeness."""
    daily_stats_heat = daily_stats.copy()
    daily_stats_heat['Date'] = pd.to_datetime(daily_stats_heat['Date'])
    daily_stats_heat['DayOfWeek'] = daily_stats_heat['Date'].dt.dayofweek
    daily_stats_heat['WeekStart'] = (
        daily_stats_heat['Date'] - pd.to_timedelta(daily_stats_heat['DayOfWeek'], unit='D')
    ).dt.strftime('%Y-%m-%d')

    heatmap_data = daily_stats_heat.pivot_table(
        index='DayOfWeek',
        columns='WeekStart',
        values='Completeness_Pct',
        aggfunc='first'
    ).reindex(range(7))

    hover_data = daily_stats_heat.pivot_table(
        index='DayOfWeek',
        columns='WeekStart',
        values='Date',
        aggfunc=lambda x: x.iloc[0].strftime('%Y-%m-%d') if len(x) > 0 else ''
    ).reindex(range(7))

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=day_names,
        colorscale=[
            [0.0, '#d73027'],
            [0.5, '#fee08b'],
            [0.9, '#d9ef8b'],
            [1.0, '#1a9850']
        ],
        zmin=0,
        zmax=100,
        colorbar=dict(title='Completeness %'),
        hovertemplate='<b>Date:</b> %{customdata}<br><b>Completeness:</b> %{z:.1f}%<extra></extra>',
        customdata=hover_data.values
    ))

    fig_heatmap.update_layout(
        title=title,
        xaxis_title='Week starting',
        yaxis_title='Day of week',
        height=330,
        xaxis=dict(side='top'),
        yaxis=dict(autorange='reversed')
    )
    return fig_heatmap


def render_completeness_legend():
    """Render shared legend for completeness heatmaps."""
    col_leg1, col_leg2, col_leg3, col_leg4 = st.columns(4)
    with col_leg1:
        st.markdown('🟥 **0-50%**: Severely incomplete')
    with col_leg2:
        st.markdown('🟨 **50-90%**: Incomplete')
    with col_leg3:
        st.markdown('🟩 **90-100%**: Good')
    with col_leg4:
        st.markdown('🟢 **100%**: Perfect')


def render_data_sources_tab(plant_name: str):
    """Render the first-page data quality gate."""
    st.header('🔎 Data Sources & Quality Gate')
    st.markdown(
        'This page checks whether the available production data is trustworthy before modeling or forecasting. '
        'The current hourly ML pipeline uses **clean/hourly_model_targets.csv**, built from 5-minute inverter data. '
        'Incomplete hours are stored as **NaN/excluded targets**, not treated as zero production.'
    )
    st.caption(
        f"Name-redacted snapshot mode: freshness labels treat **{DATA_SNAPSHOT_TODAY.date()}** as today's date, "
        'because this local snapshot uses fixed historical data rather than a live feed.'
    )

    st.subheader('All plants data quality overview')
    st.caption(
        'Data-only score: 55% usable cleaned 5-minute daylight/modeling-hour coverage, 45% source agreement. '
        'Night hours are excluded from coverage. Source agreement uses hourly/daily inverter and meter CSVs as validation references. '
        'Hover over a plant name for the CSV source recap.'
    )

    overview_df = build_all_plants_quality_overview(_quality_overview_cache_token())
    if not overview_df.empty:
        _render_all_plants_quality_overview(overview_df)
    else:
        st.info('No plant-level quality overview could be calculated.')

    st.markdown('---')

    selected_detail_plant = render_plant_selector(
        'quality_detail_plant_selector',
        '🏭 Select plant to inspect data completeness'
    )

    st.markdown('---')
    st.subheader(f"Selected plant: {selected_detail_plant}")

    source_df = summarize_source_quality(selected_detail_plant)
    model_row = source_df[source_df['Used for model'] == '✅']

    if not model_row.empty:
        row = model_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric('Date period', row['Data period'] or 'n/a')
        records_value = (
            f"{row['Collected records'] / 1000:.1f}k / {row['Expected records'] / 1000:.1f}k"
            if pd.notna(row['Collected records']) and pd.notna(row['Expected records'])
            else 'n/a'
        )
        col2.metric('Records collected / expected', records_value)
        col3.metric('Completeness', f"{row['Completeness (%)']:.1f}%" if pd.notna(row['Completeness (%)']) else 'n/a')
        col4.metric('Freshness', row['Freshness'])
    else:
        st.warning('Could not summarize the 5-minute model source for the selected plant.')

    overview_row = overview_df[overview_df['Plant'] == selected_detail_plant] if not overview_df.empty else pd.DataFrame()
    if not overview_row.empty and overview_row.iloc[0]['Warnings'] != 'None':
        st.warning(f"Data warnings: {overview_row.iloc[0]['Warnings']}")
    else:
        st.success('No major data-source warnings for the selected plant.')

    if selected_detail_plant == 'Plant E':
        st.info(
            'Plant E has known source-data issues: the cleaned 5-minute hourly targets are sparse, '
            'and meter/reference sources do not fully agree with the inverter source.'
        )

    st.subheader('Annual completeness heatmap for model source')
    st.markdown('Visual representation of 5-minute data completeness across the entire period. Each cell represents one day.')

    df_5min = load_raw_5min_data(selected_detail_plant)
    result = data_quality.analyze_data_completeness(df_5min) if df_5min is not None else None
    if result:
        daily_stats, _, start_date, end_date, _, _, _ = result
        fig_heatmap = create_annual_completeness_heatmap(
            daily_stats,
            start_date,
            end_date,
            title=f'5-minute source data completeness calendar ({start_date.year} - {end_date.year})'
        )
        st.plotly_chart(fig_heatmap, width='stretch', key='source_quality_completeness_heatmap')
        render_completeness_legend()

        incomplete_days = daily_stats[daily_stats['Status'] != 'Complete'].copy()
        if not incomplete_days.empty:
            with st.expander(f"Incomplete day details ({len(incomplete_days)} days)"):
                st.dataframe(
                    incomplete_days[['Date', 'Expected_Records', 'Records_Collected', 'Missing_Records', 'Completeness_Pct', 'Status']]
                    .sort_values('Completeness_Pct')
                    .style.format({'Completeness_Pct': '{:.1f}%'}),
                    width='stretch',
                    height=360
                )
    else:
        st.warning('Unable to build annual completeness heatmap for the 5-minute source.')

    with st.expander('Show source comparison details', expanded=False):
        st.subheader('Selected Plant Data Sources')
        st.caption(
            'The model uses cleaned hourly targets generated from 5-minute inverter production data. '
            'Other sources are shown as validation references, not blindly merged into the target.'
        )
        if not model_row.empty and pd.notna(model_row.iloc[0]['Completeness (%)']) and model_row.iloc[0]['Completeness (%)'] < 95:
            st.warning(
                'Some expected inverter records are missing, so forecasts and alerts for this plant may be lower confidence.'
            )

        source_display = source_df.copy()
        source_display['Expected / collected'] = source_display.apply(
            lambda row: (
                f"{int(row['Collected records']):,} / {int(row['Expected records']):,}"
                if pd.notna(row['Collected records']) and pd.notna(row['Expected records'])
                else 'n/a'
            ),
            axis=1
        )
        source_display_cols = [
            'Source',
            'Use in dashboard',
            'Granularity',
            'Status',
            'Data period',
            'Expected / collected',
            'Completeness (%)',
            'Freshness',
            'Notes',
        ]
        st.dataframe(
            source_display[source_display_cols].style.format({'Completeness (%)': '{:.1f}'}, na_rep='n/a'),
            width='stretch',
            hide_index=True
        )

    with st.expander('Show source agreement details', expanded=False):
        st.subheader('Source Agreement')
        st.caption(
            'Compares daily totals from reference sources against the cleaned 5-minute inverter source. '
            'A ratio near 1.0 means the sources broadly agree and supports data reliability.'
        )
        source_agreement = compare_daily_generation_sources(selected_detail_plant)
        if source_agreement and 'error' not in source_agreement:
            source_summaries = source_agreement.get('source_summaries', pd.DataFrame())
            if not source_summaries.empty:
                agreement_cols = [
                    'Source',
                    'Overlap days',
                    'Correlation',
                    'Source / 5-min total',
                    'Daily MAE (kWh)',
                    'Matching days (%)',
                ]
                st.dataframe(
                    source_summaries[agreement_cols].style.format({
                        'Correlation': '{:.2f}',
                        'Source / 5-min total': '{:.2f}',
                        'Daily MAE (kWh)': '{:.1f}',
                        'Matching days (%)': '{:.1f}',
                    }, na_rep='n/a'),
                    width='stretch',
                    hide_index=True
                )
            else:
                st.info('No overlapping daily source agreement rows were available for this plant.')

            missing_sources = source_agreement.get('missing_sources', [])
            if missing_sources:
                missing_labels = ', '.join(source['label'] for source in missing_sources)
                st.caption(f'Missing or empty comparison sources: {missing_labels}')
        elif source_agreement and 'error' in source_agreement:
            st.warning(f"Unable to compare daily source agreement: {source_agreement['error']}")
        else:
            st.info('No source agreement comparison could be calculated for this plant.')

    with st.expander('Advanced: source fallback diagnostic', expanded=False):
        st.subheader('Experimental fallback check')
        st.caption(
            'Checks whether hourly inverter data could fill gaps in the 5-minute source. '
            'This remains diagnostic only; the current model does not use fallback-filled targets.'
        )
        training_diagnostics = build_hourly_training_source_diagnostics(selected_detail_plant)
        if not training_diagnostics.empty:
            total_hours = len(training_diagnostics)
            usable_hours = int(training_diagnostics['generation_kwh_final'].notna().sum())
            five_hours = int((training_diagnostics['target_source'] == '5min_inverter').sum())
            fallback_hours = int((training_diagnostics['target_source'] == 'hourly_inverter_fallback').sum())
            missing_hours = int((training_diagnostics['target_source'] == 'missing').sum())

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric('Usable hourly targets', f"{usable_hours / total_hours * 100:.1f}%" if total_hours else 'n/a')
            col_b.metric('From 5-min inverter', f"{five_hours / total_hours * 100:.1f}%" if total_hours else 'n/a')
            col_c.metric('Hourly fallback', f"{fallback_hours / total_hours * 100:.1f}%" if total_hours else 'n/a')
            col_d.metric('Still missing', f"{missing_hours / total_hours * 100:.1f}%" if total_hours else 'n/a')

            fig_training_heatmap = create_training_source_heatmap(training_diagnostics, days=90)
            if fig_training_heatmap is not None:
                st.plotly_chart(fig_training_heatmap, width='stretch', key='training_source_heatmap')
                st.markdown(
                    '🟩 **Preferred target**: usable 5-min inverter hour, with or without hourly confirmation  |  '
                    '🟨 **Fallback/usable**: hourly inverter fallback or moderate source difference  |  '
                    '🟧 **Disagree**: 5-min and hourly inverter differ by more than 20%  |  '
                    '🟥 **Missing**: no usable hourly target'
                )
                heatmap_window = training_diagnostics[
                    training_diagnostics['generation_date'] >= training_diagnostics['generation_date'].max() - pd.Timedelta(days=90)
                ]
                if heatmap_window['hourly_available'].sum() == 0:
                    st.caption(
                        'Note: the hourly inverter CSV has no records in this 90-day window, so the heatmap cannot show '
                        '5-min vs hourly confirmation there. Green still means the preferred 5-minute hourly target is usable.'
                    )

            st.subheader('Daily controls for candidate training target')
            st.caption(
                'The candidate hourly target is summed by day and compared to daily inverter, hourly meter, and billing meter controls. '
                'These controls identify bias or missing production; they are not used to fill hourly gaps in this design.'
            )
            daily_controls = build_daily_control_comparison(selected_detail_plant, training_diagnostics)
            control_summary = summarize_daily_control_deltas(daily_controls)
            if not control_summary.empty:
                st.dataframe(
                    control_summary.style.format({
                        'Candidate / control total': '{:.2f}',
                        'Mean delta (kWh)': '{:.1f}',
                        'Min delta (kWh)': '{:.1f}',
                        'Max delta (kWh)': '{:.1f}',
                        'MAE (kWh)': '{:.1f}',
                    }, na_rep='n/a'),
                    width='stretch',
                    hide_index=True
                )
                st.caption(
                    'Large differences do not necessarily mean a calculation bug: meter CSVs can represent plant-level meters, '
                    'multiple meter IDs, or a different boundary than inverter logs. Treat meter controls as diagnostics until '
                    'the meter mapping is confirmed.'
                )

                fig_ratio = create_control_ratio_bar_chart(control_summary)
                if fig_ratio is not None:
                    st.plotly_chart(fig_ratio, width='stretch', key='control_ratio_bar_chart')

            fig_daily_controls = create_daily_control_line_chart(daily_controls, days=180)
            if fig_daily_controls is not None:
                st.plotly_chart(fig_daily_controls, width='stretch', key='daily_control_line_chart')

        else:
            st.warning('Unable to build the 5-minute + hourly inverter training-target diagnostic for this plant.')


def render_executive_summary_tab(plant_name: str, df: pd.DataFrame, plant_info: dict):
    """Render decision-oriented landing page for portfolio/executive review."""
    st.header('📌 Executive Summary')
    st.caption(
        'High-level view of fleet data quality, model reliability, production performance, and recent underperformance signals.'
    )

    quality_df = build_all_plants_quality_overview(_quality_overview_cache_token())
    model_summary_df = build_all_plants_model_summary(_model_summary_cache_token())

    st.subheader('Fleet status overview')
    if not quality_df.empty and not model_summary_df.empty:
        fleet_df = quality_df[[
            'Plant', 'Data quality band', 'Usable daylight data (%)', 'Warnings'
        ]].merge(
            model_summary_df[['Plant', 'Best model', 'R²', 'WAPE (%)', 'Reliability']],
            on='Plant',
            how='left'
        )
        fleet_style = (
            fleet_df.style.format({
                'Usable daylight data (%)': '{:.1f}',
                'R²': '{:.3f}',
                'WAPE (%)': '{:.1f}',
            }, na_rep='n/a')
            .map(_style_completeness_cell, subset=['Usable daylight data (%)'])
            .map(_style_r2_cell, subset=['R²'])
            .map(_style_wape_cell, subset=['WAPE (%)'])
        )
        st.dataframe(
            fleet_style,
            width='stretch',
            hide_index=True
        )
    else:
        st.info('Fleet summary is not available.')

    st.markdown('---')
    selected_plant = render_plant_selector('executive_plant_selector', 'Choose plant for executive view')
    if selected_plant != plant_name:
        plant_name = selected_plant
        plant_info = DEMO_PLANTS[selected_plant]
        df = load_data(selected_plant)

    st.markdown('---')
    st.subheader(f'Selected plant snapshot: {selected_plant}')
    render_quality_warning(selected_plant)

    latest_date = df.index.max().date()
    today_metrics = metrics.calculate_daily_metrics(df, latest_date, st.session_state.alert_threshold) if 'ml_predicted_kwh' in df.columns else None
    recent_dates = sorted(set(df.index.date))[-90:]
    recent_underperformance = 0
    if 'ml_predicted_kwh' in df.columns:
        for date in recent_dates:
            day_metrics = metrics.calculate_daily_metrics(df, date, st.session_state.alert_threshold)
            if day_metrics and day_metrics['has_anomaly']:
                recent_underperformance += 1

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_production = df['generation_kwh'].sum()
        st.metric('Total production', f'{total_production / 1000:.1f} MWh')
    with col2:
        capacity_factor = metrics.calculate_capacity_factor(df, plant_info['capacity_kwp'])
        st.metric('Capacity factor', f'{capacity_factor:.2f}%')
    with col3:
        if today_metrics:
            st.metric('Latest performance', f"{today_metrics['performance_ratio']:.1f}%")
        else:
            st.metric('Latest performance', 'n/a')
    with col4:
        st.metric('90-day underperformance days', recent_underperformance)

    st.markdown('---')
    col_quality, col_trend = st.columns(2)

    with col_quality:
        st.subheader('Data completeness')
        df_5min = load_raw_5min_data(selected_plant)
        result = data_quality.analyze_data_completeness(df_5min) if df_5min is not None else None
        if result:
            daily_stats, _, start_date, end_date, _, _, _ = result
            fig_heatmap = create_annual_completeness_heatmap(
                daily_stats,
                start_date,
                end_date,
                title='5-minute source completeness'
            )
            st.plotly_chart(fig_heatmap, width='stretch', key='executive_completeness_heatmap')
        else:
            st.info('Completeness heatmap unavailable.')

    with col_trend:
        st.subheader('Production trend')
        daily_prod = df.groupby(df.index.date)['generation_kwh'].sum().reset_index()
        daily_prod.columns = ['date', 'production']
        daily_prod['date'] = pd.to_datetime(daily_prod['date'])
        daily_prod['production_clean'] = daily_prod['production'].replace(0, np.nan)
        daily_prod['rolling_avg_7d'] = daily_prod['production_clean'].rolling(7, min_periods=1).mean()
        daily_prod['rolling_avg_30d'] = daily_prod['production_clean'].rolling(30, min_periods=1).mean()

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['production'],
            mode='lines',
            name='Daily production',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['rolling_avg_7d'],
            mode='lines',
            name='7-day average',
            line=dict(color='blue', width=2)
        ))
        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['rolling_avg_30d'],
            mode='lines',
            name='30-day average',
            line=dict(color='darkblue', width=3)
        ))
        fig_trend.update_layout(
            title='Daily production with moving averages',
            xaxis_title='Date',
            yaxis_title='kWh/day',
            height=330,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_trend, width='stretch', key='executive_production_trend')

    st.markdown('---')
    st.subheader('Forecast outlook')
    st.caption('Compact production outlook. Without live weather, the app uses the historical weather rows after the last production date as a 5-day forecast scenario.')

    openweather_api_key = os.getenv('ow_key', '')
    try:
        weather_forecast, forecast_source_label, is_replay_source = get_outlook_weather_input(openweather_api_key, df)

        model, feature_columns = model_loader.load_plant_model(selected_plant)
        if weather_forecast is not None and len(weather_forecast) > 0 and model is not None and feature_columns is not None:
            forecast_with_features = forecasting.make_forecast(
                model=model,
                feature_columns=feature_columns,
                weather_df=weather_forecast,
                historical_df=df,
                capacity_kwp=plant_info['capacity_kwp']
            )
        else:
            forecast_with_features = None

        if forecast_with_features is not None and not forecast_with_features.empty:
            total_5day = forecast_with_features['predicted_kwh'].sum()
            num_days = len(set(forecast_with_features.index.date))
            daily_avg = total_5day / num_days if num_days else 0
            max_hour = forecast_with_features['predicted_kwh'].max()

            is_replay = is_replay_source
            total_label = 'Replay total' if is_replay else '5-day forecast'
            avg_label = 'Replay daily average' if is_replay else 'Forecast daily average'
            peak_label = 'Replay peak hour' if is_replay else 'Forecast peak hour'

            col_forecast_1, col_forecast_2, col_forecast_3 = st.columns(3)
            col_forecast_1.metric(total_label, f'{total_5day:.1f} kWh')
            col_forecast_2.metric(avg_label, f'{daily_avg:.1f} kWh/day')
            col_forecast_3.metric(peak_label, f'{max_hour:.1f} kWh')

            fig_forecast = visualization.plot_forecast_daily_plotly(forecast_with_features, historical_df=df)
            st.plotly_chart(fig_forecast, width='stretch', key='executive_forecast_daily')
            st.caption(f'Forecast source: {forecast_source_label}.')
        else:
            st.info('Forecast outlook is not available for the selected plant.')
    except Exception as e:
        st.warning(f'Unable to build forecast outlook: {e}')

    st.caption('Use the detailed tabs for source diagnostics, model comparison, forecast assumptions, and anomaly drill-down.')

def create_alert_rules_ui():
    """Create UI for configuring anomaly detection threshold."""
    st.subheader("🔔 Detection Threshold")

    col1, col2 = st.columns(2)

    with col1:
        alert_threshold = st.slider(
            "Underperformance threshold (%)",
            min_value=5,
            max_value=50,
            value=st.session_state.alert_threshold,
            help="Flag a day when actual production is this percentage below predicted production."
        )
        st.session_state.alert_threshold = alert_threshold

    with col2:
        st.write("**Detection rule:**")
        st.write(f"- Daily performance < {100 - alert_threshold}%")
        st.write(f"- Actual production is more than {alert_threshold}% below model expectation")
        st.caption("This is a threshold-based anomaly review, not an email-delivery system.")

    return alert_threshold, False


def main():
    """Main Streamlit app"""

    # Top Bar
    st.markdown('<p class="main-header">☀️ Solar Forecasting & Anomaly Detection Dashboard</p>', unsafe_allow_html=True)

    selected_plant = st.session_state.selected_plant
    plant_info = DEMO_PLANTS[selected_plant]

    # Load data for the currently selected plant. The Data Sources & Quality page
    # owns plant selection so the first page starts with all-plant data quality.
    with st.spinner('Loading data...'):
        df = load_data(selected_plant)

    # Subtitle
    st.markdown("""
    <div style='text-align: center; margin-bottom: 0.8rem; color: #666; font-size: 0.95rem;'>
        <p style='margin: 0;'>PV production forecasting, monitoring and anomaly analytics</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if df is None:
        st.error("❌ Unable to load data. Please ensure train_model.py has been run.")
        return

    # Main tabs: executive summary first, then deeper analysis pages.
    tabs = st.tabs([
        "📌 Executive Summary",
        "🔎 Data Sources & Quality",
        "📊 Overview",
        "📉 Performance Analytics",
        "🤖 Model, Data & Features",
        "🔮 Forecast",
        "🚨 Anomaly Review"
    ])

    # TAB 0: EXECUTIVE SUMMARY
    with tabs[0]:
        render_executive_summary_tab(selected_plant, df, plant_info)

    # TAB 1: DATA SOURCES & QUALITY
    with tabs[1]:
        render_data_sources_tab(selected_plant)

    # TAB 2: OVERVIEW
    with tabs[2]:
        selected_plant = render_plant_selector('overview_plant_selector')
        plant_info = DEMO_PLANTS[selected_plant]

        st.header("Daily Overview & Key Metrics")

        latest_date = df.index.max().date()

        if 'ml_predicted_kwh' in df.columns:
            today_metrics = metrics.calculate_daily_metrics(df, latest_date, st.session_state.alert_threshold)

            if today_metrics:
                # Get peak production time
                daily_data = df[df.index.date == latest_date]
                peak_time = daily_data['generation_kwh'].idxmax()

                col1, col2, col3, col4, col5 = st.columns(5)

                with col1:
                    st.metric("📅 Latest Date", latest_date.strftime("%Y-%m-%d"))

                with col2:
                    st.metric(
                        "⚡ Actual Production (Daily Total)",
                        f"{today_metrics['actual_total']:.1f} kWh",
                        delta=f"{today_metrics['actual_total'] - today_metrics['predicted_total']:.1f} kWh",
                        help=f"Total energy produced on {latest_date.strftime('%Y-%m-%d')}"
                    )

                with col3:
                    st.metric(
                        "🎯 Predicted",
                        f"{today_metrics['predicted_total']:.1f} kWh",
                        help=f"ML model prediction for {latest_date.strftime('%Y-%m-%d')} based on weather conditions and historical data"
                    )

                with col4:
                    st.metric(
                        "📊 Performance",
                        f"{today_metrics['performance_ratio']:.1f}%",
                        delta=f"{today_metrics['performance_ratio'] - 100:.1f}%",
                        help=f"Performance ratio for {latest_date.strftime('%Y-%m-%d')}: (Actual / Predicted) × 100%. Values >100% indicate better than expected performance"
                    )

                with col5:
                    st.metric(
                        "☀️ Peak Hour",
                        f"{today_metrics['actual_peak']:.1f} kWh",
                        help=f"Maximum hourly production on {peak_time.strftime('%Y-%m-%d at %H:%M')}"
                    )

                # Alert box
                if today_metrics['has_anomaly']:
                    st.markdown(f"""
                    <div class="alert-box">
                        <h3>⚠️ Performance Alert</h3>
                        <p>Actual production is <strong>{100 - today_metrics['performance_ratio']:.1f}%</strong> below predicted value.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ System Operating Normally</h3>
                        <p>Performance is within expected range.</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Recent trends
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📅 Calendar Heatmap")
            # Get available years from data
            available_years = sorted(df.index.year.unique(), reverse=True)
            default_year = available_years[0] if len(available_years) > 0 else pd.Timestamp.now().year
            year = st.selectbox("Select Year", available_years, index=0)
            fig_calendar = visualization.plot_calendar_heatmap(df, year=year)
            st.plotly_chart(fig_calendar, width="stretch", key="calendar_heatmap")

        with col2:
            st.subheader("🕐 Time-of-Day Pattern")
            fig_tod = visualization.plot_time_of_day_analysis(df)
            st.plotly_chart(fig_tod, width="stretch", key="time_of_day_overview")

    # TAB 4: MODEL & FEATURES
    with tabs[4]:
        st.header("🤖 Model Comparison & Feature Engineering")

        st.subheader('All-Plant Model Summary')
        model_summary_df = build_all_plants_model_summary(_model_summary_cache_token())
        model_summary_style = (
            model_summary_df.style
            .format({
                'R²': '{:.3f}',
                'WAPE (%)': '{:.1f}',
                'MAE (kWh)': '{:.2f}',
                'RMSE (kWh)': '{:.2f}',
                'Training Time (s)': '{:.2f}',
            }, na_rep='n/a')
            .map(_style_r2_cell, subset=['R²'])
            .map(_style_wape_cell, subset=['WAPE (%)'])
            .background_gradient(subset=['MAE (kWh)', 'RMSE (kWh)', 'Training Time (s)'], cmap='RdYlGn_r')
        )
        st.dataframe(
            model_summary_style,
            width='stretch',
            hide_index=True
        )
        st.caption('Best model is selected per plant using test R². Models are trained on cleaned hourly targets; incomplete source hours and weak daily source-agreement days are excluded, not counted as zero. WAPE is total absolute error divided by total actual production.')

        st.info(
            'Model scores are for **reliable training/evaluation days**: hourly targets come from cleaned 5-minute inverter data, '
            'and plant-days with strong daily inverter disagreement are excluded using plant-specific thresholds. '
            'Plant E has known source-data issues, so weaker metrics are expected.'
        )

        with st.expander('Show training data policy', expanded=False):
            threshold_by_plant = ML_CONFIG.get('max_daily_inverter_ratio_difference_by_plant', {})
            policy_rows = []
            for policy_plant in DEMO_PLANTS:
                threshold = threshold_by_plant.get(
                    policy_plant,
                    ML_CONFIG.get('max_daily_inverter_ratio_difference', 1.0)
                )
                policy_rows.append({
                    'Plant': policy_plant,
                    'Hourly target source': 'clean/hourly_model_targets.csv',
                    'Incomplete-hour rule': '<80% 5-min completeness → NaN/excluded',
                    'Daily source-agreement filter': f'Exclude if daily inverter differs by >{threshold * 100:.0f}%',
                    'Interpretation': 'Known source-data issues' if policy_plant == 'Plant E' else 'Reliable-days model',
                })
            st.dataframe(pd.DataFrame(policy_rows), width='stretch', hide_index=True)
            st.caption(
                'This filter is for model training/evaluation only. It does not fill missing targets and does not hide historical data in the dashboard.'
            )

        st.markdown('---')
        selected_plant = render_plant_selector('model_plant_selector', 'Choose plant for detailed model review')
        plant_info = DEMO_PLANTS[selected_plant]

        st.markdown('---')
        st.subheader(f'Detailed Model Review: {selected_plant}')
        render_quality_warning(selected_plant)
        selected_threshold = ML_CONFIG.get('max_daily_inverter_ratio_difference_by_plant', {}).get(
            selected_plant,
            ML_CONFIG.get('max_daily_inverter_ratio_difference', 1.0)
        )
        st.caption(
            f'Training/evaluation policy for {selected_plant}: cleaned hourly 5-minute target; incomplete hours excluded; '
            f'days excluded only when daily inverter validation differs by more than {selected_threshold * 100:.0f}%. '
            f'{"Plant E has known source-data issues, so metrics are lower confidence." if selected_plant == "Plant E" else "Metrics should be read as reliable-days model performance."}'
        )

        # MODEL COMPARISON SECTION
        comparison_df = model_loader.load_all_models_comparison(selected_plant)

        if comparison_df is not None:
            error_pct_col = 'Test WAPE (%)' if 'Test WAPE (%)' in comparison_df.columns else 'Test MAPE (%)'
            error_pct_label = 'WAPE' if error_pct_col == 'Test WAPE (%)' else 'MAPE'

            best_model = comparison_df.iloc[0]

            st.subheader("🏆 Selected Model Performance")
            col_model, col_r2, col_wape, col_mae, col_rmse = st.columns(5)
            col_model.metric("Best model", best_model['Model'])
            col_r2.metric("R²", f"{best_model['Test R²']:.3f}")
            col_wape.metric(error_pct_label, f"{best_model[error_pct_col]:.1f}%")
            col_mae.metric("MAE", f"{best_model['Test MAE (kWh)']:.2f} kWh")
            col_rmse.metric("RMSE", f"{best_model['Test RMSE (kWh)']:.2f} kWh")

            st.caption(
                "R² measures explained variance; WAPE is total absolute error divided by total actual production. "
                "Lower MAE/RMSE/WAPE is better."
            )

            st.markdown("---")

            st.subheader("📈 Model Quality Comparison")
            fig_comp = visualization.plot_model_comparison_metrics(comparison_df)
            if fig_comp:
                st.plotly_chart(fig_comp, width="stretch", key="model_comparison")

            with st.expander("Show detailed model comparison", expanded=False):
                comparison_style = (
                    comparison_df.style
                    .format({
                        'Test MAE (kWh)': '{:.3f}',
                        'Test RMSE (kWh)': '{:.3f}',
                        'Test R²': '{:.4f}',
                        error_pct_col: '{:.2f}',
                        'Training Time (s)': '{:.2f}'
                    })
                    .map(_style_r2_cell, subset=['Test R²'])
                    .map(_style_wape_cell, subset=[error_pct_col])
                    .background_gradient(subset=['Test MAE (kWh)', 'Test RMSE (kWh)', 'Training Time (s)'], cmap='RdYlGn_r')
                )
                st.dataframe(
                    comparison_style,
                    width='stretch'
                )
        else:
            st.warning("Model comparison data not available. Please run train_model.py first.")

        st.markdown("---")

        # FEATURE ENGINEERING SECTION
        st.subheader("🔧 Feature Engineering Overview")

        st.markdown("""
        Model inputs are grouped by time, weather, solar position, and recent production history.
        """)

        # Load feature columns
        _, feature_columns = model_loader.load_multiple_models(selected_plant)

        # Load full featured data for analysis
        with st.spinner('Loading feature data...'):
            df_full = load_full_featured_data(selected_plant)

        if feature_columns:
            st.subheader(f"📋 Model Input Groups ({len(feature_columns)} features)")

            # Categorize features
            temporal_features = [f for f in feature_columns if any(x in f for x in ['hour', 'day', 'month', 'year', 'sin', 'cos'])]
            weather_features = [f for f in feature_columns if any(x in f for x in ['temp', 'humidity', 'cloud', 'wind', 'pressure', 'rain'])]
            solar_features = [f for f in feature_columns if any(x in f for x in ['solar', 'azimuth', 'elevation', 'zenith'])]
            lag_features = [f for f in feature_columns if 'lag' in f or 'mean' in f or 'rolling' in f]
            other_features = [f for f in feature_columns if f not in temporal_features + weather_features + solar_features + lag_features]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("🕐 Temporal Features", len(temporal_features))
                with st.expander("View Temporal Features"):
                    for f in temporal_features:
                        st.write(f"- {f}")

            with col2:
                st.metric("🌤️ Weather Features", len(weather_features))
                with st.expander("View Weather Features"):
                    for f in weather_features:
                        st.write(f"- {f}")

            with col3:
                st.metric("☀️ Solar Features", len(solar_features))
                with st.expander("View Solar Features"):
                    for f in solar_features:
                        st.write(f"- {f}")

            with col4:
                st.metric("📊 Historical Features", len(lag_features))
                with st.expander("View Historical Features"):
                    for f in lag_features:
                        st.write(f"- {f}")

            st.markdown("---")

            # Only show graphs if full data is available
            if df_full is not None:
                st.subheader("🔗 Feature Relationships with Production")
                st.caption(
                    "Shows the strongest simple correlations between engineered inputs and solar production. "
                    "This is an interpretability aid, not model-native feature importance."
                )
                fig_top_features = visualization.plot_top_features_importance(df_full, feature_columns, top_n=10)
                if fig_top_features:
                    st.plotly_chart(fig_top_features, width="stretch", key="top_features")
                    st.caption(
                        "Blue = positive relationship with production; red = negative relationship. "
                        "Correlation does not prove causation and may miss nonlinear effects."
                    )
                else:
                    st.info("Feature correlation data not available.")
            else:
                st.warning("Unable to load full feature data. Graph is not available.")
        else:
            st.warning("Feature columns not available.")


    # TAB 6: ANOMALY REVIEW
    with tabs[6]:
        selected_plant = render_plant_selector('alerts_plant_selector', 'Choose plant for anomaly review')
        plant_info = DEMO_PLANTS[selected_plant]

        st.header("🚨 Anomaly Review")
        st.caption(
            "Compares actual daily production against model-expected production to flag possible underperformance or missing-data events."
        )
        render_quality_warning(selected_plant)

        alert_threshold, enable_email = create_alert_rules_ui()

        st.markdown("---")

        st.subheader("⚠️ Recent Underperformance Days")
        st.caption("Review days from the last 90 days where actual production fell below the selected threshold.")

        # Find anomalies
        all_dates = sorted(set(df.index.date))[-90:]  # Last 90 days
        anomalous_days = []

        if 'ml_predicted_kwh' in df.columns:
            for date in all_dates:
                day_metrics = metrics.calculate_daily_metrics(df, date, alert_threshold)
                if day_metrics and day_metrics['has_anomaly']:
                    anomalous_days.append({
                        'Date': date,
                        'Actual (kWh)': day_metrics['actual_total'],
                        'Predicted (kWh)': day_metrics['predicted_total'],
                        'Performance (%)': day_metrics['performance_ratio'],
                        'Deficit (kWh)': day_metrics['predicted_total'] - day_metrics['actual_total']
                    })

            if anomalous_days:
                st.info(f"Detected {len(anomalous_days)} underperformance days in the last 90 days.")

                anomaly_df = pd.DataFrame(anomalous_days)
                anomaly_df = anomaly_df.sort_values('Date', ascending=False)

                # Replace Deficit column with Status column that shows "Missing Data" when Actual = 0
                anomaly_df['Status'] = anomaly_df.apply(
                    lambda row: 'Missing Data' if row['Actual (kWh)'] == 0 else f"{row['Deficit (kWh)']:.1f} kWh deficit",
                    axis=1
                )

                # Display dataframe with Status instead of Deficit
                display_df = anomaly_df[['Date', 'Actual (kWh)', 'Predicted (kWh)', 'Performance (%)', 'Status']]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Underperformance Days", len(anomalous_days))
                with col2:
                    avg_perf = anomaly_df['Performance (%)'].mean()
                    st.metric("Avg Performance", f"{avg_perf:.1f}%")
                with col3:
                    # Calculate deficit only for actual anomalies (exclude missing data)
                    total_deficit = anomaly_df[anomaly_df['Actual (kWh)'] > 0]['Deficit (kWh)'].sum()
                    st.metric("Total Energy Deficit", f"{total_deficit:.1f} kWh")

                # Function to apply conditional styling for Performance (%)
                def style_performance(row):
                    styles = [''] * len(row)
                    # If Status is "Missing Data", apply grey background to Performance column
                    if row['Status'] == 'Missing Data':
                        styles[3] = 'background-color: #D3D3D3; color: black'
                    return styles

                styled_df = display_df.style.format({
                    'Actual (kWh)': '{:.1f}',
                    'Predicted (kWh)': '{:.1f}',
                    'Performance (%)': '{:.1f}'
                }).apply(style_performance, axis=1)

                # Apply gradient only to rows where Status is not "Missing Data"
                for idx in display_df.index:
                    if display_df.loc[idx, 'Status'] != 'Missing Data':
                        perf_value = display_df.loc[idx, 'Performance (%)']
                        # Calculate color based on performance (red to yellow to green)
                        if perf_value < 65:
                            color = f'background-color: rgba(255, {int((perf_value-50)/15*255)}, 0, 0.7)'
                        elif perf_value < 80:
                            color = f'background-color: rgba({int((80-perf_value)/15*255)}, 255, 0, 0.7)'
                        else:
                            color = f'background-color: rgba(0, 255, {int((perf_value-80)/20*100)}, 0.7)'
                        styled_df = styled_df.set_properties(subset=(idx, 'Performance (%)'), **{'background-color': color.split(':')[1].strip()})

                st.markdown("**Flagged days**")
                st.dataframe(
                    styled_df,
                    width='stretch',
                    height=400
                )

                # Anomaly trend - separate missing data from actual anomalies
                fig_anomaly = go.Figure()

                # Separate missing data (Actual = 0) from actual anomalies
                missing_data = anomaly_df[anomaly_df['Actual (kWh)'] == 0].copy()
                actual_anomalies = anomaly_df[anomaly_df['Actual (kWh)'] > 0].copy()

                # Add trace for actual anomalies (red)
                if not actual_anomalies.empty:
                    fig_anomaly.add_trace(go.Scatter(
                        x=actual_anomalies['Date'],
                        y=actual_anomalies['Performance (%)'],
                        mode='markers+lines',
                        marker=dict(size=10, color='red'),
                        line=dict(color='red', width=2),
                        name='Anomaly',
                        customdata=actual_anomalies[['Actual (kWh)', 'Predicted (kWh)', 'Deficit (kWh)']].values,
                        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                                      '<b>Performance:</b> %{y:.1f}%<br>' +
                                      '<b>Actual:</b> %{customdata[0]:.1f} kWh<br>' +
                                      '<b>Predicted:</b> %{customdata[1]:.1f} kWh<br>' +
                                      '<b>Deficit:</b> %{customdata[2]:.1f} kWh<extra></extra>'
                    ))

                # Add trace for missing data (grey)
                if not missing_data.empty:
                    fig_anomaly.add_trace(go.Scatter(
                        x=missing_data['Date'],
                        y=missing_data['Performance (%)'],
                        mode='markers+lines',
                        marker=dict(size=10, color='grey'),
                        line=dict(color='grey', width=2, dash='dash'),
                        name='Missing Data',
                        customdata=missing_data[['Predicted (kWh)']].values,
                        hovertemplate='<b>Date:</b> %{x|%Y-%m-%d}<br>' +
                                      '<b>Status:</b> Missing Data<br>' +
                                      '<b>Predicted:</b> %{customdata[0]:.1f} kWh<extra></extra>'
                    ))
                fig_anomaly.add_hline(y=80, line_dash="dash", line_color="orange")
                fig_anomaly.update_layout(
                    title='Underperformance Trend',
                    xaxis_title='Date',
                    yaxis_title='Performance (%)',
                    height=400
                )
                st.plotly_chart(fig_anomaly, width="stretch", key="anomaly_trend")
            else:
                st.success("✅ No underperformance days detected in the last 90 days.")
        else:
            st.info("Predictions are not available for anomaly review.")

    # TAB 3: PERFORMANCE ADEMO_CYTICS
    with tabs[3]:
        selected_plant = render_plant_selector('performance_plant_selector')
        plant_info = DEMO_PLANTS[selected_plant]

        st.header("📉 Performance Analytics")

        col1, col2, col3 = st.columns(3)

        with col1:
            total_production = df['generation_kwh'].sum()
            st.metric("Total Production", f"{total_production/1000:.1f} MWh")

        with col2:
            capacity_factor = metrics.calculate_capacity_factor(df, plant_info['capacity_kwp'])
            st.metric(
                "Capacity Factor",
                f"{capacity_factor:.2f}%",
                help="Capacity Factor measures how efficiently the solar plant operates compared to its maximum theoretical output. "
                     "Formula: (Total Actual Production / Maximum Possible Production) × 100. "
                     "It accounts for all hours including nighttime (0 production), weather conditions, seasonal variations, and system efficiency. "
                     "Typical values: 10-15% (good for tropical regions), 15-20% (very good), 20-25% (exceptional in ideal desert conditions). "
                     "A sudden drop may indicate maintenance issues, while consistent low values suggest systematic problems."
            )

        with col3:
            daily_avg = df.groupby(df.index.date)['generation_kwh'].sum().mean()
            st.metric("Daily Average", f"{daily_avg:.1f} kWh")

        st.markdown("---")

        # Performance over time
        st.subheader("📈 Long-term Performance Trend")

        daily_prod = df.groupby(df.index.date)['generation_kwh'].sum().reset_index()
        daily_prod.columns = ['date', 'production']
        daily_prod['date'] = pd.to_datetime(daily_prod['date'])

        # Replace 0 values with NaN to exclude missing data from rolling averages
        daily_prod['production_clean'] = daily_prod['production'].replace(0, np.nan)
        daily_prod['rolling_avg_7d'] = daily_prod['production_clean'].rolling(7, min_periods=1).mean()
        daily_prod['rolling_avg_30d'] = daily_prod['production_clean'].rolling(30, min_periods=1).mean()

        fig_trend = go.Figure()

        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['production'],
            mode='lines',
            name='Daily Production',
            line=dict(color='lightblue', width=1),
            opacity=0.5
        ))

        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['rolling_avg_7d'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='blue', width=2)
        ))

        fig_trend.add_trace(go.Scatter(
            x=daily_prod['date'],
            y=daily_prod['rolling_avg_30d'],
            mode='lines',
            name='30-Day Average',
            line=dict(color='darkblue', width=3)
        ))

        fig_trend.update_layout(
            title='Production Trend with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Daily Production (kWh)',
            height=500,
            hovermode='x unified'
        )

        st.plotly_chart(fig_trend, width="stretch", key="performance_trend")

        st.markdown("---")

        # ENHANCED VISUALIZATIONS (merged from Enhanced Analytics)
        st.subheader("📊 Monthly Production Trends")
        fig_monthly = visualization.plot_monthly_comparison(df)
        st.plotly_chart(fig_monthly, width="stretch", key="monthly_trends")

        st.markdown("---")

        st.subheader("📅 Last 7 Days Production")
        fig_weekly = visualization.plot_weekly_production_pattern(df)
        st.plotly_chart(fig_weekly, width="stretch", key="weekly_pattern_analytics")

        st.markdown("---")

        # Monsoon-aware seasonal analysis for Bangladesh
        st.subheader("🌧️ Monsoon-Season Performance")
        st.caption("Production grouped by Bangladesh climate seasons, where cloud and rain patterns strongly affect solar output.")
        df_seasonal = df.copy()
        df_seasonal['month'] = df_seasonal.index.month
        season_order = ['Winter / Dry', 'Pre-monsoon', 'Monsoon', 'Post-monsoon']
        df_seasonal['season'] = df_seasonal['month'].map({
            12: 'Winter / Dry', 1: 'Winter / Dry', 2: 'Winter / Dry',
            3: 'Pre-monsoon', 4: 'Pre-monsoon', 5: 'Pre-monsoon',
            6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
            10: 'Post-monsoon', 11: 'Post-monsoon'
        })

        seasonal_prod = df_seasonal.groupby('season')['generation_kwh'].agg(['sum', 'mean', 'std']).reindex(season_order).reset_index()

        fig_seasonal = go.Figure()
        fig_seasonal.add_trace(go.Bar(
            x=seasonal_prod['season'],
            y=seasonal_prod['sum'],
            marker_color=['#6baed6', '#fd8d3c', '#3182bd', '#74c476'],
            text=[f"{x/1000:.1f}k kWh" for x in seasonal_prod['sum']],
            textposition='inside',
            textfont=dict(color='white', size=16, family='Arial')
        ))
        fig_seasonal.update_layout(
            title='Total Production by Bangladesh Climate Season',
            xaxis_title='Climate season',
            yaxis_title='Total Production (kWh)',
            height=400
        )
        st.plotly_chart(fig_seasonal, width="stretch", key="seasonal_analysis")
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #666; margin-top: -0.5rem;">
                <strong>Season mapping:</strong>
                Dec–Feb = Winter / Dry · clearer skies, cooler temperatures &nbsp;|&nbsp;
                Mar–May = Pre-monsoon · hotter, variable clouds/storms &nbsp;|&nbsp;
                Jun–Sep = Monsoon · higher cloud/rain impact &nbsp;|&nbsp;
                Oct–Nov = Post-monsoon · transition back to clearer conditions
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Highest and lowest production days
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🏆 Top 5 Highest Production Days")
            top_days = df.groupby(df.index.date)['generation_kwh'].sum().nlargest(5).reset_index()
            top_days.columns = ['Date', 'Production (kWh)']
            st.dataframe(
                top_days.style.format({'Production (kWh)': '{:.1f}'}),
                width='stretch'
            )

        with col2:
            st.subheader("📉 Top 5 Lowest Production Days")
            # Exclude days with 0 production (missing data)
            daily_totals = df.groupby(df.index.date)['generation_kwh'].sum()
            lowest_days = daily_totals[daily_totals > 0].nsmallest(5).reset_index()
            lowest_days.columns = ['Date', 'Production (kWh)']
            st.dataframe(
                lowest_days.style.format({'Production (kWh)': '{:.1f}'}),
                width='stretch'
            )

        st.caption("Lowest-production table excludes days with no recorded production or insufficient source data.")

    # TAB 5: FORECAST / REPLAY
    with tabs[5]:
        selected_plant = render_plant_selector('forecast_plant_selector')
        plant_info = DEMO_PLANTS[selected_plant]

        st.header("🔮 Production Forecast / Replay")
        st.caption(
            "Forecast pipeline: weather input → feature engineering → trained plant model → expected production. "
            "Forecast confidence depends on weather quality and model reliability for the selected plant."
        )
        render_quality_warning(selected_plant)

        # Use live OpenWeather data when configured; otherwise use a historical replay for local review.
        openweather_api_key = os.getenv('ow_key', '')
        plant_info = DEMO_PLANTS[selected_plant]

        with st.spinner('Preparing weather input...'):
            weather_forecast, forecast_source_label, is_replay_source = get_outlook_weather_input(openweather_api_key, df)

        if weather_forecast is not None and len(weather_forecast) > 0:
            # Load model and features for selected plant
            model, feature_columns = model_loader.load_plant_model(selected_plant)

            if model is not None and feature_columns is not None:
                with st.spinner('Creating features and predicting production...'):
                    # Use the complete forecasting pipeline
                    forecast_with_features = forecasting.make_forecast(
                        model=model,
                        feature_columns=feature_columns,
                        weather_df=weather_forecast,
                        historical_df=df,
                        capacity_kwp=plant_info['capacity_kwp']
                    )

                    if forecast_with_features is not None:

                        # Summary metrics
                        is_replay = is_replay_source
                        summary_title = '📊 Historical Model Replay' if is_replay else '📊 Forecast Summary'
                        st.subheader(f"{summary_title} for {selected_plant}")

                        total_5day = forecast_with_features['predicted_kwh'].sum()
                        # Count actual days in forecast (using unique dates)
                        num_days = len(set(forecast_with_features.index.date))
                        daily_avg = total_5day / num_days if num_days > 0 else 0
                        max_hour = forecast_with_features['predicted_kwh'].max()

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Replay/Forecast Production", f"{total_5day:.1f} kWh")
                        with col2:
                            st.metric("Daily Average", f"{daily_avg:.1f} kWh/day")
                        with col3:
                            st.metric("Peak Hour", f"{max_hour:.1f} kWh")

                        # Daily aggregated forecast with historical data
                        st.subheader("📅 Daily Production Replay" if is_replay else "📅 Daily Production Forecast")
                        fig_daily = visualization.plot_forecast_daily_plotly(forecast_with_features, historical_df=df)
                        st.plotly_chart(fig_daily, width='stretch', key="forecast_daily")

                        with st.expander("Forecast input details", expanded=False):
                            st.write(f"**Weather source:** {forecast_source_label}")
                            st.write(f"**Weather points:** {len(weather_forecast)}")
                            st.write(f"**Plant:** {selected_plant} ({plant_info['capacity_kwp']} kWp)")
                            st.write("**Location:** Shared weather location for all plants")
                            if is_replay:
                                st.caption("Live weather is not available and no post-production weather window was found, so this uses the latest historical weather rows as a model replay.")
                            elif forecast_source_label.startswith('Historical weather'):
                                st.caption("Live weather is not configured, so this uses historical weather rows after the last production timestamp as the 5-day forecast window.")

                        # Weather conditions for all 10 days (5 historical + 5 forecast)
                        st.subheader("🌤️ Weather Inputs & Production Replay" if is_replay else "🌤️ Weather Assumptions & Production Outlook")
                        st.caption(
                            "Shows latest real historical weather rows used for model replay."
                            if is_replay else
                            "Shows recent historical context alongside the weather inputs used for the next production forecast."
                        )

                        # Prepare weather data for forecast days
                        forecast_dates = sorted(set(forecast_with_features.index.date))

                        # Create daily summary for forecast
                        weather_summary = []
                        for date in forecast_dates:
                            day_data = forecast_with_features[forecast_with_features.index.date == date]

                            summary = {
                                'Date': date.strftime('%a, %b %d'),
                                'Full_Date': date
                            }

                            if 'temp' in day_data.columns:
                                summary['Min Temp (°C)'] = day_data['temp'].min()
                                summary['Max Temp (°C)'] = day_data['temp'].max()
                            if 'humidity' in day_data.columns:
                                summary['Avg Humidity (%)'] = day_data['humidity'].mean()
                            if 'clouds_all' in day_data.columns:
                                summary['Avg Clouds (%)'] = day_data['clouds_all'].mean()
                            if 'wind_speed' in day_data.columns:
                                summary['Avg Wind (m/s)'] = day_data['wind_speed'].mean()

                            if is_replay:
                                actual_day = df[df.index.date == date]['generation_kwh'].sum()
                                summary['Actual Production (kWh)'] = actual_day if not pd.isna(actual_day) else None
                                summary['Type'] = 'Replay'
                            else:
                                summary['Actual Production (kWh)'] = None
                                summary['Type'] = 'Forecast'
                            summary['Predicted Production (kWh)'] = day_data['predicted_kwh'].sum()

                            weather_summary.append(summary)

                        # Add historical data (last 5 days) with weather data
                        # Load weather data explicitly with humidity and wind_speed
                        try:
                            weather_hist = pd.read_csv(DATA_PATHS['weather'])
                            weather_hist['generation_date'] = pd.to_datetime(
                                weather_hist['dt_iso'].str.replace(' UTC', ''), utc=True
                            ).dt.tz_convert(PLANT_CONFIG['timezone'])
                            # Select all needed columns including humidity and wind_speed
                            weather_cols = ['generation_date', 'temp', 'humidity', 'clouds_all', 'wind_speed']
                            weather_hist = weather_hist[weather_cols]
                        except Exception as e:
                            st.warning(f"Could not load weather data: {e}")
                            weather_hist = None

                        last_5_dates = [] if is_replay else sorted(set(df.tail(120).index.date))[-5:]

                        for date in last_5_dates:
                            day_data_hist = df[df.index.date == date]

                            summary = {
                                'Date': date.strftime('%a, %b %d'),
                                'Full_Date': date
                            }

                            # Try to get weather data for this date
                            if weather_hist is not None:
                                # Filter weather data for this date
                                weather_day = weather_hist[weather_hist['generation_date'].dt.date == date]

                                if not weather_day.empty:
                                    if 'temp' in weather_day.columns:
                                        summary['Min Temp (°C)'] = weather_day['temp'].min()
                                        summary['Max Temp (°C)'] = weather_day['temp'].max()
                                    else:
                                        summary['Min Temp (°C)'] = None
                                        summary['Max Temp (°C)'] = None

                                    if 'humidity' in weather_day.columns:
                                        summary['Avg Humidity (%)'] = weather_day['humidity'].mean()
                                    else:
                                        summary['Avg Humidity (%)'] = None

                                    if 'clouds_all' in weather_day.columns:
                                        summary['Avg Clouds (%)'] = weather_day['clouds_all'].mean()
                                    else:
                                        summary['Avg Clouds (%)'] = None

                                    if 'wind_speed' in weather_day.columns:
                                        summary['Avg Wind (m/s)'] = weather_day['wind_speed'].mean()
                                    else:
                                        summary['Avg Wind (m/s)'] = None
                                else:
                                    summary['Min Temp (°C)'] = None
                                    summary['Max Temp (°C)'] = None
                                    summary['Avg Humidity (%)'] = None
                                    summary['Avg Clouds (%)'] = None
                                    summary['Avg Wind (m/s)'] = None
                            else:
                                summary['Min Temp (°C)'] = None
                                summary['Max Temp (°C)'] = None
                                summary['Avg Humidity (%)'] = None
                                summary['Avg Clouds (%)'] = None
                                summary['Avg Wind (m/s)'] = None

                            # Calculate actual and predicted production
                            actual_prod = day_data_hist['generation_kwh'].sum()
                            summary['Actual Production (kWh)'] = actual_prod if not pd.isna(actual_prod) and actual_prod > 0 else 0.0

                            if 'ml_predicted_kwh' in day_data_hist.columns:
                                pred_prod = day_data_hist['ml_predicted_kwh'].sum()
                                summary['Predicted Production (kWh)'] = pred_prod if not pd.isna(pred_prod) else None
                            else:
                                summary['Predicted Production (kWh)'] = None

                            summary['Type'] = 'Historical'

                            weather_summary.append(summary)

                        # Create DataFrame and sort by date
                        weather_df = pd.DataFrame(weather_summary)
                        weather_df = weather_df.sort_values('Full_Date').reset_index(drop=True)

                        # Display table (without index column)
                        display_cols = ['Date', 'Type', 'Min Temp (°C)', 'Max Temp (°C)', 'Avg Humidity (%)',
                                      'Avg Clouds (%)', 'Avg Wind (m/s)', 'Actual Production (kWh)', 'Predicted Production (kWh)']

                        # Filter columns that exist
                        display_cols = [col for col in display_cols if col in weather_df.columns]

                        # Format numbers but handle None/NaN values
                        def format_value(val, fmt):
                            if pd.isna(val) or val is None:
                                return ''
                            return fmt.format(val)

                        format_dict = {}
                        if 'Min Temp (°C)' in weather_df.columns:
                            format_dict['Min Temp (°C)'] = lambda x: format_value(x, '{:.1f}')
                        if 'Max Temp (°C)' in weather_df.columns:
                            format_dict['Max Temp (°C)'] = lambda x: format_value(x, '{:.1f}')
                        if 'Avg Humidity (%)' in weather_df.columns:
                            format_dict['Avg Humidity (%)'] = lambda x: format_value(x, '{:.0f}')
                        if 'Avg Clouds (%)' in weather_df.columns:
                            format_dict['Avg Clouds (%)'] = lambda x: format_value(x, '{:.0f}')
                        if 'Avg Wind (m/s)' in weather_df.columns:
                            format_dict['Avg Wind (m/s)'] = lambda x: format_value(x, '{:.1f}')
                        if 'Actual Production (kWh)' in weather_df.columns:
                            format_dict['Actual Production (kWh)'] = lambda x: format_value(x, '{:.1f}')
                        if 'Predicted Production (kWh)' in weather_df.columns:
                            format_dict['Predicted Production (kWh)'] = lambda x: format_value(x, '{:.1f}')

                        # Apply color gradient to weather columns (red for high, yellow for low)
                        # Use full year data for min/max reference
                        styled_df = weather_df[display_cols].style.format(format_dict)

                        # Calculate yearly min/max from the full weather dataset
                        yearly_ranges = {}
                        if weather_hist is not None and not weather_hist.empty:
                            if 'temp' in weather_hist.columns:
                                yearly_ranges['Min Temp (°C)'] = (weather_hist['temp'].min(), weather_hist['temp'].max())
                                yearly_ranges['Max Temp (°C)'] = (weather_hist['temp'].min(), weather_hist['temp'].max())
                            if 'humidity' in weather_hist.columns:
                                yearly_ranges['Avg Humidity (%)'] = (weather_hist['humidity'].min(), weather_hist['humidity'].max())
                            if 'clouds_all' in weather_hist.columns:
                                yearly_ranges['Avg Clouds (%)'] = (weather_hist['clouds_all'].min(), weather_hist['clouds_all'].max())
                            if 'wind_speed' in weather_hist.columns:
                                yearly_ranges['Avg Wind (m/s)'] = (weather_hist['wind_speed'].min(), weather_hist['wind_speed'].max())

                        # Apply gradient to each weather column (yellow to red)
                        weather_cols_to_color = ['Min Temp (°C)', 'Max Temp (°C)', 'Avg Humidity (%)',
                                                 'Avg Clouds (%)', 'Avg Wind (m/s)']

                        for col in weather_cols_to_color:
                            if col in weather_df.columns and col in yearly_ranges:
                                vmin, vmax = yearly_ranges[col]
                                styled_df = styled_df.background_gradient(
                                    subset=[col],
                                    cmap='YlOrRd',  # Yellow (low) to Orange to Red (high)
                                    vmin=vmin,
                                    vmax=vmax
                                )

                        # Add thick black border between Historical and Forecast
                        def add_separator(row):
                            # Find first Forecast row
                            if row.name > 0 and weather_df.iloc[row.name]['Type'] in ('Forecast', 'Replay') and weather_df.iloc[row.name - 1]['Type'] == 'Historical':
                                return ['border-top: 3px solid black'] * len(row)
                            return [''] * len(row)

                        styled_df = styled_df.apply(add_separator, axis=1)

                        st.dataframe(
                            styled_df,
                            width='stretch',
                            height=400,
                            hide_index=True
                        )

                        with st.expander("Advanced: forecast feature diagnostics", expanded=False):
                            st.markdown("**Key features used in predictions:**")
                            diag_df = visualization.create_feature_diagnostics_table(forecast_with_features)
                            if not diag_df.empty:
                                st.dataframe(diag_df, width='stretch')

                            st.markdown("**Sample forecast points (first 5 daytime hours):**")
                            sample_cols = [
                                'predicted_kwh', 'clouds_all', 'cloud_impact',
                                'production_lag_24h', 'clearsky_expected_kwh', 'solar_elevation'
                            ]
                            available_cols = [col for col in sample_cols if col in forecast_with_features.columns]
                            st.dataframe(forecast_with_features[available_cols].head(5).round(2), width='stretch')
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: gray;'>
        <p>Solar Forecasting & Anomaly Detection Dashboard | {selected_plant}</p>
        <p>Last Updated: {df.index.max().strftime("%Y-%m-%d %H:%M")} | Capacity: {plant_info['capacity_kwp']} kWp</p>
    </div>
    """, unsafe_allow_html=True)


def handle_data_upload():
    """
    Check if the local generation data exists; otherwise show the upload setup flow.

    Uploaded data is for local use only. Use reviewed data before sharing.
    """
    data_file_path = Path(DATA_PATHS['generation_5m'])

    if not data_file_path.exists():
        st.markdown('<p class="main-header">☀️ Solar Forecasting & Anomaly Detection Dashboard - Data Setup</p>', unsafe_allow_html=True)
        st.markdown("---")

        st.warning("⚠️ **Generation data file not found.** Upload a local CSV to continue.")

        st.markdown("""
        ### 📁 Upload required file
        Please upload `inverter_five_minutes_generation_logs.csv` to start using the dashboard.

        The file is saved to the local `data/` path. Use reviewed data for shared copies.
        """)

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload the 5-minute generation logs CSV file",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            # Reset file pointer to beginning (important for re-reading)
            uploaded_file.seek(0)

            # Show file info
            df_preview = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! ({len(df_preview):,} rows)")

            # Reset file pointer again for saving later
            uploaded_file.seek(0)

            with st.expander("📊 Preview uploaded data"):
                st.dataframe(df_preview.head(10))
                st.write(f"**Columns:** {', '.join(df_preview.columns.tolist())}")

            st.markdown("---")
            st.markdown("### 🤖 Choose Training Option")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="metric-card">
                <h4>⚡ Fast Training</h4>
                <p><strong>Ridge Regression Only</strong></p>
                <ul>
                <li>Duration: ~1 minute</li>
                <li>Trains Ridge for all plants</li>
                <li>Keeps stored comparison data</li>
                <li>Recommended for quick updates</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🚀 Upload & Retrain Ridge-only (<1 min)", type="primary", use_container_width=True):
                    with st.spinner('Starting training...'):
                        train_with_upload(uploaded_file, df_preview, ridge_only=True)

            with col2:
                st.markdown("""
                <div class="metric-card">
                <h4>🔬 Complete Training</h4>
                <p><strong>All Models</strong></p>
                <ul>
                <li>Duration: ~10 minutes</li>
                <li>Trains Ridge, RandomForest, XGBoost, etc.</li>
                <li>Full model comparison</li>
                <li>Recommended for initial setup</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)

                if st.button("🔬 Upload & Retrain Multi-Model (>10 min)", use_container_width=True):
                    with st.spinner('Starting training...'):
                        train_with_upload(uploaded_file, df_preview, ridge_only=False)

        else:
            st.info("👆 Please upload the CSV file above to continue")

        st.stop()  # Don't proceed to main app until file is uploaded

    return True


def train_with_upload(uploaded_file, df, ridge_only=True):
    """Save an uploaded local CSV and trigger model training."""

    st.info("🔄 Starting upload and training process...")

    data_file_path = Path(DATA_PATHS['generation_5m'])

    # Create directory if needed
    data_file_path.parent.mkdir(parents=True, exist_ok=True)
    st.info(f"📁 Directory ready: {data_file_path.parent}")

    # Save the uploaded file
    with st.spinner('💾 Saving data file...'):
        try:
            df.to_csv(data_file_path, index=False)
            file_size = data_file_path.stat().st_size / (1024*1024)
            st.success(f"✅ Data saved to {data_file_path} ({file_size:.1f} MB)")
        except Exception as e:
            st.error(f"❌ Failed to save file: {e}")
            return

    # Train models
    mode = "Ridge-only" if ridge_only else "Multi-Model"
    st.markdown(f"### 🤖 Starting {mode} Training...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Import training modules
        from train_all_models import train_plant_model, PLANTS

        training_results = []
        plant_count = len(PLANTS)

        for idx, (plant_name, plant_info) in enumerate(PLANTS.items()):
            status_text.text(f"Training {plant_name}... ({idx+1}/{plant_count})")
            progress_bar.progress((idx) / plant_count)

            result = train_plant_model(plant_name, plant_info, ridge_only)
            training_results.append(result)

        progress_bar.progress(1.0)
        status_text.text("Training complete!")

        # Show results
        success_count = sum(1 for r in training_results if r['success'])

        if success_count == plant_count:
            st.success(f"🎉 All {plant_count} plants trained successfully!")
        else:
            st.warning(f"⚠️ {success_count}/{plant_count} plants trained successfully")

        # Show detailed results
        with st.expander("📊 Training Results Details"):
            for result in training_results:
                if result['success']:
                    st.write(f"✅ **{result['plant']}** - {result['best_model']} | "
                            f"R²={result['r2']:.4f} | MAE={result['mae']:.2f} kWh")
                else:
                    st.write(f"❌ **{result['plant']}** - ERROR: {result['error']}")

        st.success("✅ Setup complete! Reloading dashboard...")
        st.balloons()

        # Wait a moment then rerun
        import time
        time.sleep(2)
        st.rerun()

    except Exception as e:
        st.error(f"❌ Training failed: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def cleanup_uploaded_file():
    """Initialize session state without deleting the bundled/local data file."""
    if 'session_initialized' not in st.session_state:
        st.session_state.session_initialized = True


if __name__ == "__main__":
    # Initialize per-session state without touching local data files
    cleanup_uploaded_file()

    # Check and handle data upload first
    handle_data_upload()

    # Then run main app
    main()
