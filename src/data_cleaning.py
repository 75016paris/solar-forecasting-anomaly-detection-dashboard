"""
Data cleaning helpers for the name-redacted solar forecasting data.

The cleaned outputs are generated from local CSVs and keep quality columns
explicit before model training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pvlib.location import Location

from .config import DATA_PATHS, ML_CONFIG, PLANT_CONFIG


SELECTED_PLANTS = {
    'Plant A': {'plant_id': 900001, 'capacity_kwp': 269.28},
    'Plant B': {'plant_id': 900002, 'capacity_kwp': 522.72},
    'Plant C': {'plant_id': 900003, 'capacity_kwp': 227.04},
    'Plant D': {'plant_id': 900004, 'capacity_kwp': 525.36},
    'Plant E': {'plant_id': 900005, 'capacity_kwp': 285.12},
}

SELECTED_PLANT_IDS = {info['plant_id'] for info in SELECTED_PLANTS.values()}
PLANT_NAME_BY_ID = {info['plant_id']: name for name, info in SELECTED_PLANTS.items()}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA_DIRS = [PROJECT_ROOT / 'data']
SOURCE_WEATHER_DIRS = [PROJECT_ROOT / 'open_data']


def _find_source_file(filename: str, directories: Iterable[Path]) -> Path:
    """Return the first matching source file from the project data dirs."""
    for directory in directories:
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find source file: {filename}")


def _parse_utc_to_local(series: pd.Series) -> pd.Series:
    """Parse a timestamp series as UTC and convert to the configured local timezone."""
    return pd.to_datetime(series, utc=True, errors='coerce').dt.tz_convert(PLANT_CONFIG['timezone'])


def _numeric_amount(series: pd.Series) -> pd.Series:
    """Parse numeric values that may contain comma separators."""
    return pd.to_numeric(series.astype(str).str.replace(',', '', regex=False), errors='coerce')


def clean_plants() -> pd.DataFrame:
    """Clean and filter plant metadata to the five plants used by the app."""
    path = _find_source_file('inverter_plants.csv', SOURCE_DATA_DIRS)
    plants = pd.read_csv(path)
    plants = plants[plants['plant_id'].isin(SELECTED_PLANT_IDS)].copy()
    plants['plant_name'] = plants['plant_id'].map(PLANT_NAME_BY_ID).fillna(plants['plant_name'])
    plants['plant_address'] = plants['plant_address'].fillna('Solar Region')
    plants['selected_for_dashboard'] = True

    sort_order = {name: idx for idx, name in enumerate(SELECTED_PLANTS)}
    plants['_sort'] = plants['plant_name'].map(sort_order)
    plants = plants.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)
    return plants


def clean_generation_5min(plants_clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean raw five-minute inverter generation rows for selected plants only."""
    path = _find_source_file('inverter_five_minutes_generation_logs.csv', SOURCE_DATA_DIRS)
    generation = pd.read_csv(path)
    generation = generation[generation['plant_id'].isin(SELECTED_PLANT_IDS)].copy()
    generation['generation_date'] = _parse_utc_to_local(generation['generation_date'])
    generation['generation_amount_wh'] = _numeric_amount(generation['generation_amount'])
    generation = generation.dropna(subset=['generation_date', 'generation_amount_wh'])

    # Negative inverter interval generation is not physically meaningful for this target.
    generation['is_negative_generation'] = generation['generation_amount_wh'] < 0
    generation.loc[generation['is_negative_generation'], 'generation_amount_wh'] = np.nan
    generation = generation.dropna(subset=['generation_amount_wh'])

    generation['plant_name'] = generation['plant_id'].map(PLANT_NAME_BY_ID)
    generation['generation_kwh'] = generation['generation_amount_wh'] / 1000

    before = len(generation)
    generation = (
        generation.sort_values(['plant_id', 'generation_date'])
        .drop_duplicates(subset=['plant_id', 'generation_date'], keep='last')
    )
    generation['duplicate_rows_removed_for_same_plant_timestamp'] = 0
    generation.attrs['duplicates_removed'] = before - len(generation)

    cols = [
        'plant_id',
        'plant_name',
        'generation_date',
        'generation_amount_wh',
        'generation_kwh',
        'is_negative_generation',
        'duplicate_rows_removed_for_same_plant_timestamp',
    ]
    return generation[cols].reset_index(drop=True)


def build_hourly_model_targets(generation_5min_clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned five-minute rows into hourly modeling targets."""
    expected_records = 12
    min_completeness = ML_CONFIG.get('min_hourly_completeness_pct', 80)
    hourly_parts = []

    for plant_name, plant_info in SELECTED_PLANTS.items():
        plant_id = plant_info['plant_id']
        plant_df = generation_5min_clean[generation_5min_clean['plant_id'] == plant_id].copy()
        if plant_df.empty:
            continue

        plant_df = plant_df.sort_values('generation_date').set_index('generation_date')
        hourly = plant_df.resample('1h')['generation_kwh'].agg([
            ('generation_kwh_raw', lambda x: x.sum(min_count=1)),
            ('records_collected', 'count'),
        ])
        hourly = hourly.reset_index()
        hourly['plant_id'] = plant_id
        hourly['plant_name'] = plant_name
        hourly['expected_records'] = expected_records
        hourly['missing_records'] = expected_records - hourly['records_collected']
        hourly['data_completeness_pct'] = hourly['records_collected'] / expected_records * 100
        hourly['data_available'] = hourly['data_completeness_pct'] >= min_completeness
        hourly['generation_kwh'] = hourly['generation_kwh_raw'].where(hourly['data_available'])
        hourly['target_source'] = '5min_inverter_cleaned'
        hourly['target_quality_flag'] = np.where(hourly['data_available'], 'usable_5min_hour', 'incomplete_hour_excluded')
        hourly_parts.append(hourly)

    if not hourly_parts:
        return pd.DataFrame()

    cols = [
        'plant_id',
        'plant_name',
        'generation_date',
        'generation_kwh',
        'generation_kwh_raw',
        'records_collected',
        'expected_records',
        'missing_records',
        'data_completeness_pct',
        'data_available',
        'target_source',
        'target_quality_flag',
    ]
    return pd.concat(hourly_parts, ignore_index=True)[cols].sort_values(['plant_name', 'generation_date'])


def _daily_generation_log(filename: str, key: str, amount_divisor: float, date_shift_days: int = 0) -> pd.DataFrame:
    """Return daily kWh totals from an inverter generation-log source."""
    try:
        path = _find_source_file(filename, SOURCE_DATA_DIRS)
    except FileNotFoundError:
        return pd.DataFrame()

    df = pd.read_csv(path, usecols=['plant_id', 'generation_date', 'generation_amount'])
    df = df[df['plant_id'].isin(SELECTED_PLANT_IDS)].copy()
    df['timestamp'] = _parse_utc_to_local(df['generation_date'])
    df['date'] = df['timestamp'].dt.date
    if date_shift_days:
        df['date'] = (pd.to_datetime(df['date']) + pd.Timedelta(days=date_shift_days)).dt.date
    df['generation_amount'] = _numeric_amount(df['generation_amount'])
    df = df.dropna(subset=['timestamp', 'generation_amount'])
    df['plant_name'] = df['plant_id'].map(PLANT_NAME_BY_ID)

    daily_total = df.groupby(['plant_id', 'plant_name', 'date'])['generation_amount'].sum(min_count=1) / amount_divisor
    daily_records = df.groupby(['plant_id', 'plant_name', 'date']).size()
    return pd.concat([
        daily_total.rename(key),
        daily_records.rename(f'{key}_records'),
    ], axis=1)


def _daily_meter_reading(filename: str, key: str) -> pd.DataFrame:
    """Return daily kWh totals from cumulative meter readings."""
    try:
        path = _find_source_file(filename, SOURCE_DATA_DIRS)
    except FileNotFoundError:
        return pd.DataFrame()

    df = pd.read_csv(path)
    required_cols = {'plant_id', 'meter_id', 'meter_reading', 'date'}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    df = df[df['plant_id'].isin(SELECTED_PLANT_IDS)].copy()
    df['timestamp'] = _parse_utc_to_local(df['date'])
    df['date'] = df['timestamp'].dt.date
    df['meter_reading'] = pd.to_numeric(df['meter_reading'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'meter_reading'])
    df['plant_name'] = df['plant_id'].map(PLANT_NAME_BY_ID)

    parts = []
    group_cols = ['plant_id', 'plant_name', 'meter_id']
    for _, meter_df in df.groupby(group_cols):
        meter_df = meter_df.sort_values('timestamp').copy()
        meter_df[key] = meter_df['meter_reading'].diff() / 1000
        meter_df.loc[meter_df[key] < 0, key] = np.nan
        parts.append(meter_df[['plant_id', 'plant_name', 'date', key]])

    if not parts:
        return pd.DataFrame()

    rows = pd.concat(parts, ignore_index=True)
    daily_total = rows.groupby(['plant_id', 'plant_name', 'date'])[key].sum(min_count=1)
    daily_records = rows.groupby(['plant_id', 'plant_name', 'date'])[key].count()
    return pd.concat([
        daily_total.rename(key),
        daily_records.rename(f'{key}_records'),
    ], axis=1)


def build_daily_source_agreement() -> pd.DataFrame:
    """Build daily agreement diagnostics across generation and meter sources."""
    daily_sources = [
        _daily_generation_log('inverter_five_minutes_generation_logs.csv', 'five_min_kwh', 1000),
        _daily_generation_log('inverter_hourly_generation_logs.csv', 'hourly_inverter_kwh', 1000),
        _daily_generation_log('inverter_daily_generation_logs.csv', 'daily_inverter_kwh', 1, date_shift_days=-1),
        _daily_meter_reading('plants_meter_data_logs_hourly.csv', 'hourly_meter_kwh'),
        _daily_meter_reading('plants_billing_meter_logs.csv', 'billing_meter_kwh'),
    ]
    daily_sources = [df for df in daily_sources if not df.empty]
    if not daily_sources:
        return pd.DataFrame()

    agreement = pd.concat(daily_sources, axis=1).reset_index().sort_values(['plant_name', 'date'])
    baseline = agreement['five_min_kwh'] if 'five_min_kwh' in agreement.columns else pd.Series(np.nan, index=agreement.index)

    for col in ['hourly_inverter_kwh', 'daily_inverter_kwh', 'hourly_meter_kwh', 'billing_meter_kwh']:
        if col not in agreement.columns:
            continue
        agreement[f'{col}_minus_5min_kwh'] = agreement[col] - baseline
        agreement[f'{col}_ratio_to_5min'] = np.where(baseline.abs() > 0.01, agreement[col] / baseline, np.nan)

    ratio_cols = [col for col in agreement.columns if col.endswith('_ratio_to_5min')]
    if ratio_cols:
        ratio_distance = (agreement[ratio_cols] - 1).abs().min(axis=1, skipna=True)
        agreement['source_agreement_flag'] = 'no_reference_overlap'
        agreement.loc[ratio_distance <= 0.05, 'source_agreement_flag'] = 'close_agreement'
        agreement.loc[(ratio_distance > 0.05) & (ratio_distance <= 0.20), 'source_agreement_flag'] = 'moderate_difference'
        agreement.loc[ratio_distance > 0.20, 'source_agreement_flag'] = 'large_difference'
    else:
        agreement['source_agreement_flag'] = 'no_reference_overlap'

    return agreement


def clean_weather_hourly() -> pd.DataFrame:
    """Clean OpenWeather history into hourly weather features."""
    path = _find_source_file('gazipur_weather.csv', SOURCE_WEATHER_DIRS)
    weather = pd.read_csv(path)
    weather['generation_date'] = _parse_utc_to_local(weather['dt_iso'].str.replace(' UTC', '', regex=False))

    candidate_cols = [
        'generation_date',
        'temp',
        'visibility',
        'rain_1h',
        'clouds_all',
        'pressure',
        'humidity',
        'wind_speed',
        'feels_like',
        'wind_deg',
    ]
    weather_cols = [col for col in candidate_cols if col in weather.columns]
    weather = weather[weather_cols].copy()

    numeric_cols = [col for col in weather.columns if col != 'generation_date']
    for col in numeric_cols:
        weather[col] = pd.to_numeric(weather[col], errors='coerce')

    if 'rain_1h' in weather.columns:
        weather['rain_1h'] = weather['rain_1h'].fillna(0)

    weather = (
        weather.groupby('generation_date')[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values('generation_date')
    )
    return weather


def build_data_quality_summary(hourly_targets: pd.DataFrame, daily_agreement: pd.DataFrame) -> pd.DataFrame:
    """Summarize cleaned target completeness and reference-source agreement by plant."""
    if hourly_targets.empty:
        return pd.DataFrame()

    location = Location(
        latitude=PLANT_CONFIG['latitude'],
        longitude=PLANT_CONFIG['longitude'],
        tz=PLANT_CONFIG['timezone'],
    )

    rows = []
    for plant_name, plant_info in SELECTED_PLANTS.items():
        plant_hours = hourly_targets[hourly_targets['plant_name'] == plant_name].copy()
        if plant_hours.empty:
            continue

        times = pd.DatetimeIndex(plant_hours['generation_date'])
        solpos = location.get_solarposition(times)
        daylight_mask = solpos['elevation'].values > ML_CONFIG.get('min_sun_elevation', 5)
        daylight = plant_hours[daylight_mask]

        total_hours = len(plant_hours)
        usable_hours = int(plant_hours['data_available'].sum())
        daylight_hours = len(daylight)
        usable_daylight_hours = int(daylight['data_available'].sum()) if daylight_hours else 0

        plant_daily = daily_agreement[daily_agreement['plant_name'] == plant_name] if not daily_agreement.empty else pd.DataFrame()
        billing_ratio = np.nan
        billing_corr = np.nan
        if not plant_daily.empty and {'five_min_kwh', 'billing_meter_kwh'}.issubset(plant_daily.columns):
            comparison = plant_daily[['five_min_kwh', 'billing_meter_kwh']].dropna()
            if not comparison.empty:
                billing_total = comparison['billing_meter_kwh'].sum()
                billing_ratio = comparison['five_min_kwh'].sum() / billing_total if billing_total else np.nan
                billing_corr = comparison['five_min_kwh'].corr(comparison['billing_meter_kwh']) if len(comparison) > 1 else np.nan

        warnings = []
        daylight_availability_pct = usable_daylight_hours / daylight_hours * 100 if daylight_hours else np.nan
        if pd.notna(daylight_availability_pct) and daylight_availability_pct < 60:
            warnings.append('Low usable daylight coverage')
        if plant_name == 'Plant E':
            warnings.append('Known source-data issue')
        if pd.notna(billing_ratio) and (billing_ratio < 0.8 or billing_ratio > 1.2):
            warnings.append('Billing-meter total differs from 5-minute inverter total')

        rows.append({
            'plant_id': plant_info['plant_id'],
            'plant_name': plant_name,
            'total_hourly_targets': total_hours,
            'usable_hourly_targets': usable_hours,
            'usable_hourly_targets_pct': usable_hours / total_hours * 100 if total_hours else np.nan,
            'daylight_modeling_hours': daylight_hours,
            'usable_daylight_modeling_hours': usable_daylight_hours,
            'usable_daylight_modeling_hours_pct': daylight_availability_pct,
            'five_min_records_collected': int(plant_hours['records_collected'].sum()),
            'five_min_records_expected': int(plant_hours['expected_records'].sum()),
            'five_min_record_completeness_pct': (
                plant_hours['records_collected'].sum() / plant_hours['expected_records'].sum() * 100
            ) if plant_hours['expected_records'].sum() else np.nan,
            'billing_meter_to_5min_overlap_corr': billing_corr,
            'five_min_total_div_billing_meter_total': billing_ratio,
            'warnings': '; '.join(warnings) if warnings else 'None',
        })

    return pd.DataFrame(rows)


def build_clean_dataset(output_dir: str | Path = 'clean') -> dict[str, pd.DataFrame]:
    """Build all cleaned CSV outputs and return them as dataframes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plants = clean_plants()
    generation_5min = clean_generation_5min(plants)
    hourly_targets = build_hourly_model_targets(generation_5min)
    daily_agreement = build_daily_source_agreement()
    weather_hourly = clean_weather_hourly()
    quality_summary = build_data_quality_summary(hourly_targets, daily_agreement)

    outputs = {
        'plants_clean': plants,
        'generation_5min_clean': generation_5min,
        'hourly_model_targets': hourly_targets,
        'daily_source_agreement': daily_agreement,
        'weather_hourly_clean': weather_hourly,
        'data_quality_summary': quality_summary,
    }

    for name, df in outputs.items():
        df.to_csv(output_dir / f'{name}.csv', index=False)

    return outputs
