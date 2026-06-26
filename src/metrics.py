"""
Metrics module for Solar Production Prediction.
Handles calculation of performance metrics and KPIs.
"""

import pandas as pd
from typing import Dict, Optional
from datetime import date


def calculate_daily_metrics(
    df: pd.DataFrame,
    target_date: date,
    alert_threshold_pct: float
) -> Optional[Dict]:
    """
    Calculate daily performance metrics for a specific date.

    Args:
        df: DataFrame with hourly production data (indexed by timestamp)
        target_date: Date to calculate metrics for
        alert_threshold_pct: Threshold percentage for anomaly detection

    Returns:
        Dictionary with daily metrics or None if no data available
    """
    daily_data = df[df.index.date == target_date]

    if len(daily_data) == 0:
        return None

    metrics = {
        'actual_total': daily_data['generation_kwh'].sum(),
        'predicted_total': daily_data['ml_predicted_kwh'].sum() if 'ml_predicted_kwh' in daily_data.columns else 0,
        'clearsky_total': daily_data['clearsky_expected_kwh'].sum() if 'clearsky_expected_kwh' in daily_data.columns else 0,
        'actual_peak': daily_data['generation_kwh'].max(),
        'predicted_peak': daily_data['ml_predicted_kwh'].max() if 'ml_predicted_kwh' in daily_data.columns else 0,
        'num_hours': len(daily_data),
    }

    # Calculate performance ratio
    if metrics['predicted_total'] > 0:
        metrics['performance_ratio'] = (metrics['actual_total'] / metrics['predicted_total']) * 100
    else:
        metrics['performance_ratio'] = 0

    # Check for anomalies
    metrics['has_anomaly'] = metrics['performance_ratio'] < (100 - alert_threshold_pct)

    return metrics


def calculate_capacity_factor(df: pd.DataFrame, capacity_kwp: float) -> float:
    """
    Calculate capacity factor (actual production vs maximum possible production).

    Capacity Factor = (Actual Production / Maximum Possible Production) × 100%

    Args:
        df: DataFrame with hourly production data
        capacity_kwp: Plant capacity in kWp

    Returns:
        Capacity factor as a percentage
    """
    total_actual = df['generation_kwh'].sum()
    hours = len(df)
    max_possible = capacity_kwp * hours

    if max_possible > 0:
        capacity_factor = (total_actual / max_possible) * 100
    else:
        capacity_factor = 0

    return capacity_factor


def calculate_monthly_metrics(df: pd.DataFrame, capacity_kwp: float) -> pd.DataFrame:
    """
    Calculate monthly performance metrics.

    Args:
        df: DataFrame with hourly production data (indexed by timestamp)
        capacity_kwp: Plant capacity in kWp

    Returns:
        DataFrame with monthly aggregated metrics
    """
    # Aggregate by month
    monthly = df.groupby(df.index.to_period('M')).agg({
        'generation_kwh': ['sum', 'mean', 'max', 'count']
    })

    monthly.columns = ['total_kwh', 'avg_hourly_kwh', 'peak_hour_kwh', 'num_hours']
    monthly['month'] = monthly.index.to_timestamp()

    # Calculate capacity factor for each month
    monthly['capacity_factor'] = (monthly['total_kwh'] / (capacity_kwp * monthly['num_hours'])) * 100

    # Calculate predicted vs actual if available
    if 'ml_predicted_kwh' in df.columns:
        monthly_predicted = df.groupby(df.index.to_period('M'))['ml_predicted_kwh'].sum()
        monthly['predicted_kwh'] = monthly_predicted
        monthly['performance_ratio'] = (monthly['total_kwh'] / monthly['predicted_kwh']) * 100

    return monthly.reset_index(drop=True)


def calculate_performance_ratio(
    actual_kwh: float,
    predicted_kwh: float
) -> float:
    """
    Calculate performance ratio (actual vs predicted).

    Args:
        actual_kwh: Actual production in kWh
        predicted_kwh: Predicted production in kWh

    Returns:
        Performance ratio as a percentage
    """
    if predicted_kwh > 0:
        return (actual_kwh / predicted_kwh) * 100
    else:
        return 0.0


def identify_underperforming_days(
    df: pd.DataFrame,
    threshold_pct: float = 80
) -> pd.DataFrame:
    """
    Identify days where actual production is significantly below prediction.

    Args:
        df: DataFrame with hourly production data (indexed by timestamp)
        threshold_pct: Threshold percentage for underperformance (default: 80%)

    Returns:
        DataFrame with underperforming days and their metrics
    """
    # Aggregate by day
    daily = df.groupby(df.index.date).agg({
        'generation_kwh': 'sum',
        'ml_predicted_kwh': 'sum' if 'ml_predicted_kwh' in df.columns else lambda x: 0
    })

    if 'ml_predicted_kwh' not in df.columns:
        return pd.DataFrame()

    daily.columns = ['actual_kwh', 'predicted_kwh']
    daily['performance_ratio'] = (daily['actual_kwh'] / daily['predicted_kwh']) * 100

    # Filter underperforming days
    underperforming = daily[daily['performance_ratio'] < threshold_pct].copy()
    underperforming['gap_kwh'] = underperforming['predicted_kwh'] - underperforming['actual_kwh']

    return underperforming.reset_index().rename(columns={'index': 'date'})
