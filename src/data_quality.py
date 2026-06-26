"""
Data Quality module for Solar Production Prediction.
Handles data audit, completeness analysis, and quality metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from pvlib.location import Location

from .config import PLANT_CONFIG


def audit_data_quality(df_5min: pd.DataFrame, reference_now: pd.Timestamp = None) -> Optional[Dict]:
    """
    Comprehensive data quality audit for inverter data.

    Detects:
    - Coverage gaps (data stops before current date)
    - Timestamp gaps (missing 5-minute intervals)
    - Duplicate timestamps
    - Negative generation values
    - Suspicious nighttime production
    - Data freshness issues

    Args:
        df_5min: DataFrame with 5-minute data (indexed by generation_date)

    Returns:
        dict with quality metrics and issues, or None if input is empty
    """
    if df_5min is None or df_5min.empty:
        return None

    audit_results = {}
    issues = []

    # Basic stats
    start_date = df_5min.index.min()
    end_date = df_5min.index.max()
    total_days = (end_date - start_date).days + 1

    # Current time. The app can pass a historical snapshot date when working
    # with fixed local data instead of a live data feed.
    now = reference_now if reference_now is not None else pd.Timestamp.now(tz='Asia/Dhaka')
    if now.tzinfo is None:
        now = now.tz_localize('Asia/Dhaka')
    else:
        now = now.tz_convert('Asia/Dhaka')
    audit_results['audit_date'] = now
    audit_results['start_date'] = start_date
    audit_results['end_date'] = end_date
    audit_results['total_days'] = total_days

    # === Use complete interval method for accurate completeness ===
    # Create complete 5-minute interval index
    complete_intervals = pd.DataFrame(
        index=pd.date_range(start=start_date, end=end_date, freq='5min', tz=df_5min.index.tz)
    )

    # Merge with actual data
    merged = complete_intervals.merge(
        df_5min[['generation_kwh']],
        how='left',
        left_index=True,
        right_index=True
    )

    # Calculate completeness metrics
    total_expected = len(complete_intervals)
    total_collected = merged['generation_kwh'].notna().sum()
    total_missing = merged['generation_kwh'].isna().sum()
    completeness_pct = (total_collected / total_expected * 100) if total_expected > 0 else 0

    audit_results['total_expected'] = total_expected
    audit_results['total_records'] = total_collected
    audit_results['total_missing'] = total_missing
    audit_results['completeness_pct'] = completeness_pct

    # Issue 1: Data coverage gap (last data vs current date)
    days_since_last_data = (now - end_date).days
    if days_since_last_data > 1:
        issues.append({
            'type': 'Coverage Gap',
            'severity': 'HIGH',
            'description': f'Data stops on {end_date.date()}, {days_since_last_data} days ago',
            'count': days_since_last_data
        })

    # Issue 2: Timestamp gaps (more than 5 minutes between consecutive records)
    df_sorted = df_5min.sort_index()
    time_diffs = df_sorted.index.to_series().diff()
    expected_interval = pd.Timedelta(minutes=5)
    gaps = time_diffs[time_diffs > expected_interval]

    # Count significant gaps (> 1 hour)
    significant_gaps = gaps[gaps > pd.Timedelta(hours=1)] if len(gaps) > 0 else pd.Series([], dtype='timedelta64[ns]')

    if len(significant_gaps) > 0:
        issues.append({
            'type': 'Timestamp Gaps',
            'severity': 'MEDIUM',
            'description': f'{len(significant_gaps)} gaps > 1 hour found in data sequence',
            'count': len(significant_gaps),
            'largest_gap': gaps.max()
        })

    audit_results['timestamp_gaps'] = len(significant_gaps)
    audit_results['largest_gap_hours'] = gaps.max().total_seconds() / 3600 if len(gaps) > 0 else 0

    # Issue 3: Negative generation values
    negative_values = df_5min[df_5min['generation_kwh'] < 0]
    if len(negative_values) > 0:
        issues.append({
            'type': 'Negative Values',
            'severity': 'HIGH',
            'description': f'{len(negative_values)} records with negative generation',
            'count': len(negative_values)
        })

    audit_results['negative_values'] = len(negative_values)

    # Issue 4: Suspicious nighttime production (using solar elevation)
    try:
        location = Location(
            latitude=PLANT_CONFIG['latitude'],
            longitude=PLANT_CONFIG['longitude'],
            tz=PLANT_CONFIG['timezone']
        )

        # Calculate solar position for all timestamps
        solpos = location.get_solarposition(df_5min.index)
        df_with_solar = df_5min.copy()
        df_with_solar['elevation'] = solpos['elevation'].values

        # Night = solar elevation ≤ 0° (sun below horizon)
        nighttime = df_with_solar[df_with_solar['elevation'] <= 0]
        nighttime_production = nighttime[nighttime['generation_kwh'] > 0.1]  # > 100 Wh

        if len(nighttime_production) > 0:
            issues.append({
                'type': 'Nighttime Production',
                'severity': 'MEDIUM',
                'description': f'{len(nighttime_production)} records with production when sun is below horizon (solar elevation ≤ 0°)',
                'count': len(nighttime_production)
            })
    except Exception as e:
        # Fallback to simple hour-based check if solar calculation fails
        print(f"⚠️ Could not calculate solar elevation: {e}. Using fallback hour-based check.")
        df_with_hour = df_5min.copy()
        df_with_hour['hour'] = df_with_hour.index.hour
        nighttime = df_with_hour[(df_with_hour['hour'] >= 20) | (df_with_hour['hour'] < 5)]
        nighttime_production = nighttime[nighttime['generation_kwh'] > 0.1]

        if len(nighttime_production) > 0:
            issues.append({
                'type': 'Nighttime Production',
                'severity': 'LOW',
                'description': f'{len(nighttime_production)} records with production during night hours (20:00-5:00) - approximate check',
                'count': len(nighttime_production)
            })

    audit_results['nighttime_production'] = len(nighttime_production)

    # Issue 5: Data freshness
    if days_since_last_data == 0:
        freshness = 'CURRENT'
    elif days_since_last_data <= 7:
        freshness = 'RECENT'
    elif days_since_last_data <= 30:
        freshness = 'STALE'
    else:
        freshness = 'VERY STALE'

    audit_results['data_freshness'] = freshness
    audit_results['issues'] = issues
    audit_results['issue_count'] = len(issues)

    return audit_results


def analyze_data_completeness(df_5min: pd.DataFrame) -> Optional[Tuple]:
    """
    Analyze data completeness from 5-minute raw data using complete interval method.

    Method: Creates a complete 5-minute interval index from start to end date,
    merges with actual data, and counts NaN values to identify missing records.
    This is the most accurate method and doesn't require guessing expected records.

    Args:
        df_5min: DataFrame with 5-minute data (indexed by generation_date)

    Returns:
        tuple: (daily_stats, total_intervals, start_date, end_date,
                total_expected, total_collected, total_missing)
        or None if input is empty
    """
    if df_5min is None or df_5min.empty:
        return None

    # ===== STEP 1: Create complete 5-minute interval index =====
    # Normalize to full days (00:00 to 23:55) to ensure each day has exactly 288 intervals
    start_dt = df_5min.index.min().normalize()  # Start at 00:00 of first day
    end_dt = df_5min.index.max().normalize() + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)  # End at 23:55 of last day

    # Create complete interval DataFrame
    complete_intervals = pd.DataFrame(
        index=pd.date_range(start=start_dt, end=end_dt, freq='5min', tz=df_5min.index.tz)
    )

    # ===== STEP 2: Merge with actual data =====
    # Left join to keep all intervals, mark actual data
    merged = complete_intervals.merge(
        df_5min[['generation_kwh']],
        how='left',
        left_index=True,
        right_index=True
    )

    # ===== STEP 3: Calculate overall statistics =====
    total_expected = len(complete_intervals)
    total_collected = merged['generation_kwh'].notna().sum()
    total_missing = merged['generation_kwh'].isna().sum()

    # ===== STEP 4: Calculate daily statistics =====
    # Add date column for grouping
    merged['date'] = merged.index.date

    # Create a complete date range to ensure ALL days are included
    all_dates = pd.date_range(start=start_dt.date(), end=end_dt.date(), freq='D')

    daily_stats = merged.groupby('date').agg(
        Records_Collected=('generation_kwh', lambda x: x.notna().sum()),
        Total_Production=('generation_kwh', 'sum'),
        Expected_Records=('generation_kwh', 'size')  # Use 'size' to count ALL rows including NaN
    ).reset_index()

    # Ensure all dates are present (even those with no data at all)
    daily_stats['Date'] = pd.to_datetime(daily_stats['date'])
    all_dates_df = pd.DataFrame({'Date': all_dates})
    daily_stats = all_dates_df.merge(daily_stats, left_on='Date', right_on='Date', how='left')

    # Fill missing values for days with no data
    daily_stats['Records_Collected'] = daily_stats['Records_Collected'].fillna(0).astype(int)
    daily_stats['Total_Production'] = daily_stats['Total_Production'].fillna(0)
    daily_stats['Expected_Records'] = daily_stats['Expected_Records'].fillna(288).astype(int)

    # Keep only necessary columns
    daily_stats = daily_stats[['Date', 'Records_Collected', 'Total_Production', 'Expected_Records']]

    # Calculate missing records and completeness
    daily_stats['Missing_Records'] = daily_stats['Expected_Records'] - daily_stats['Records_Collected']
    daily_stats['Completeness_Pct'] = (daily_stats['Records_Collected'] / daily_stats['Expected_Records'] * 100).round(1)

    # ===== STEP 5: Status classification =====
    # Initialize with default status
    daily_stats['Status'] = 'Complete'

    # Override with more specific status (order matters: most specific last)
    daily_stats.loc[daily_stats['Missing_Records'] > 0, 'Status'] = 'Partial Data'
    daily_stats.loc[daily_stats['Total_Production'].isna() | (daily_stats['Total_Production'] == 0), 'Status'] = 'No Production Data'

    # Get date range
    start_date = start_dt.date()
    end_date = end_dt.date()

    return daily_stats, len(complete_intervals), start_date, end_date, total_expected, total_collected, total_missing
