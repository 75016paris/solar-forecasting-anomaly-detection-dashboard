"""
Feature engineering module for Solar Production Prediction.
Creates all features needed for ML models.

IMPORTANT: This uses the OPTIMIZED feature set (24 features).
After extensive testing, adding more weather features (humidity, wind, pressure)
DEGRADED performance. See notebook for full analysis.
"""

import pandas as pd
import numpy as np
import pvlib
from pvlib.location import Location

from .config import PLANT_CONFIG, ML_CONFIG


def add_solar_position(df: pd.DataFrame, capacity_kwp: float = None) -> pd.DataFrame:
    """
    Add solar position features (elevation, azimuth, GHI).

    Args:
        df: DataFrame with 'generation_date' column
        capacity_kwp: Plant capacity in kWp. If None, uses value from PLANT_CONFIG.

    Returns:
        DataFrame with solar position features added
    """
    print("☀️  Adding solar position...")

    # Use provided capacity or fallback to config
    if capacity_kwp is None:
        capacity_kwp = PLANT_CONFIG['capacity_kwp']

    location = Location(
        latitude=PLANT_CONFIG['latitude'],
        longitude=PLANT_CONFIG['longitude'],
        tz=PLANT_CONFIG['timezone']
    )

    times = pd.DatetimeIndex(df['generation_date'])
    solpos = location.get_solarposition(times)
    clearsky = location.get_clearsky(times)

    df['elevation'] = solpos['elevation'].values
    df['azimuth'] = solpos['azimuth'].values
    df['ghi'] = clearsky['ghi'].values

    # Calculate clear-sky expected production using correct capacity
    df['clearsky_expected_kwh'] = (
        (df['ghi'] / 1000) *
        capacity_kwp *
        PLANT_CONFIG['typical_pr']
    )

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features (hour, day of year, cyclical encodings).

    Args:
        df: DataFrame with 'generation_date' column

    Returns:
        DataFrame with temporal features added
    """
    # Linear temporal features
    df['hour'] = df['generation_date'].dt.hour
    df['day_of_year'] = df['generation_date'].dt.dayofyear
    df['month'] = df['generation_date'].dt.month
    df['day_of_week'] = df['generation_date'].dt.dayofweek

    # Cyclical encoding (CRITICAL for time series)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-based features.
    OPTIMIZED: Only uses features that improve performance.

    Args:
        df: DataFrame with weather columns (temp, clouds_all, visibility, rain_1h)

    Returns:
        DataFrame with weather features added
    """
    # Cloud impact (0-1 scale, 1 = clear sky)
    df['cloud_impact'] = 1 - (df['clouds_all'] / 100)

    # Rain indicator
    df['has_rain'] = (df['rain_1h'] > 0).astype(int)

    # Temperature non-linear effect
    # Panel efficiency decreases at high temps
    df['temp_squared'] = df['temp'] ** 2

    # Effective irradiance (GHI adjusted for clouds)
    # NOTE: Original formula. Tested adding humidity but it degraded performance.
    df['effective_irradiance'] = df['ghi'] * df['cloud_impact']

    # Elevation corrected by clouds
    df['elevation_x_cloud'] = df['elevation'] * df['cloud_impact']

    return df


def add_lagged_features(df: pd.DataFrame, target_col: str = 'generation_kwh') -> pd.DataFrame:
    """
    Add lagged features (historical production values).

    Args:
        df: DataFrame with target column
        target_col: Name of target column

    Returns:
        DataFrame with lagged features added
    """
    # Production from 24 hours ago (yesterday same time)
    df['production_lag_24h'] = df[target_col].shift(24)

    # Production from 168 hours ago (1 week ago)
    df['production_lag_168h'] = df[target_col].shift(168)

    return df


def add_rolling_features(df: pd.DataFrame, target_col: str = 'generation_kwh') -> pd.DataFrame:
    """
    Add rolling window features (moving averages).

    Args:
        df: DataFrame with target and feature columns
        target_col: Name of target column

    Returns:
        DataFrame with rolling features added
    """
    # 7-day rolling average of production
    df['production_7d_mean'] = df[target_col].rolling(168, min_periods=24).mean()

    # 7-day rolling average of temperature
    df['temp_7d_mean'] = df['temp'].rolling(168, min_periods=24).mean()

    return df


def filter_daytime_hours(df: pd.DataFrame, min_elevation: float = None) -> pd.DataFrame:
    """
    Filter to keep only daytime hours (when sun is up).

    Args:
        df: DataFrame with 'elevation' column
        min_elevation: Minimum sun elevation in degrees. Defaults to config value.

    Returns:
        Filtered DataFrame
    """
    if min_elevation is None:
        min_elevation = ML_CONFIG['min_sun_elevation']

    df = df[df['elevation'] > min_elevation].copy()
    return df


def create_all_features(df: pd.DataFrame, capacity_kwp: float = None) -> pd.DataFrame:
    """
    Main function: Create all features for ML modeling.
    This is the complete feature engineering pipeline.

    Args:
        df: DataFrame with generation_date, generation_kwh, and weather columns
        capacity_kwp: Plant capacity in kWp. If None, uses value from PLANT_CONFIG.

    Returns:
        DataFrame with all features added
    """
    print("🔧 Creating features...")

    # Solar position (elevation, azimuth, GHI, clearsky) - pass capacity
    df = add_solar_position(df, capacity_kwp=capacity_kwp)

    # Temporal features (hour, day_of_year, cyclical encodings)
    df = add_temporal_features(df)

    # Weather features (cloud_impact, rain flags, etc.)
    df = add_weather_features(df)

    # Historical features (lags and rolling averages)
    df = add_lagged_features(df)
    df = add_rolling_features(df)

    # Filter to daytime only
    df = filter_daytime_hours(df)

    # Drop rows with missing target
    df = df.dropna(subset=['generation_kwh'])

    # Count features
    from .config import EXCLUDE_FEATURES
    feature_count = len([col for col in df.columns if col not in EXCLUDE_FEATURES])

    print(f"✅ Feature engineering complete:")
    print(f"   {len(df):,} daytime hours")
    print(f"   {feature_count} features created\n")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature column names (excludes target and metadata).

    Args:
        df: DataFrame with all columns

    Returns:
        List of feature column names
    """
    from .config import EXCLUDE_FEATURES
    return [col for col in df.columns if col not in EXCLUDE_FEATURES]


def prepare_features_target(
    df: pd.DataFrame,
    feature_cols: list = None,
    target_col: str = 'generation_kwh'
) -> tuple:
    """
    Prepare X (features) and y (target) for modeling.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature columns. If None, auto-detected.
        target_col: Name of target column

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols]
    y = df[target_col]

    # Fill any remaining NaN in features (should be rare after filtering)
    X = X.ffill().bfill().fillna(0)

    return X, y
