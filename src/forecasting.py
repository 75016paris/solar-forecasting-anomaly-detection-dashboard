"""
Forecasting module for Solar Production Prediction.
Handles weather data fetching and forecast feature creation.
"""

import pandas as pd
import numpy as np
import requests
from typing import Optional, Tuple

from . import feature_engineering


def fetch_weather_forecast(api_key: str, lat: float, lon: float) -> Optional[pd.DataFrame]:
    """
    Fetch 5-day weather forecast from OpenWeather API.

    Args:
        api_key: OpenWeather API key
        lat: Latitude of the location
        lon: Longitude of the location

    Returns:
        DataFrame with forecast weather data (3-hour intervals)
        Returns None if API call fails
    """
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"

        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        forecast_data = []

        for item in data.get('list', []):
            forecast_data.append({
                'timestamp': pd.to_datetime(item['dt'], unit='s'),
                'temp': item['main']['temp'],
                'feels_like': item['main']['feels_like'],
                'pressure': item['main']['pressure'],
                'humidity': item['main']['humidity'],
                'clouds_all': item['clouds']['all'],
                'wind_speed': item['wind']['speed'],
                'wind_deg': item['wind'].get('deg', 0),
                'weather_main': item['weather'][0]['main'],
                'weather_description': item['weather'][0]['description'],
                'visibility': item.get('visibility', 10000),
                'rain_1h': item.get('rain', {}).get('1h', 0)
            })

        df = pd.DataFrame(forecast_data)
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def resample_forecast_to_hourly(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample 3-hourly forecast data to hourly using linear interpolation.

    Args:
        weather_df: DataFrame with 3-hourly forecast data

    Returns:
        DataFrame resampled to hourly intervals
    """
    # Resample to hourly with linear interpolation
    # Infer object dtypes first to avoid deprecation warning
    weather_df = weather_df.infer_objects(copy=False)
    weather_hourly = weather_df.resample('1h').interpolate(method='linear')

    # Forward fill any remaining NaN (shouldn't happen with linear interpolation)
    weather_hourly = weather_hourly.ffill()

    print(f"✓ Resampled from {len(weather_df)} to {len(weather_hourly)} hourly data points")

    return weather_hourly


def create_lag_features_hour_matched(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame
) -> Tuple[list, list, list]:
    """
    Create hour-matched lag features from historical production data.

    This ensures that lag features reflect actual hour-specific production patterns:
    - 6 AM forecast gets lag values from 6 AM historical data (~5-10 kWh)
    - 12 PM forecast gets lag values from 12 PM historical data (~80-100 kWh)
    - 6 PM forecast gets lag values from 6 PM historical data (~20-30 kWh)

    This prevents all hours from looking identical to the model, allowing
    weather features to have proper influence on predictions.

    Args:
        forecast_df: DataFrame with forecast times (index)
        historical_df: DataFrame with historical production data

    Returns:
        Tuple of (lag_24h_values, lag_168h_values, seven_day_means)
    """
    # Get the last 24 hours and last 168 hours of actual production
    last_24h = historical_df.tail(24)[['generation_kwh']].copy()
    last_168h = historical_df.tail(168)[['generation_kwh']].copy()

    # Add hour column for matching
    last_24h['hour'] = last_24h.index.hour
    last_168h['hour'] = last_168h.index.hour

    # For each forecast hour, match it with the corresponding historical hour
    lag_24h_values = []
    lag_168h_values = []
    seven_day_means = []

    for forecast_time in forecast_df.index:
        hour_of_day = forecast_time.hour

        # Find the production from 24h ago at the same hour
        matching_24h = last_24h[last_24h['hour'] == hour_of_day]
        if len(matching_24h) > 0:
            lag_24h_value = matching_24h['generation_kwh'].iloc[-1]  # Most recent match
        else:
            lag_24h_value = 0

        # Find the production from 168h ago (7 days) at the same hour
        matching_168h = last_168h[last_168h['hour'] == hour_of_day]
        if len(matching_168h) > 0:
            lag_168h_value = matching_168h['generation_kwh'].mean()  # Average of this hour over last week
        else:
            lag_168h_value = 0

        # 7-day rolling mean (daytime only to avoid zeros skewing the mean)
        daytime_7d = last_168h[last_168h['generation_kwh'] > 0]['generation_kwh']
        if len(daytime_7d) > 0:
            seven_day_mean = daytime_7d.mean()
        else:
            seven_day_mean = 0

        lag_24h_values.append(lag_24h_value)
        lag_168h_values.append(lag_168h_value)
        seven_day_means.append(seven_day_mean)

    return lag_24h_values, lag_168h_values, seven_day_means


def create_forecast_features(
    weather_df: pd.DataFrame,
    historical_df: Optional[pd.DataFrame] = None,
    capacity_kwp: Optional[float] = None
) -> Optional[pd.DataFrame]:
    """
    Create complete feature set for forecasting.
    Uses same features as trained model.

    Args:
        weather_df: DataFrame with forecast weather data (hourly)
        historical_df: DataFrame with historical production data (for lag features)
        capacity_kwp: Plant capacity in kWp (for correct clearsky calculation)

    Returns:
        DataFrame with all features ready for model prediction
        Returns None if feature creation fails
    """
    df = weather_df.copy()

    # Add generation_date column from index (required by feature engineering)
    df['generation_date'] = df.index

    # Add generation_kwh column (dummy - not used for prediction)
    df['generation_kwh'] = 0

    try:
        # Add solar position features with correct plant capacity
        df = feature_engineering.add_solar_position(df, capacity_kwp=capacity_kwp)

        # Add temporal features
        df = feature_engineering.add_temporal_features(df)

        # Add weather features
        df = feature_engineering.add_weather_features(df)

        # Add historical lag features
        if historical_df is not None and 'generation_kwh' in historical_df.columns:
            # Use hour-matched lag features for realistic patterns
            lag_24h, lag_168h, seven_day_mean = create_lag_features_hour_matched(
                df, historical_df
            )

            df['production_lag_24h'] = lag_24h
            df['production_lag_168h'] = lag_168h
            df['production_7d_mean'] = seven_day_mean

            print(f"✓ Using hour-matched lag values from historical data")
            print(f"  Lag 24h range: {df['production_lag_24h'].min():.1f} - {df['production_lag_24h'].max():.1f} kWh")
            print(f"  7d mean: {df['production_7d_mean'].mean():.1f} kWh")
        else:
            # Fallback: use reasonable default values (not 0!)
            df['production_lag_24h'] = 50.0  # Typical hourly daytime production
            df['production_lag_168h'] = 50.0
            df['production_7d_mean'] = 40.0

            print("⚠️  WARNING: No historical data provided. Using default lag values.")

        # Add temp_7d_mean if not already present
        if 'temp_7d_mean' not in df.columns:
            temp_col = 'temp' if 'temp' in df.columns else 'temperature'
            df['temp_7d_mean'] = df[temp_col].rolling(
                window=168, min_periods=1
            ).mean().fillna(df[temp_col].mean())

        # Filter daytime hours
        df = feature_engineering.filter_daytime_hours(df)

        return df

    except Exception as e:
        print(f"Error creating forecast features: {e}")
        import traceback
        traceback.print_exc()
        return None


def make_forecast(
    model,
    feature_columns: list,
    weather_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    capacity_kwp: float
) -> Optional[pd.DataFrame]:
    """
    Complete forecasting pipeline: from weather data to predictions.

    Args:
        model: Trained ML model
        feature_columns: List of feature column names expected by model
        weather_df: DataFrame with forecast weather data
        historical_df: DataFrame with historical production data
        capacity_kwp: Plant capacity in kWp

    Returns:
        DataFrame with predictions and all features
        Returns None if forecasting fails
    """
    print("🔮 Starting forecast pipeline...")

    # Step 1: Resample to hourly
    print("📊 Step 1: Resampling to hourly intervals...")
    weather_hourly = resample_forecast_to_hourly(weather_df)

    # Step 2: Create features
    print("🔧 Step 2: Creating forecast features...")
    forecast_with_features = create_forecast_features(
        weather_hourly,
        historical_df=historical_df,
        capacity_kwp=capacity_kwp
    )

    if forecast_with_features is None:
        print("❌ Failed to create features")
        return None

    # Step 3: Select model features
    print("🎯 Step 3: Selecting model features...")
    X_forecast = forecast_with_features[feature_columns]

    # Step 4: Make predictions
    print("🤖 Step 4: Making predictions...")
    predictions = model.predict(X_forecast)
    predictions = np.maximum(predictions, 0)  # Non-negative constraint

    forecast_with_features['predicted_kwh'] = predictions

    print(f"✅ Forecast complete: {len(forecast_with_features)} hourly predictions")
    print(f"   Total predicted: {predictions.sum():.1f} kWh")
    print(f"   Daily average: {predictions.sum() / len(set(forecast_with_features.index.date)):.1f} kWh/day")

    return forecast_with_features
