"""
Configuration module for Solar Production Prediction.
Centralizes all configuration parameters.
"""

# ==================================================================================
# PLANT CONFIGURATION
# ==================================================================================

PLANT_CONFIG = {
    'name': 'Plant A',
    'capacity_kwp': 269.28,
    'latitude': 24.022350694140282,
    'longitude': 90.29576719011767,
    'timezone': 'Asia/Dhaka',
    'typical_pr': 0.82  # Performance Ratio
}

# ==================================================================================
# DATA PATHS
# ==================================================================================

DATA_PATHS = {
    'inverter_plants': 'data/inverter_plants.csv',
    'generation_5m': 'data/inverter_five_minutes_generation_logs.csv',
    'weather': 'open_data/gazipur_weather.csv',
    'plants_clean': 'clean/plants_clean.csv',
    'generation_5m_clean': 'clean/generation_5min_clean.csv',
    'hourly_model_targets': 'clean/hourly_model_targets.csv',
    'daily_source_agreement': 'clean/daily_source_agreement.csv',
    'weather_hourly_clean': 'clean/weather_hourly_clean.csv',
    'data_quality_summary': 'clean/data_quality_summary.csv',
    'model': 'models/ridge_model.pkl',
    'feature_columns': 'models/feature_columns.pkl',
    'model_metrics': 'models/model_metrics.pkl'
}

# ==================================================================================
# ML CONFIGURATION
# ==================================================================================

ML_CONFIG = {
    # Train/test split - Hybrid approach
    # Priority 1: Plant-specific dates (use for problematic plants)
    'plant_split_dates': {
        # Use June 2025 as split to avoid recent data quality issues.
        'Plant A': '2025-06-01',
        'Plant B': '2025-06-01',
        'Plant C': '2025-06-01',
        'Plant D': '2025-06-01',
        'Plant E': '2025-04-01',  # Plant E has data issues after April
    },

    # Priority 2: Use last N months as test set (fallback)
    'test_months': 3,

    # Priority 3: Percentage fallback (if not enough data for test_months)
    'train_test_ratio': 0.80,  # 80% training, 20% test

    # Validation - minimum test samples required
    'min_test_samples': 500,

    # Filtering
    'min_sun_elevation': 5,  # Degrees - filter out night hours
    'min_hourly_completeness_pct': 80,  # Drop hourly targets with <80% of expected 5-minute records
    'source_agreement_training_filter': True,
    # Plant-specific data filter for model training/evaluation. Keep Plant E loose
    # because it has known source-data issues.
    'max_daily_inverter_ratio_difference': 1.00,
    'max_daily_inverter_ratio_difference_by_plant': {
        'Plant A': 1.00,
        'Plant B': 1.00,
        'Plant C': 1.00,
        'Plant D': 0.20,
        'Plant E': 1.00,
    },
    'min_daily_kwh_for_agreement_filter': 1.0,  # Avoid ratio filtering on tiny/night-only daily totals

    # Hyperparameter tuning
    # Disabled during the data-cleaning pass; tune only after cleaned targets are verified.
    'tune_hyperparameters': False,
    'cv_splits': 3,  # Time series CV splits

    # LSTM specific
    'lstm_lookback': 24,  # Hours of history
    'lstm_epochs': 50,
    'lstm_batch_size': 32,

    # Model export
    'export_dir': 'models',
    'output_dir': 'output'
}

# ==================================================================================
# FEATURE CONFIGURATION
# ==================================================================================

# Features to use for modeling (optimized set - 24 features)
# Based on experimentation: adding more weather features degraded performance
FEATURE_GROUPS = {
    'solar_position': ['elevation', 'azimuth', 'ghi'],

    'temporal_cyclical': ['hour_sin', 'hour_cos', 'day_sin', 'day_cos'],

    'temporal_linear': ['hour', 'month', 'day_of_week', 'day_of_year'],

    'weather_basic': ['temp', 'temp_squared', 'clouds_all', 'visibility', 'rain_1h', 'has_rain'],

    'weather_derived': ['cloud_impact', 'effective_irradiance', 'elevation_x_cloud'],

    'historical': ['production_lag_24h', 'production_lag_168h', 'production_7d_mean', 'temp_7d_mean']
}

# Flatten all features into a single list
ALL_FEATURES = [feat for group in FEATURE_GROUPS.values() for feat in group]

# Features to exclude from modeling
EXCLUDE_FEATURES = [
    'generation_date',
    'generation_kwh',
    'generation_kwh_raw',
    'clearsky_expected_kwh',
    'records_collected',
    'expected_records',
    'data_completeness_pct',
    'data_available',
    'plant_id',
    'plant_name',
    'missing_records',
    'target_source',
    'target_quality_flag'
]

# ==================================================================================
# VISUALIZATION CONFIGURATION
# ==================================================================================

PLOT_CONFIG = {
    'figure_dpi': 100,
    'style': 'whitegrid',
    'color_palette': {
        'actual': '#3498DB',
        'ml_predicted': '#4ECDC4',
        'clearsky': '#FF6B6B',
        'residual_normal': 'lightblue',
        'residual_alert': 'red',
        'best_model': '#2ecc71',
        'other_models': '#3498db'
    },
    'fonts': {
        'title': 16,
        'subtitle': 14,
        'label': 12,
        'legend': 13
    }
}

# ==================================================================================
# HYPERPARAMETER GRIDS
# ==================================================================================

HYPERPARAMETER_GRIDS = {
    'Ridge': {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    },

    'RandomForest': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_split': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        'min_samples_leaf': [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    },

    'GradientBoosting': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        'learning_rate': [0.05, 0.1, 0.2],
        'min_samples_split': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },

    'XGBoost': {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0, 0.9, 0.95, 1.0],
        'colsample_bytree': [0.8, 1.0, 0.9, 0.95, 1.0]
    }
}

# Number of random combinations to test (for RandomizedSearchCV)
# This is a single integer that says "test N random combinations from the grid"
# Higher = more thorough search but slower
N_ITER_RANDOM_SEARCH = {
    'RandomForest': 20,       # Test 20 random combinations
    'GradientBoosting': 20,   # Test 20 random combinations
    'XGBoost': 20             # Test 20 random combinations
}
