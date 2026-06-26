"""
Model training module for Solar Production Prediction.
Handles training of multiple ML models with hyperparameter tuning.
"""

import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

from .config import ML_CONFIG, HYPERPARAMETER_GRIDS, N_ITER_RANDOM_SEARCH
from .evaluation import evaluate_model


def get_time_series_cv():
    """Get TimeSeriesSplit cross-validator."""
    return TimeSeriesSplit(n_splits=ML_CONFIG['cv_splits'])


def train_ridge(X_train, y_train, X_test, y_test, tune=True):
    """
    Train Ridge Regression with optional hyperparameter tuning.

    Returns:
        dict with model, metrics, and training time
    """
    print("="*60)
    print("MODEL: Ridge Regression")
    print("="*60)

    if tune and ML_CONFIG['tune_hyperparameters']:
        print("🔍 Tuning hyperparameters...")
        grid_search = GridSearchCV(
            Ridge(),
            HYPERPARAMETER_GRIDS['Ridge'],
            cv=get_time_series_cv(),
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"  Best params: {best_params}")
    else:
        model = Ridge(alpha=100.0)  # Default from tuning
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_params = {'alpha': 100.0}

    metrics = evaluate_model(model, "Ridge Regression", X_train, y_train, X_test, y_test, train_time)
    print()

    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params,
        'train_time': train_time
    }


def train_random_forest(X_train, y_train, X_test, y_test, tune=True):
    """Train Random Forest with optional hyperparameter tuning."""
    print("="*60)
    print("MODEL: Random Forest")
    print("="*60)

    if tune and ML_CONFIG['tune_hyperparameters']:
        print("🔍 Tuning hyperparameters (this may take a few minutes)...")
        random_search = RandomizedSearchCV(
            RandomForestRegressor(random_state=42, n_jobs=-1),
            HYPERPARAMETER_GRIDS['RandomForest'],
            n_iter=N_ITER_RANDOM_SEARCH['RandomForest'],
            cv=get_time_series_cv(),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        model = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"  Best params: {best_params}")
    else:
        model = RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_split=10,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_params = {}

    metrics = evaluate_model(model, "Random Forest", X_train, y_train, X_test, y_test, train_time)
    print()

    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params,
        'train_time': train_time
    }


def train_gradient_boosting(X_train, y_train, X_test, y_test, tune=True):
    """Train Gradient Boosting with optional hyperparameter tuning."""
    print("="*60)
    print("MODEL: Gradient Boosting")
    print("="*60)

    if tune and ML_CONFIG['tune_hyperparameters']:
        print("🔍 Tuning hyperparameters...")
        random_search = RandomizedSearchCV(
            GradientBoostingRegressor(random_state=42),
            HYPERPARAMETER_GRIDS['GradientBoosting'],
            n_iter=N_ITER_RANDOM_SEARCH['GradientBoosting'],
            cv=get_time_series_cv(),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        model = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"  Best params: {best_params}")
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.1,
            min_samples_split=10, random_state=42
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_params = {}

    metrics = evaluate_model(model, "Gradient Boosting", X_train, y_train, X_test, y_test, train_time)
    print()

    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params,
        'train_time': train_time
    }


def train_xgboost(X_train, y_train, X_test, y_test, tune=True):
    """Train XGBoost with optional hyperparameter tuning."""
    try:
        import xgboost as xgb
    except ImportError:
        print("⚠️  Skipping XGBoost (not installed)\n")
        return None

    print("="*60)
    print("MODEL: XGBoost")
    print("="*60)

    if tune and ML_CONFIG['tune_hyperparameters']:
        print("🔍 Tuning hyperparameters...")
        random_search = RandomizedSearchCV(
            xgb.XGBRegressor(random_state=42, n_jobs=-1),
            HYPERPARAMETER_GRIDS['XGBoost'],
            n_iter=N_ITER_RANDOM_SEARCH['XGBoost'],
            cv=get_time_series_cv(),
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=42
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        train_time = time.time() - start_time

        model = random_search.best_estimator_
        best_params = random_search.best_params_
        print(f"  Best params: {best_params}")
    else:
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=7, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        best_params = {}

    metrics = evaluate_model(model, "XGBoost", X_train, y_train, X_test, y_test, train_time)
    print()

    return {
        'model': model,
        'metrics': metrics,
        'best_params': best_params,
        'train_time': train_time
    }


def train_all_models(X_train, y_train, X_test, y_test, models_to_train=None):
    """
    Train all available models.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        models_to_train: List of model names to train. If None, trains all.

    Returns:
        dict with results for each model
    """
    available_models = {
        'Ridge': train_ridge,
        'RandomForest': train_random_forest,
        'GradientBoosting': train_gradient_boosting,
        'XGBoost': train_xgboost
        #'SVR': train_svr # (slow and inacurate)
    }

    if models_to_train is None:
        models_to_train = list(available_models.keys())

    tuning_status = 'with hyperparameter tuning' if ML_CONFIG['tune_hyperparameters'] else 'without hyperparameter tuning'
    print(f"\n🤖 Training Models {tuning_status}...\n")

    results = {}
    for model_name in models_to_train:
        if model_name in available_models:
            result = available_models[model_name](X_train, y_train, X_test, y_test)
            if result is not None:
                results[model_name] = result

    return results


def save_model(model, filepath):
    """Save model to disk."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath):
    """Load model from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
