"""
Evaluation module for Solar Production Prediction.
Handles model evaluation and metrics calculation.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, model_name, X_train, y_train, X_test, y_test, train_time):
    """
    Evaluate a trained model and return comprehensive metrics.

    Args:
        model: Trained model
        model_name: Name of the model
        X_train, y_train: Training data
        X_test, y_test: Test data
        train_time: Training time in seconds

    Returns:
        dict with all metrics
    """
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_r2': r2_score(y_test, y_test_pred),
        'train_time': train_time
    }

    # WAPE (Weighted Absolute Percentage Error)
    # More stable than MAPE for solar data, where low/zero production hours are common.
    test_denominator = np.abs(y_test).sum()
    metrics['test_wape'] = (
        np.abs(y_test - y_test_pred).sum() / test_denominator * 100
        if test_denominator > 0 else np.nan
    )

    # Print metrics
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Test MAE:      {metrics['test_mae']:.3f} kWh")
    print(f"  Test RMSE:     {metrics['test_rmse']:.3f} kWh")
    print(f"  Test R²:       {metrics['test_r2']:.4f}")
    print(f"  Test WAPE:     {metrics['test_wape']:.2f}%")

    return metrics


def create_comparison_dataframe(results):
    """
    Create comparison DataFrame from model results.

    Args:
        results: dict of model results from train_all_models()

    Returns:
        DataFrame sorted by R² score (best first)
    """
    comparison_df = pd.DataFrame([
        {
            'Model': results[key]['metrics']['model_name'],
            'Test MAE (kWh)': results[key]['metrics']['test_mae'],
            'Test RMSE (kWh)': results[key]['metrics']['test_rmse'],
            'Test R²': results[key]['metrics']['test_r2'],
            'Test WAPE (%)': results[key]['metrics']['test_wape'],
            'Training Time (s)': results[key]['metrics']['train_time']
        }
        for key in results.keys()
    ]).sort_values('Test R²', ascending=False)

    return comparison_df


def print_model_comparison(comparison_df):
    """
    Print model comparison table.

    Args:
        comparison_df: DataFrame from create_comparison_dataframe()
    """
    print("\n" + "="*110)
    print("  MODEL COMPARISON - TEST SET PERFORMANCE")
    print("="*110)
    print(comparison_df.to_string(index=False))
    print("="*110)

    # Best model
    best_model_name = comparison_df.iloc[0]['Model']
    best_r2 = comparison_df.iloc[0]['Test R²']
    best_mae = comparison_df.iloc[0]['Test MAE (kWh)']

    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   → R² Score: {best_r2:.4f}")
    print(f"   → MAE: {best_mae:.3f} kWh")


def get_best_model(results):
    """
    Get the best model based on R² score.

    Args:
        results: dict of model results from train_all_models()

    Returns:
        tuple of (model_name, model_dict)
    """
    comparison_df = create_comparison_dataframe(results)
    best_model_name = comparison_df.iloc[0]['Model']

    # Find the key in results
    for key in results.keys():
        if results[key]['metrics']['model_name'] == best_model_name:
            return key, results[key]

    return None, None


def calculate_residuals(y_true, y_pred):
    """
    Calculate residuals (actual - predicted).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        numpy array of residuals
    """
    return y_true - y_pred


def detect_anomalies(residuals, threshold_std=2):
    """
    Detect anomalies based on residual threshold.

    Args:
        residuals: Array of residuals
        threshold_std: Number of standard deviations for threshold

    Returns:
        tuple of (anomaly_mask, threshold_value)
    """
    threshold = residuals.std() * threshold_std
    anomaly_mask = residuals < -threshold
    return anomaly_mask, threshold


def print_hyperparameters(results):
    """
    Print best hyperparameters for each model.

    Args:
        results: dict of model results from train_all_models()
    """
    print("\n⚙️  Best Hyperparameters:")
    for model_name, result in results.items():
        if 'best_params' in result and result['best_params']:
            print(f"   {model_name}: {result['best_params']}")
