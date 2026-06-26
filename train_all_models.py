#!/usr/bin/env python3
"""
Master training script to train models for all name-redacted plants.

This script trains models for all 5 plants: Plant A, Plant B, Plant C, Plant D, Plant E.
Each plant gets its own dedicated model saved to models/{PLANT}/ directory.

Usage:
    python train_all_models.py

    # To train only Ridge (faster):
    python train_all_models.py --ridge-only

    # Bounded tuning experiment for reliable plants only:
    python train_all_models.py --reliable-only --tune
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from src import (
    data_loader,
    feature_engineering,
    model_training,
    evaluation
)
from src.config import DATA_PATHS, ML_CONFIG, HYPERPARAMETER_GRIDS, N_ITER_RANDOM_SEARCH

RELIABLE_PLANTS = ['Plant A', 'Plant B', 'Plant C', 'Plant D']

# Plant configurations
PLANTS = {
    'Plant A': {
        'plant_id': 900001,
        'capacity_kwp': 269.28,
        'output_dir': Path('models/DEMO_A')
    },
    'Plant B': {
        'plant_id': 900002,
        'capacity_kwp': 522.72,
        'output_dir': Path('models/DEMO_B')
    },
    'Plant C': {
        'plant_id': 900003,
        'capacity_kwp': 227.04,
        'output_dir': Path('models/DEMO_C')
    },
    'Plant D': {
        'plant_id': 900004,
        'capacity_kwp': 525.36,
        'output_dir': Path('models/DEMO_D')
    },
    'Plant E': {
        'plant_id': 900005,
        'capacity_kwp': 285.12,
        'output_dir': Path('models/DEMO_E')
    }
}


def apply_bounded_tuning_config():
    """Use small search spaces for a one-pass reliable-plants tuning experiment."""
    ML_CONFIG['tune_hyperparameters'] = True

    HYPERPARAMETER_GRIDS['Ridge'] = {
        'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    }
    HYPERPARAMETER_GRIDS['RandomForest'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 30],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
    }
    HYPERPARAMETER_GRIDS['GradientBoosting'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [1, 3, 5],
    }
    HYPERPARAMETER_GRIDS['XGBoost'] = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }
    N_ITER_RANDOM_SEARCH['RandomForest'] = 8
    N_ITER_RANDOM_SEARCH['GradientBoosting'] = 8
    N_ITER_RANDOM_SEARCH['XGBoost'] = 8


def load_existing_best_metrics(plants: dict) -> pd.DataFrame:
    """Load current best metrics before an experiment overwrites model files."""
    rows = []
    for plant_name, plant_info in plants.items():
        path = plant_info['output_dir'] / 'model_comparison.csv'
        if not path.exists():
            continue
        comparison = pd.read_csv(path)
        if comparison.empty:
            continue
        best = comparison.iloc[0]
        rows.append({
            'Plant': plant_name,
            'Before best model': best['Model'],
            'Before R²': best['Test R²'],
            'Before WAPE (%)': best.get('Test WAPE (%)'),
            'Before MAE (kWh)': best['Test MAE (kWh)'],
        })
    return pd.DataFrame(rows)


def save_tuning_comparison(before_df: pd.DataFrame, training_results: list):
    """Save a before/after table for the bounded tuning experiment."""
    after_rows = []
    for result in training_results:
        if result['success']:
            after_rows.append({
                'Plant': result['plant'],
                'After best model': result['best_model'],
                'After R²': result['r2'],
                'After WAPE (%)': result['wape'],
                'After MAE (kWh)': result['mae'],
            })
    after_df = pd.DataFrame(after_rows)
    if after_df.empty:
        return

    comparison = before_df.merge(after_df, on='Plant', how='right') if not before_df.empty else after_df
    if {'Before R²', 'After R²'}.issubset(comparison.columns):
        comparison['R² delta'] = comparison['After R²'] - comparison['Before R²']
    if {'Before WAPE (%)', 'After WAPE (%)'}.issubset(comparison.columns):
        comparison['WAPE delta (%)'] = comparison['After WAPE (%)'] - comparison['Before WAPE (%)']

    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'bounded_tuning_reliable_plants.csv'
    comparison.to_csv(output_path, index=False)
    print(f"\n📄 Tuning comparison saved to {output_path}")
    print(comparison.to_string(index=False))


def load_complete_dataset_for_plant(plant_name: str) -> pd.DataFrame:
    """
    Load complete dataset for a specific plant with generation and weather data.

    Args:
        plant_name: Name of the plant (e.g., 'Plant B')

    Returns:
        DataFrame with all data ready for feature engineering
    """
    print(f"📊 Loading complete dataset for {plant_name}...")

    # Load generation data for specific plant. data_loader prefers
    # clean/hourly_model_targets.csv when build_clean_dataset.py has run.
    source_path = Path(DATA_PATHS['hourly_model_targets'])
    source_label = str(source_path) if source_path.exists() else DATA_PATHS['generation_5m']
    print(f"  Source: {source_label}")
    df_generation = data_loader.load_generation_data(plant_name=plant_name)
    print(f"  ✓ {len(df_generation):,} hourly generation records loaded")

    # Load weather data
    df_weather = data_loader.load_weather_data()
    print(f"  ✓ Weather data loaded")

    # Merge
    df = data_loader.merge_generation_weather(df_generation, df_weather)
    print(f"  ✓ Data merged")

    # Filter weak validation days before model training/evaluation.
    df = data_loader.filter_source_agreement_days(df, plant_name=plant_name)

    return df


def train_plant_model(plant_name: str, plant_info: dict, ridge_only: bool = False):
    """Train model for a specific plant."""

    print("\n" + "="*80)
    print(f"TRAINING: {plant_name}")
    print("="*80)
    print(f"Plant ID:   {plant_info['plant_id']}")
    print(f"Capacity:   {plant_info['capacity_kwp']} kWp")
    print(f"Output Dir: {plant_info['output_dir']}")
    print("="*80)

    try:
        # Load data
        print(f"\n📊 STEP 1: Loading Data for {plant_name}...")
        df = load_complete_dataset_for_plant(plant_name)

        # Create features with correct plant capacity
        print("\n🔧 STEP 2: Creating Features...")
        df = feature_engineering.create_all_features(df, capacity_kwp=plant_info['capacity_kwp'])

        # Split data
        print("\n✂️  STEP 3: Splitting Train/Test...")
        train_df, test_df = data_loader.split_train_test(df, plant_name=plant_name)

        # Prepare features/target
        print("\n🎯 STEP 4: Preparing Features and Target...")
        feature_cols = feature_engineering.get_feature_columns(df)
        X_train, y_train = feature_engineering.prepare_features_target(train_df, feature_cols)
        X_test, y_test = feature_engineering.prepare_features_target(test_df, feature_cols)

        print(f"   Training: {len(X_train):,} samples × {len(feature_cols)} features")
        print(f"   Test:     {len(X_test):,} samples × {len(feature_cols)} features")

        # Train models
        print("\n🤖 STEP 5: Training Models...")
        if ridge_only:
            print("   Training: Ridge only (fast mode)")
            models_to_train = ['Ridge']
        else:
            print("   Training: Ridge, RandomForest, GradientBoosting, XGBoost")
            models_to_train = None

        results = model_training.train_all_models(
            X_train, y_train, X_test, y_test,
            models_to_train=models_to_train
        )

        # Evaluate
        print("\n📈 STEP 6: Evaluating Models...")
        comparison_df = evaluation.create_comparison_dataframe(results)
        evaluation.print_model_comparison(comparison_df)

        # Save best model
        print(f"\n💾 STEP 7: Saving Best Model for {plant_name}...")
        best_name, best_result = evaluation.get_best_model(results)
        print(f"   Best model: {best_name}")

        # Create output directory
        output_dir = plant_info['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f'{best_name.lower().replace(" ", "_")}_model.pkl'
        model_training.save_model(best_result['model'], model_path)
        print(f"   ✅ Model saved to {model_path}")

        # Save feature columns
        import pickle
        with open(output_dir / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        print(f"   ✅ Feature columns saved")

        # Save comparison metrics
        comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
        print(f"   ✅ Comparison metrics saved")

        # Save plant info
        threshold_by_plant = ML_CONFIG.get('max_daily_inverter_ratio_difference_by_plant', {})
        plant_metadata = {
            'plant_name': plant_name,
            'plant_id': plant_info['plant_id'],
            'capacity_kwp': plant_info['capacity_kwp'],
            'target_source': DATA_PATHS['hourly_model_targets'],
            'incomplete_hour_rule': 'generation_kwh is NaN/excluded when hourly 5-minute completeness is below threshold',
            'min_hourly_completeness_pct': ML_CONFIG.get('min_hourly_completeness_pct'),
            'source_agreement_training_filter': ML_CONFIG.get('source_agreement_training_filter'),
            'max_daily_inverter_ratio_difference': threshold_by_plant.get(
                plant_name,
                ML_CONFIG.get('max_daily_inverter_ratio_difference')
            ),
            'model_scope_note': 'Known source-data issues' if plant_name == 'Plant E' else 'Reliable-days model'
        }
        with open(output_dir / 'plant_info.pkl', 'wb') as f:
            pickle.dump(plant_metadata, f)
        print(f"   ✅ Plant info saved")

        return {
            'plant': plant_name,
            'success': True,
            'best_model': best_name,
            'r2': best_result['metrics']['test_r2'],
            'wape': best_result['metrics']['test_wape'],
            'mae': best_result['metrics']['test_mae']
        }

    except Exception as e:
        print(f"\n❌ ERROR training {plant_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'plant': plant_name,
            'success': False,
            'error': str(e)
        }


def main():
    """Main training pipeline for all plants."""

    # Check CLI flags
    ridge_only = '--ridge-only' in sys.argv
    tune = '--tune' in sys.argv
    reliable_only = '--reliable-only' in sys.argv

    if tune:
        apply_bounded_tuning_config()

    plants_to_train = {
        plant_name: plant_info
        for plant_name, plant_info in PLANTS.items()
        if not reliable_only or plant_name in RELIABLE_PLANTS
    }
    before_metrics = load_existing_best_metrics(plants_to_train) if tune else pd.DataFrame()

    start_time = datetime.now()

    print("="*80)
    print("SOLAR PRODUCTION PREDICTION - TRAIN ALL PLANTS")
    print("="*80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    if ridge_only:
        training_mode = 'Ridge only (FAST)'
    elif tune:
        training_mode = 'All models with bounded tuning'
    else:
        training_mode = 'All models (NO GRID SEARCH DURING CLEANING PASS)'
    print(f"Training mode: {training_mode}")
    print(f"Plants to train: {len(plants_to_train)}")
    print("="*80)

    # Train all plants
    training_results = []
    for plant_name, plant_info in plants_to_train.items():
        result = train_plant_model(plant_name, plant_info, ridge_only)
        training_results.append(result)

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {duration}")
    print()

    # Success/failure summary
    success_count = sum(1 for r in training_results if r['success'])
    print(f"✅ Successful: {success_count}/{len(plants_to_train)}")
    print(f"❌ Failed: {len(plants_to_train) - success_count}/{len(plants_to_train)}")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    for result in training_results:
        if result['success']:
            print(f"  ✅ {result['plant']:20s} | {result['best_model']:20s} | "
                  f"R²={result['r2']:.4f} | MAE={result['mae']:.2f} kWh")
        else:
            print(f"  ❌ {result['plant']:20s} | ERROR: {result['error']}")

    if tune:
        save_tuning_comparison(before_metrics, training_results)

    print("\n" + "="*80)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check models/ directory for all plant models")
    print("  2. Restart Streamlit app to use new models")
    print("  3. Test anomaly detection for all plants")
    print()
    if not ridge_only:
        print("💡 TIP: Use 'python train_all_models.py --ridge-only' for faster iteration")
        print("   Full training is required when you want a real model comparison.")
    print("="*80)


if __name__ == "__main__":
    main()
