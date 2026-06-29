#!/usr/bin/env python3
"""
Simple example script showing how to use the modular solar prediction code.

Usage:
    python train_model.py
"""

from pathlib import Path

from src import (
    data_loader,
    feature_engineering,
    model_training,
    evaluation,
    visualization
)
from src.config import DATA_PATHS

def main():
    """Main training pipeline using modular code."""

    print("="*80)
    print("SOLAR PRODUCTION PREDICTION - MODULAR TRAINING PIPELINE")
    print("="*80)

    # ===== STEP 1: LOAD DATA =====
    print("\n📊 STEP 1: Loading Data...")
    source_path = Path(DATA_PATHS['hourly_model_targets'])
    source_label = str(source_path) if source_path.exists() else DATA_PATHS['generation_5m']
    print(f"   Generation source: {source_label}")
    df = data_loader.load_complete_dataset()
    df = data_loader.filter_source_agreement_days(df)

    # ===== STEP 2: CREATE FEATURES =====
    print("\n🔧 STEP 2: Creating Features...")
    df = feature_engineering.create_all_features(df)

    # ===== STEP 3: SPLIT DATA =====
    print("\n✂️  STEP 3: Splitting Train/Test...")
    train_df, test_df = data_loader.split_train_test(df)

    # ===== STEP 4: PREPARE FEATURES/TARGET =====
    print("\n🎯 STEP 4: Preparing Features and Target...")
    feature_cols = feature_engineering.get_feature_columns(df)
    X_train, y_train = feature_engineering.prepare_features_target(train_df, feature_cols)
    X_test, y_test = feature_engineering.prepare_features_target(test_df, feature_cols)

    print(f"   Training: {len(X_train):,} samples × {len(feature_cols)} features")
    print(f"   Test:     {len(X_test):,} samples × {len(feature_cols)} features")

    # ===== STEP 5: TRAIN MODELS =====
    print("\n🤖 STEP 5: Training Models...")
    print("   This will train: Ridge, RandomForest, GradientBoosting, XGBoost")
    print("   Hyperparameter tuning is disabled during the data-cleaning pass")

    results = model_training.train_all_models(X_train, y_train, X_test, y_test)

    # ===== STEP 6: EVALUATE =====
    print("\n📈 STEP 6: Evaluating Models...")
    comparison_df = evaluation.create_comparison_dataframe(results)
    evaluation.print_model_comparison(comparison_df)

    # Print hyperparameters
    evaluation.print_hyperparameters(results)

    # ===== STEP 7: GET BEST MODEL =====
    print("\n🏆 STEP 7: Saving Best Model...")
    best_name, best_result = evaluation.get_best_model(results)
    print(f"   Best model: {best_name}")

    # Save best model
    model_training.save_model(
        best_result['model'],
        'models/ridge_model.pkl'
    )
    print(f"   ✅ Saved to models/ridge_model.pkl")

    # Save feature columns
    import pickle
    with open('models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print(f"   ✅ Saved feature columns to models/feature_columns.pkl")

    # ===== STEP 8: VISUALIZE =====
    print("\n📊 STEP 8: Creating Visualizations...")

    # Model comparison
    print("   Plotting model comparison...")
    visualization.plot_model_comparison(comparison_df)

    # Feature importance (if tree-based model available)
    print("   Plotting feature importance...")
    visualization.plot_feature_importance(results, feature_cols)

    # Ridge coefficients (if Ridge is best)
    if best_name == 'Ridge':
        print("   Plotting Ridge coefficients...")
        visualization.plot_ridge_coefficients(best_result['model'], feature_cols)

    # Three-line comparison
    print("   Plotting three-line comparison (first 30 days)...")
    predictions = {name: results[name]['model'].predict(X_test) for name in results.keys()}
    test_df_with_pred = test_df.copy()

    visualization.plot_three_line_comparison(
        test_df_with_pred,
        predictions,
        best_name,
        start_date=test_df['generation_date'].min(),
        end_date=test_df['generation_date'].min() + pd.Timedelta(days=30)
    )

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("  1. Check models/ folder for saved models")
    print("  2. Run: streamlit run app_solar_monitoring_enhanced.py")
    print("  3. See README.md for local run and verification commands")
    print("="*80)


if __name__ == "__main__":
    import pandas as pd  # Need this for Timedelta in visualization
    main()
