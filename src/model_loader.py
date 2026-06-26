"""
Model Loader module for Solar Production Prediction.
Handles loading trained models and feature columns for all plants.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import model_training


# Plant to directory mapping
PLANT_DIR_MAP = {
    'Plant A': 'models/DEMO_A',
    'Plant B': 'models/DEMO_B',
    'Plant C': 'models/DEMO_C',
    'Plant D': 'models/DEMO_D',
    'Plant E': 'models/DEMO_E'
}

# Backward-compatible labels from older local sessions/screenshots. Without
# this, stale session state can silently fall back to Plant A and duplicate its
# metrics across other plants.
PLANT_NAME_ALIASES = {
    'Demo Plant A': 'Plant A',
    'Demo Plant B': 'Plant B',
    'Demo Plant C': 'Plant C',
    'Demo Plant D': 'Plant D',
    'Demo Plant E': 'Plant E',
    'Demo Plant E (Low Quality)': 'Plant E',
}

MODEL_FILE_MAP = {
    'Ridge Regression': 'ridge_model.pkl',
    'Random Forest': 'randomforest_model.pkl',
    'Gradient Boosting': 'gradientboosting_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}


def normalize_plant_name(plant_name: str) -> str:
    """Normalize legacy plant labels to the current public labels."""
    return PLANT_NAME_ALIASES.get(plant_name, plant_name)


def get_plant_directory(plant_name: str) -> Path:
    """
    Get the model directory path for a given plant.

    Args:
        plant_name: Name of the plant (e.g., 'Plant A')

    Returns:
        Path object for the plant's model directory
    """
    normalized_name = normalize_plant_name(plant_name)
    if normalized_name not in PLANT_DIR_MAP:
        raise ValueError(f"Unknown plant for model directory: {plant_name}")
    return Path(PLANT_DIR_MAP[normalized_name])


def load_plant_model(plant_name: str) -> Tuple[Optional[object], Optional[list]]:
    """
    Load the best saved model for a plant and its feature columns.

    The best model is read from the first row of `model_comparison.csv`, which is
    sorted by test R² during training. Falls back to Ridge for older artifacts.

    Args:
        plant_name: Name of the plant (e.g., 'Plant A')

    Returns:
        Tuple of (model, feature_columns) or (None, None) if loading fails
    """
    try:
        model_dir = get_plant_directory(plant_name)
        features_path = model_dir / 'feature_columns.pkl'

        model_filename = 'ridge_model.pkl'
        comparison_path = model_dir / 'model_comparison.csv'
        if comparison_path.exists():
            comparison_df = pd.read_csv(comparison_path)
            if not comparison_df.empty:
                best_model_name = comparison_df.iloc[0]['Model']
                model_filename = MODEL_FILE_MAP.get(best_model_name, model_filename)

        model_path = model_dir / model_filename
        if not model_path.exists():
            print(f"⚠️ Best model not found: {model_path}")
            return None, None

        # Load model
        model = model_training.load_model(str(model_path))

        # Load feature columns
        if features_path.exists():
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
        else:
            print(f"⚠️ Feature columns not found: {features_path}")
            feature_columns = None

        print(f"✓ Loaded {model_filename} for {plant_name}")
        return model, feature_columns

    except Exception as e:
        print(f"❌ Error loading model for {plant_name}: {e}")
        return None, None


def load_all_models_comparison(plant_name: str) -> Optional[pd.DataFrame]:
    """
    Load model comparison data for the selected plant.

    Args:
        plant_name: Name of the plant (e.g., 'Plant A')

    Returns:
        DataFrame with model comparison metrics or None if not found
    """
    try:
        model_dir = get_plant_directory(plant_name)
        comparison_path = model_dir / 'model_comparison.csv'

        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            print(f"✓ Loaded model comparison for {plant_name}")
            return df
        else:
            print(f"⚠️ Model comparison not found: {comparison_path}")
            return None

    except Exception as e:
        print(f"❌ Error loading model comparison for {plant_name}: {e}")
        return None


def load_multiple_models(plant_name: str) -> Tuple[Dict, Optional[list]]:
    """
    Load all available trained models for the selected plant.

    Args:
        plant_name: Name of the plant (e.g., 'Plant A')

    Returns:
        Tuple of (models_dict, feature_columns)
        - models_dict: Dictionary {model_name: model_object}
        - feature_columns: List of feature column names
    """
    models_dict = {}
    try:
        models_dir = get_plant_directory(plant_name)
    except Exception as e:
        print(f"❌ Error loading models for {plant_name}: {e}")
        return models_dict, None

    # Try to load different model types
    model_files = {
        'Ridge': 'ridge_model.pkl',
        'RandomForest': 'randomforest_model.pkl',
        'GradientBoosting': 'gradientboosting_model.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }

    for model_name, model_file in model_files.items():
        model_path = models_dir / model_file
        if model_path.exists():
            try:
                models_dict[model_name] = model_training.load_model(str(model_path))
                print(f"✓ Loaded {model_name} for {plant_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {model_name}: {e}")
                continue

    # Load feature columns
    features_path = models_dir / 'feature_columns.pkl'
    if features_path.exists():
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        print(f"✓ Loaded feature columns for {plant_name}")
    else:
        print(f"⚠️ Feature columns not found: {features_path}")
        feature_columns = None

    print(f"✓ Loaded {len(models_dict)} models for {plant_name}")
    return models_dict, feature_columns


def load_plant_metadata(plant_name: str) -> Optional[Dict]:
    """
    Load plant metadata (capacity, plant_id, etc.).

    Args:
        plant_name: Name of the plant (e.g., 'Plant A')

    Returns:
        Dictionary with plant metadata or None if not found
    """
    try:
        model_dir = get_plant_directory(plant_name)
        metadata_path = model_dir / 'plant_info.pkl'

        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✓ Loaded metadata for {plant_name}")
            return metadata
        else:
            print(f"⚠️ Metadata not found: {metadata_path}")
            return None

    except Exception as e:
        print(f"❌ Error loading metadata for {plant_name}: {e}")
        return None
