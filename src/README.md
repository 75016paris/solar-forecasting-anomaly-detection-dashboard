# Solar Production Prediction - Modular Architecture

This folder modular Python code for solar production prediction.

## 📁 **Module Structure**

```
src/
├── __init__.py                # Package initialization
├── config.py                  # Central configuration (paths, parameters)
├── data_loader.py            # Data loading and preprocessing
├── feature_engineering.py    # Feature creation (24 optimized features)
├── model_training.py         # Model training with hyperparameter tuning
├── evaluation.py             # Model evaluation and metrics
└── visualization.py          # Plotting functions
```

---

## 🚀 **Quick Start**

### **Example 1: Train models from scratch**

```python
from src import data_loader, feature_engineering, model_training, evaluation

# 1. Load data
df = data_loader.load_complete_dataset()

# 2. Create features
df = feature_engineering.create_all_features(df)

# 3. Split train/test
train_df, test_df = data_loader.split_train_test(df)

# 4. Prepare features and target
feature_cols = feature_engineering.get_feature_columns(df)
X_train, y_train = feature_engineering.prepare_features_target(train_df, feature_cols)
X_test, y_test = feature_engineering.prepare_features_target(test_df, feature_cols)

# 5. Train all models
results = model_training.train_all_models(X_train, y_train, X_test, y_test)

# 6. Compare models
comparison_df = evaluation.create_comparison_dataframe(results)
evaluation.print_model_comparison(comparison_df)
```

### **Example 2: Train only specific models**

```python
# Train only Ridge and Random Forest
results = model_training.train_all_models(
    X_train, y_train, X_test, y_test,
    models_to_train=['Ridge', 'RandomForest']
)
```

### **Example 3: Load and use saved model**

```python
from src import model_training, data_loader

# Load saved model
model = model_training.load_model('models/ridge_model.pkl')

# Load new data and predict
df = data_loader.load_predictions()  # Loads pre-computed predictions
```

---

## 📊 **Module Details**

### **`config.py`** - Configuration

Centralizes all configuration:
- Plant configuration (location, capacity, etc.)
- File paths
- ML parameters (train/test split date, CV splits, etc.)
- Feature groups
- Hyperparameter grids

**Usage:**
```python
from src.config import PLANT_CONFIG, ML_CONFIG, DATA_PATHS

print(PLANT_CONFIG['name'])  # 'Plant A'
print(ML_CONFIG['train_test_split_date'])  # '2024-09-01'
```

### **`data_loader.py`** - Data Loading

Functions for loading generation and weather data:
- `load_generation_data()` - Load 5-min generation, resample to hourly
- `load_weather_data()` - Load weather data
- `load_complete_dataset()` - Main function: loads and merges everything
- `split_train_test()` - Split data by date

**Usage:**
```python
from src import data_loader

# Load complete dataset
df = data_loader.load_complete_dataset()

# Split into train/test
train_df, test_df = data_loader.split_train_test(df)
```

### **`feature_engineering.py`** - Feature Creation

Creates the optimized 24-feature set:
- Solar position (elevation, azimuth, GHI)
- Temporal features (cyclical encodings)
- Weather features (cloud_impact, rain, etc.)
- Historical features (lags, rolling averages)

**IMPORTANT:** After extensive testing, we found that adding more weather features (humidity, wind, pressure) **DEGRADED** performance. The current 24-feature set is optimal.

**Usage:**
```python
from src import feature_engineering

# Create all features
df = feature_engineering.create_all_features(df)

# Get feature column names
feature_cols = feature_engineering.get_feature_columns(df)

# Prepare X, y for modeling
X, y = feature_engineering.prepare_features_target(df, feature_cols)
```

### **`model_training.py`** - Model Training

Train multiple ML models with hyperparameter tuning:
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost (if installed)
- SVR

**Usage:**
```python
from src import model_training

# Train all models
results = model_training.train_all_models(X_train, y_train, X_test, y_test)

# Train specific model
ridge_result = model_training.train_ridge(X_train, y_train, X_test, y_test)

# Save/load models
model_training.save_model(model, 'models/my_model.pkl')
model = model_training.load_model('models/my_model.pkl')
```

### **`evaluation.py`** - Model Evaluation

Calculate metrics and compare models:
- `evaluate_model()` - Calculate MAE, RMSE, R², WAPE
- `create_comparison_dataframe()` - Create comparison table
- `get_best_model()` - Get best performing model
- `detect_anomalies()` - Detect production anomalies

**Usage:**
```python
from src import evaluation

# Create comparison
comparison_df = evaluation.create_comparison_dataframe(results)
evaluation.print_model_comparison(comparison_df)

# Get best model
best_name, best_result = evaluation.get_best_model(results)

# Detect anomalies
residuals = evaluation.calculate_residuals(y_test, predictions)
anomalies, threshold = evaluation.detect_anomalies(residuals)
```

### **`visualization.py`** - Plotting

Reusable plotting functions:
- `plot_model_comparison()` - Compare model performance
- `plot_three_line_comparison()` - Actual vs ML vs Clear-Sky
- `plot_feature_importance()` - Tree-based feature importance
- `plot_ridge_coefficients()` - Ridge coefficient analysis

**Usage:**
```python
from src import visualization

# Plot model comparison
visualization.plot_model_comparison(comparison_df)

# Plot predictions
visualization.plot_three_line_comparison(test_df, predictions, 'Ridge')

# Plot feature importance
visualization.plot_feature_importance(results, feature_cols)
```

---

## 🎯 **Benefits of Modular Architecture**

| Before | After |
|--------|-------|
| 1 notebook with 3,594 lines | 6 modules of ~200-400 lines each |
| Code duplicated in notebook & app | Code shared and reusable |
| Hard to test | Each function testable |
| Hard to maintain | Clear separation of concerns |
| Large git diffs | Targeted changes |

---

## 📝 **Example: Complete Pipeline**

See `notebooks/example_usage.ipynb` for a complete example notebook using the modules.

```python
# Complete pipeline in ~20 lines

from src import (
    data_loader,
    feature_engineering,
    model_training,
    evaluation,
    visualization
)

# 1. Load and prepare data
df = data_loader.load_complete_dataset()
df = feature_engineering.create_all_features(df)

# 2. Split and prepare
train_df, test_df = data_loader.split_train_test(df)
feature_cols = feature_engineering.get_feature_columns(df)
X_train, y_train = feature_engineering.prepare_features_target(train_df, feature_cols)
X_test, y_test = feature_engineering.prepare_features_target(test_df, feature_cols)

# 3. Train models
results = model_training.train_all_models(X_train, y_train, X_test, y_test)

# 4. Evaluate
comparison_df = evaluation.create_comparison_dataframe(results)
evaluation.print_model_comparison(comparison_df)

# 5. Visualize
visualization.plot_model_comparison(comparison_df)
```

---

## 🔧 **Configuration**

All configuration is in `config.py`. To change settings:

```python
# Edit src/config.py

ML_CONFIG = {
    'train_test_split_date': '2024-09-01',  # Change split date
    'tune_hyperparameters': False,           # Disable tuning for speed
    'cv_splits': 5,                          # More CV folds
    # ...
}
```

---

## 📚 **Documentation**

Each module has detailed docstrings. Use Python's help:

```python
from src import data_loader
help(data_loader.load_complete_dataset)
```

---

## 🧪 **Testing**

Each module can be tested independently:

```python
# Test data loading
from src import data_loader
df = data_loader.load_complete_dataset()
assert len(df) > 0, "Data loading failed"

# Test feature engineering
from src import feature_engineering
df_with_features = feature_engineering.create_all_features(df)
assert 'elevation' in df_with_features.columns, "Features not created"
```

---

## 🆘 **Troubleshooting**

**Import errors:**
```python
# Run from project root
import sys
sys.path.append('/path/to/solar-forecasting-dashboard')
from src import config
```

**Module not found:**
```bash
# Install required packages
pip install pandas numpy scikit-learn pvlib matplotlib seaborn
```

---

## 📖 **Additional Resources**

- Streamlit app: `app_solar_monitoring_enhanced.py`
- Feature analysis: See comments in `feature_engineering.py` for why certain features were removed
