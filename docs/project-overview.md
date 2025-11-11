# Project Overview 🏗️

## The Big Picture

This project is a **machine learning pipeline** that automatically detects web attacks. Think of it as a security system that learns from examples.

```
Raw HTTP Requests
        ↓
Feature Extraction (Convert to numbers)
        ↓
Machine Learning Model (Learn patterns)
        ↓
Trained Security System (Detect attacks)
```

## 🔄 The Complete Pipeline

### Step 1: Data Loading 📥

**File**: `src/data/data_loader.py`
**What it does**: Reads the CSV file and prepares the data

```python
# Raw data from CSV:
# Method, URL, User-Agent, Content, Classification
# GET, /login.php, Mozilla/5.0..., username=admin, 0

# Becomes:
# DataFrame with clean columns, balanced classes
```

### Step 2: Feature Engineering 🔧

**File**: `src/features/feature_engineer.py`
**What it does**: Converts text into numbers the computer can learn from

```python
# HTTP Request Text:
# "GET /admin.php?cmd=DROP TABLE users"

# Becomes Features:
# url_length: 35
# has_sql_injection_patterns: 1
# method_is_standard: 1
# suspicious_pattern_score: 2
```

### Step 3: Model Training 🧠

**File**: `src/models/logistic_regression_model.py`
**What it does**: The actual machine learning

```python
# Features: (1600 samples × 39 features)
# Labels: 0=normal, 1=attack (1600 labels)

# Result: Trained model that predicts new requests
```

### Step 4: Pipeline Orchestration 🎯

**File**: `src/models/trainer.py`
**What it does**: Coordinates all the steps

```python
trainer = ModelTrainer()
results = trainer.run_full_pipeline()  # Does everything automatically
```

### Step 5: Main Entry Point 🚀

**File**: `main.py`
**What it does**: Command-line interface

```bash
python main.py --nrows 2000  # Run with 2000 samples
```

## 📊 Understanding the Output

### Console Output

```
2025-11-11 23:00:54 - src.models.trainer - INFO - Step 1/5: Loading and preprocessing data...
2025-11-11 23:00:54 - src.models.trainer - INFO - Step 2/5: Extracting features...
2025-11-11 23:00:54 - src.models.trainer - INFO - Step 3/5: Training model...
2025-11-11 23:00:54 - src.models.trainer - INFO - Step 4/5: Evaluating model...
2025-11-11 23:00:54 - src.models.trainer - INFO - Step 5/5: Saving artifacts...

ROC AUC Score: 0.8036
Accuracy: 0.7300
Precision (Anomalous): 0.7212
Recall (Anomalous): 0.7500
F1-Score (Anomalous): 0.7353

Confusion Matrix:
  True Normal:        142    False Positive:     58
  False Negative:      50    True Anomalous:    150
```

### Saved Files

#### `models/vulnerability_detector.pkl`

```python
# Contains trained model + metadata
model_data = {
    'pipeline': trained_pipeline,      # The actual ML model
    'feature_names': [...],            # Names of 39 features
    'random_state': 42,                # For reproducibility
    'max_iter': 1000                   # Training configuration
}
```

#### `results/evaluation_plots.png`

- **Confusion Matrix**: Shows correct/incorrect predictions
- **ROC Curve**: Shows discrimination ability
- **Feature Importance**: Top 20 most influential features

#### `results/evaluation_results.json`

```json
{
  "evaluation_metrics": {
    "roc_auc": 0.8036,
    "classification_report": {
      "accuracy": 0.73,
      "1": { "precision": 0.7212, "recall": 0.75, "f1-score": 0.7353 }
    },
    "confusion_matrix": [
      [142, 58],
      [50, 150]
    ]
  },
  "feature_importance_top_20": [
    { "feature": "suspicious_pattern_score", "coefficient": 1.234 },
    { "feature": "url_has_suspicious_chars", "coefficient": 0.987 }
  ]
}
```

## 🎯 Key Design Decisions

### Why Logistic Regression?

- **Interpretability**: Provides feature importance through coefficients
- **Efficiency**: Fast training and prediction on large datasets
- **Probabilistic Output**: Gives confidence scores (0.0-1.0)
- **Linearity**: Well-suited for the engineered features
- **Robustness**: Handles feature scaling and outliers effectively
- **Baseline Performance**: Establishes benchmark for comparison with complex models

### Why 39 Features?

- **Comprehensive**: Covers URL, content, headers, patterns
- **Not too many**: Prevents overfitting
- **Interpretable**: Each feature has clear meaning

### Why Stratified Sampling?

- **Balanced classes**: Model learns both patterns
- **Better learning**: Model doesn't favor majority class
- **Realistic testing**: Test set matches training distribution

## 🔧 Configuration System

**File**: `src/utils/config.py`

- **Logging**: Centralized logging setup
- **Exceptions**: Custom error types for different failures
- **Settings**: Default configuration values

## 📊 Results Interpretation

When you run the system, you get:

### Confusion Matrix

```
Predicted →    Normal (0)    Attack (1)
Actually ↓
Normal (0)        142          58         ← True Negatives | False Positives
Attack (1)         50          150        ← False Negatives | True Positives

Total Predictions: 400
Correct Predictions: 292 (73.0%)
```

### Key Metrics

- **ROC AUC (0.80)**: How well it distinguishes attacks from normal traffic
- **Accuracy (0.73)**: Overall correct predictions
- **Precision (0.72)**: When it flags attack, how often right?
- **Recall (0.75)**: How many real attacks does it catch?

## 🚨 Error Handling

### 5 Levels of Error Protection

1. **Input validation** (wrong command-line arguments)

   ```python
   if args.test_size <= 0 or args.test_size >= 1:
       parser.error("test-size must be between 0 and 1")
   ```

2. **File system** (missing files, permission issues)

   ```python
   if not data_path.exists():
       raise FileNotFoundError(f"Dataset not found: {data_path}")
   ```

3. **Data quality** (corrupted CSV, missing columns)

   ```python
   if col not in df.columns:
       raise ValueError(f"Missing required column: {col}")
   ```

4. **Pipeline state** (calling methods in wrong order)

   ```python
   self.validate_pipeline_state('features_extracted')
   ```

5. **Unexpected errors** (with detailed logging)
   ```python
   except Exception as e:
       logger.error(f"Unexpected error: {e}")
       raise PipelineError(f"Pipeline failed: {e}") from e
   ```

## 🎮 How to Experiment

### Try Different Settings

```bash
# Small dataset (fast but less accurate)
python main.py --nrows 1000

# Save model for later use
python main.py --save-model models/my_model.pkl

# See which features matter most
python main.py --feature-importance
```

### Modify the Code

- **Add features**: Edit `src/features/feature_engineer.py`
- **Change algorithm**: Modify `src/models/logistic_regression_model.py`
- **Adjust parameters**: Change values in `src/models/trainer.py`

## 📈 Pipeline Metadata

The pipeline collects metadata for reproducibility:

```python
pipeline_metadata = {
    'training_samples': 1600,
    'test_samples': 400,
    'num_features': 39,
    'model_type': 'LogisticRegression',
    'feature_engineering_version': '2.0',
    'stratified_sampling': True,
    'random_state': 42
}
```

## 🎯 Why Pipeline Design Matters

### Advantages

1. **Reproducible**: Same inputs → same results
2. **Maintainable**: Changes isolated to specific steps
3. **Testable**: Each step can be tested independently
4. **Reusable**: Same pipeline for training/evaluation
5. **Debuggable**: Clear error messages and logging

### Common Pipeline Patterns

- **Data Loading** → **Preprocessing** → **Feature Engineering** → **Model Training** → **Evaluation**
- Each step is a pure function (same input → same output)
- Steps communicate via well-defined interfaces

## 🔧 Customizing the Pipeline

### Add New Features

```python
# In src/features/feature_engineer.py
def _extract_custom_features(self, df):
    features['custom_feature'] = df['URL'].apply(custom_logic)
    return features
```

### Change Model Algorithm

```python
# In src/models/trainer.py
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
```

### Add Evaluation Metrics

```python
# In src/models/logistic_regression_model.py
def evaluate(self, X_test, y_test):
    # Add custom metrics
    results['custom_metric'] = calculate_custom_metric(y_true, y_pred)
    return results
```

---

**Next**: Let's dive deep into each component, starting with [Data Loading](data-loading.md)!
