# Training Pipeline 🎯

## What is a Training Pipeline?

A training pipeline is an **automated workflow** that takes raw data and produces a trained machine learning model. It's like a factory assembly line for AI.

```
Raw Data → Clean Data → Features → Model → Evaluation
```

## 🔄 The 5-Step Pipeline

### Step 1: Load & Preprocess Data 📥

**File**: `src/data/data_loader.py`
**What it does**: Reads CSV and prepares clean data

```python
logger.info("Step 1/5: Loading and preprocessing data...")
self.load_data(nrows)
```

**Result**:

- `train_df`: 1600 samples for training
- `test_df`: 400 samples for testing

### Step 2: Extract Features 🔧

**File**: `src/features/feature_engineer.py`
**What it does**: Converts text to 38 numerical features

```python
logger.info("Step 2/5: Extracting features...")
self.extract_features()
```

**Result**:

- `X_train`: (1600, 38) feature matrix
- `y_train`: (1600,) target labels

### Step 3: Train Model 🧠

**File**: `src/models/logistic_regression_model.py`
**What it does**: Learns patterns from training data

```python
logger.info("Step 3/5: Training model...")
self.train_model()
```

**Result**:

- Trained `LogisticRegression` model

### Step 4: Evaluate Model 📊

**File**: `src/models/logistic_regression_model.py`
**What it does**: Tests model on unseen data

```python
logger.info("Step 4/5: Evaluating model...")
evaluation_results = self.evaluate_model()
```

**Result**:

- ROC AUC, accuracy, precision, recall, F1-score
- Confusion matrix

### Step 5: Save Artifacts 💾

**What it does**: Saves model and results for later use

```python
logger.info("Step 5/5: Saving artifacts...")
model.save_model('models/vulnerability_detector.pkl')
# Save plots and metrics
```

## 🎯 Pipeline State Validation

### Why State Validation?

Prevents calling methods in wrong order:

```python
trainer = ModelTrainer()

# ❌ Wrong order - will fail
trainer.train_model()  # Error: No features extracted yet!

# ✅ Correct order
trainer.load_data()
trainer.extract_features()
trainer.train_model()  # Now it works
```

### Validation Code

```python
def validate_pipeline_state(self, required_stage: str):
    stages = {
        'data_loaded': self.train_df is not None,
        'features_extracted': self.X_train is not None,
        'model_trained': self.model.pipeline is not None
    }

    stage_order = ['data_loaded', 'features_extracted', 'model_trained']
    required_index = stage_order.index(required_stage)

    for i, stage in enumerate(stage_order[:required_index + 1]):
        if not stages[stage]:
            raise ValueError(f"Pipeline stage '{stage}' not completed")
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

Contains trained model with:

- Learned weights/coefficients
- Feature names and scaling parameters
- Model configuration

#### `results/evaluation_plots.png`

- Confusion Matrix visualization
- ROC Curve showing discriminative ability
- Feature Importance bar chart

#### `results/evaluation_results.json`

```json
{
  "evaluation_metrics": {
    "roc_auc": 0.8036,
    "classification_report": {
      "accuracy": 0.73,
      "1": { "precision": 0.7212, "recall": 0.75, "f1-score": 0.7353 }
    }
  },
  "feature_importance_top_20": [
    { "feature": "suspicious_pattern_score", "coefficient": 1.234 }
  ]
}
```

## 🎮 Command Line Options

### Basic Usage

```bash
# Use all data (61,000 samples - takes longer)
python main.py

# Quick test with smaller dataset
python main.py --nrows 2000

# Save model for later use
python main.py --save-model models/my_model.pkl

# See which features matter most
python main.py --feature-importance
```

### Advanced Options

```bash
# Change train/test split ratio
python main.py --test-size 0.3  # 70% train, 30% test

# Different random seed
python main.py --random-state 123
```

## 🚨 Error Handling

### 5 Levels of Error Protection

1. **Input validation** (wrong command-line arguments)
2. **File system** (missing files, permission issues)
3. **Data quality** (corrupted CSV, missing columns)
4. **Pipeline state** (calling methods in wrong order)
5. **Unexpected errors** (with detailed logging)

## 🔧 Customizing the Pipeline

### Add New Features

```python
# Modify src/features/feature_engineer.py
def _extract_custom_features(self, df):
    features['custom_feature'] = df['URL'].apply(custom_logic)
    return features
```

### Change Model Algorithm

```python
# Modify src/models/trainer.py
from sklearn.ensemble import RandomForestClassifier
self.model = RandomForestClassifier(n_estimators=100, random_state=random_state)
```

### Add Evaluation Metrics

```python
# Modify src/models/logistic_regression_model.py
def evaluate(self, X_test, y_test):
    results['custom_metric'] = calculate_custom_metric(y_true, y_pred)
    return results
```

## 📈 Pipeline Metadata

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

---

**Next**: [Results & Evaluation](results.md) - Understanding what the numbers mean!
