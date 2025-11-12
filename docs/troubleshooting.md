# Troubleshooting 🔧

## Common Issues & Solutions

### "This solver needs samples of at least 2 classes" Error

**Symptoms:**

```
This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
```

**Causes & Solutions:**

1. **Small sample size without stratification**

   ```bash
   # ❌ Only gets normal requests
   python main.py --nrows 1000

   # ✅ Ensures balanced classes
   python main.py --nrows 2000
   ```

2. **Data quality issue**
   - Check if CSV has both classes: `df['classification'].value_counts()`
   - Verify stratification is working in data loader

### "File not found" Errors

**Symptoms:**

```
FileNotFoundError: Dataset file not found: data/csic_database.csv
```

**Solutions:**

1. **Check file location:**

   ```bash
   ls -la data/
   # Should show: csic_database.csv
   ```

2. **Download the dataset:**

   - Original source: [CSIC 2010 Dataset](http://www.isi.csic.es/dataset/)
   - Place in `data/csic_database.csv`

3. **Check permissions:**
   ```bash
   chmod 644 data/csic_database.csv
   ```

### Memory Errors

**Symptoms:**

```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Use smaller sample:**

   ```bash
   python main.py --nrows 5000
   ```

2. **Free memory:**

   ```bash
   # Close other programs
   # Restart Python session
   ```

3. **Check system resources:**
   ```bash
   free -h  # Linux
   # or
   top      # Check memory usage
   ```

### Poor Model Performance

**Symptoms:**

- ROC AUC < 0.7
- Accuracy < 60%

**Possible causes:**

1. **Imbalanced data:**

   - Solution: Use stratified sampling

2. **Poor features:**

   - Solution: Check feature engineering code

3. **Wrong hyperparameters:**

   - Solution: Adjust model parameters

4. **Overfitting:**
   - Solution: Use cross-validation, simpler model

### Import Errors

**Symptoms:**

```
ModuleNotFoundError: No module named 'src'
```

**Solutions:**

1. **Run from correct directory:**

   ```bash
   cd /path/to/your/project
   python main.py
   ```

2. **Check virtual environment:**

   ```bash
   source .venv/bin/activate
   which python  # Should point to .venv/bin/python
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Plotting Errors

**Symptoms:**

```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```

**Cause:** Running on server without display

**Solutions:**

1. **Use non-interactive backend:**

   ```python
   import matplotlib
   matplotlib.use('Agg')  # Before importing pyplot
   ```

2. **Save plots without displaying:**
   - The code already does this correctly

## 🔍 Debugging Steps

### Step 1: Check Environment

```bash
# Verify setup
which python
python --version
pip list | grep scikit-learn

# Check files
ls -la data/
ls -la models/
ls -la results/
```

### Step 2: Test Components Individually

```python
# Test data loading
from src.data.data_loader import CSICDataLoader
loader = CSICDataLoader()
df = loader.load_data(nrows=100)
print("Data loaded successfully:", df.shape)

# Test feature engineering
from src.features.feature_engineer import HTTPFeatureEngineer
engineer = HTTPFeatureEngineer()
features = engineer.extract_features(df.head(10))
print("Features extracted:", features.shape)

# Test model
from src.models.logistic_regression_model import VulnerabilityDetector
model = VulnerabilityDetector()
model.fit(features, df.head(10)['target'])
print("Model trained successfully")
```

### Step 3: Check Data Quality

```python
import pandas as pd

# Load raw data
df = pd.read_csv('data/csic_database.csv', nrows=1000)

# Check basic stats
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Classification distribution:")
print(df['classification'].value_counts())
print("Missing values:")
print(df.isnull().sum())
```

### Step 4: Monitor Resource Usage

```bash
# During training
top -p $(pgrep python)  # Memory/CPU usage
watch -n 1 free -h     # Memory over time
```

## 🚨 Advanced Issues

### Model Not Converging

**Symptoms:**

```
ConvergenceWarning: lbfgs failed to converge
```

**Solutions:**

```python
# Increase iterations
model = LogisticRegression(max_iter=2000)

# Change solver
model = LogisticRegression(solver='liblinear')

# Scale features better
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Feature Scaling Issues

**Symptoms:**

- Poor performance
- Features with very different scales

**Check:**

```python
# Check feature scales
print(X.describe())
# Look for features with very different ranges

# Solution: Ensure StandardScaler is applied
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Class Imbalance Problems

**Symptoms:**

- High accuracy but low recall
- Model predicts majority class most of the time

**Solutions:**

```python
# Use class weighting
model = LogisticRegression(class_weight='balanced')

# Or manual weighting
class_weights = {0: 1, 1: 5}  # Give more weight to minority class
model = LogisticRegression(class_weight=class_weights)
```

## 📞 Getting Help

### Debug Information to Provide

When asking for help, include:

1. **Full error traceback**
2. **Your command:**

   ```bash
   python main.py --nrows 2000
   ```

3. **Environment info:**

   ```bash
   python --version
   pip list | grep -E "(scikit-learn|pandas|numpy)"
   ```

4. **Data info:**

   ```python
   import pandas as pd
   df = pd.read_csv('data/csic_database.csv', nrows=1000)
   print("Shape:", df.shape)
   print("Columns:", len(df.columns))
   print("Classes:", df['classification'].nunique())
   ```

5. **System resources:**
   ```bash
   free -h
   df -h
   ```

### Common Fixes Summary

| Issue            | Quick Fix                                                 |
| ---------------- | --------------------------------------------------------- |
| Only one class   | Use `--nrows 2000` (stratified sampling)                  |
| Memory error     | Use smaller `--nrows` or add RAM                          |
| Poor performance | Check feature engineering, use cross-validation           |
| Import errors    | Activate virtual environment: `source .venv/bin/activate` |
| No output        | Add `print()` statements or check logging level           |

## 🎯 Performance Tuning

### Speed Optimizations

```python
# Use faster solver
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Reduce features
# Remove low-importance features based on feature_importance

# Use parallel processing (if available)
# sklearn automatically uses multiple cores for some operations
```

### Memory Optimizations

```python
# Process in batches
for batch in pd.read_csv('data.csv', chunksize=10000):
    process_batch(batch)

# Use sparse matrices for large datasets
from scipy.sparse import csr_matrix

# Delete unused variables
del large_dataframe
import gc
gc.collect()
```

## 📊 Validation Checks

### Before Running Pipeline

```python
# Check data exists and is readable
assert Path('data/csic_database.csv').exists()

# Check virtual environment
import sys
assert 'venv' in sys.executable

# Check dependencies
import sklearn, pandas, numpy
print("All dependencies available")
```

### After Training

```python
# Sanity checks
assert results['roc_auc'] > 0.5  # Better than random
assert len(feature_importance) > 0  # Features extracted
assert model.pipeline is not None  # Model trained

print("All validation checks passed!")
```


