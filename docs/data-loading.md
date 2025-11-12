# Data Loading 📥

## What is Data Loading?

Data loading is the **first step** in our machine learning pipeline. It takes raw data from files and converts it into a format the computer can work with.

## 📄 The CSIC 2010 Dataset

### What We Have

- **61,065 HTTP requests** (real website traffic)
- **36,000 Normal requests** (safe browsing)
- **25,065 Anomalous requests** (attacks)

### Raw CSV Format

The data comes as a CSV file with columns like:

```
Method, User-Agent, Pragma, Cache-Control, Accept, Accept-encoding,
Accept-charset, language, host, cookie, content-type, connection,
length, content, classification, URL
```

## 🔧 The DataLoader Class

### File: `src/data/data_loader.py`

```python
class CSICDataLoader:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_data(self, nrows=None, stratified_sample=True):
        # 1. Read CSV file
        # 2. Handle column mapping
        # 3. Stratified sampling (if needed)
        # Returns: DataFrame with clean data

    def preprocess_data(self, df):
        # 1. Convert classification to numbers
        # 2. Handle missing values
        # 3. Clean column names
        # Returns: Ready-to-use DataFrame

    def get_train_test_split(self, df, test_size=0.2):
        # 1. Split into train/test sets
        # 2. Maintain class balance
        # Returns: train_df, test_df
```

## 📊 Step-by-Step Data Processing

### 1. Reading the CSV

```python
# Raw CSV has headers, but we need to map column names
df = pd.read_csv('data/csic_database.csv')

# Result: DataFrame with proper column names
print(df.columns)
# Index(['Method', 'User-Agent', 'Pragma', ..., 'URL'], dtype='object')
```

### 2. Understanding the Classification Column

```python
# The 'classification' column contains our labels:
# 0 = Normal (safe) request
# 1 = Anomalous (attack) request

print(df['classification'].value_counts())
# 0    36000  ← Normal requests
# 1    25065  ← Attack requests
```

### 3. Converting to Target Variable

```python
# We create a 'target' column for machine learning:
df['target'] = df['classification'].astype(int)

# Now we have:
# - Features: Method, URL, User-Agent, etc.
# - Target: 0 (normal) or 1 (attack)
```

### 4. Handling Missing Values

```python
# Many columns can be empty, we fill them with empty strings:
df['content'] = df['content'].fillna('')
df['cookie'] = df['cookie'].fillna('')

# Content length becomes 0 if missing:
df['content_length'] = df['content_length'].fillna(0)
```

## 🎯 Stratified Sampling (The Smart Part!)

### The Problem

If you take the first 2000 rows, you get mostly normal requests:

```python
df.head(2000)['classification'].value_counts()
# 0    1995  ← Almost all normal!
# 1       5  ← Very few attacks!
```

**Result**: Model learns "everything is normal" - useless!

### The Solution: Stratified Sampling

```python
# Instead of taking first N rows, we take equal amounts of each class:

# Separate normal and attack requests
normal_requests = df[df['classification'] == 0]
attack_requests = df[df['classification'] == 1]

# Take 1000 of each
sampled_normal = normal_requests.sample(n=1000, random_state=42)
sampled_attack = attack_requests.sample(n=1000, random_state=42)

# Combine them
balanced_df = pd.concat([sampled_normal, sampled_attack])

# Result: Perfect balance!
print(balanced_df['target'].value_counts())
# 0    1000
# 1    1000
```

## 🧮 Train/Test Split

### Why Split Data?

- **Training set**: Used to teach the model (80% of data)
- **Test set**: Used to evaluate performance (20% of data)
- **Never show test data during training!**

### Stratified Split

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    balanced_df,
    test_size=0.2,        # 20% for testing
    random_state=42,      # Reproducible results
    stratify=balanced_df['target']  # Keep class balance
)

print("Training set:", len(train_df))  # 1600 samples
print("Test set:", len(test_df))       # 400 samples
```

## 📋 What the DataLoader Returns

After processing, you get clean DataFrames ready for machine learning:

### Training Data (train_df)

```python
# 1600 rows × 18 columns
train_df.head()
#   Method  User-Agent  ...  content  target
# 0    GET   Mozilla/5.0 ...         0
# 1   POST   Mozilla/5.0 ...  user=admin  1
```

### Test Data (test_df)

```python
# 400 rows × 18 columns
# Same structure, but model never sees this during training
```

## 🔍 Data Quality Checks

The loader includes several validation checks:

```python
# 1. File exists
if not self.data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {self.data_path}")

# 2. Required columns present
required_cols = ['Method', 'URL', 'classification']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# 3. Data not empty
if len(df) == 0:
    raise ValueError("Dataset is empty")

# 4. Classes present
unique_targets = df['target'].unique()
if len(unique_targets) < 2:
    raise ValueError("Need at least 2 classes for classification")
```

## 📊 Understanding Your Data

After loading, you can explore:

```python
# Basic statistics
print("Dataset shape:", df.shape)
print("Class distribution:")
print(df['target'].value_counts(normalize=True))

# Sample of each class
print("\nNormal request sample:")
print(df[df['target'] == 0].head(1)[['Method', 'URL']])

print("\nAttack request sample:")
print(df[df['target'] == 1].head(1)[['Method', 'URL']])
```

## 🎯 Why This Matters

**Good data loading = Good model performance**

- **Balanced classes**: Model learns both patterns
- **Clean data**: No missing values confuse the algorithm
- **Proper splits**: Realistic evaluation of performance
- **Validation**: Catches problems early

## 🚨 Common Issues & Solutions

### Issue: "Only one class found"

**Cause**: Not using stratified sampling
**Solution**: Use `stratified_sample=True` or `--nrows 2000`

### Issue: Memory errors

**Cause**: Loading full 61k dataset
**Solution**: Use `--nrows 5000` for testing

### Issue: "Column not found"

**Cause**: CSV format changed
**Solution**: Check column names with `df.columns`

## 🔧 Configuration Options

```python
# In the DataLoader constructor
loader = CSICDataLoader(
    data_path='data/csic_database.csv'  # Path to CSV file
)

# When loading data
df = loader.load_data(
    nrows=2000,             # Limit rows (for testing)
    stratified_sample=True  # Ensure balanced classes
)
```

## 📈 What Happens Next

After data loading, the clean DataFrames go to:

1. **Feature Engineering**: Convert text to numbers
2. **Model Training**: Learn patterns from the data
3. **Evaluation**: Test on unseen data

---

**Next**: [Feature Engineering](feature-engineering.md) - Converting text into numbers the computer can learn from!
