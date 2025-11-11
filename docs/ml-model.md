# The Machine Learning Model 🧠

## What is Logistic Regression?

Logistic Regression is a **classification algorithm** that predicts whether something belongs to one category or another.

### Simple Analogy

Imagine sorting emails into "Important" vs "Spam":

**Traditional Rules:**

- If email contains "URGENT" → Important
- If email contains "Free money" → Spam

**Logistic Regression:**

- Learns from 1000 examples of important/spam emails
- Finds patterns: "Word frequencies + sender reputation + time of day"
- Creates a formula that scores each email from 0.0 to 1.0
- If score > 0.5 → Spam, else → Important

## 📊 How It Works Mathematically

### The Sigmoid Function

Logistic regression uses the **sigmoid function** to convert any number into a probability (0.0 to 1.0):

```
f(x) = 1 / (1 + e^(-x))
```

Where `x` is a weighted sum of features:

```
x = (feature1 × weight1) + (feature2 × weight2) + ... + bias
```

### Example Calculation

```
Features: [url_length=35, has_suspicious_chars=1, content_length=150]
Weights:  [0.01, 2.5, 0.001]
Bias:    -1.2

x = (35 × 0.01) + (1 × 2.5) + (150 × 0.001) + (-1.2)
  = 0.35 + 2.5 + 0.15 - 1.2
  = 1.8

probability = 1 / (1 + e^(-1.8))
           = 1 / (1 + 0.165)
           = 0.858

Prediction: 1 (Attack!) because 0.858 > 0.5
```

## 🏗️ The LogisticRegressionModel Class

### File: `src/models/logistic_regression_model.py`

```python
class VulnerabilityDetector:
    def __init__(self, random_state=42, max_iter=1000):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),        # Normalize features
            ('classifier', LogisticRegression(   # The ML algorithm
                random_state=random_state,
                max_iter=max_iter,
                class_weight='balanced'           # Handle imbalanced classes
            ))
        ])

    def fit(self, X, y):
        # Train the model
        self.pipeline.fit(X, y)

    def predict(self, X):
        # Make predictions (0 or 1)
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        # Get probabilities (0.0 to 1.0)
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        # Calculate performance metrics
        return evaluation_metrics
```

## 🔄 Training Process

### Step 1: Prepare Data

```python
# Features: (1600 samples × 39 features)
X_train = feature_matrix  # Numbers from feature engineering

# Labels: (1600 samples × 1)
y_train = target_labels   # 0=normal, 1=attack
```

### Step 2: Feature Scaling

```python
# Different features have different scales:
# url_length: 10-200
# suspicious_pattern_score: 0-5

# StandardScaler makes them comparable:
# url_length: -2.5 to +3.2
# suspicious_pattern_score: -1.0 to +2.1
```

### Step 3: Optimize Weights

The algorithm tries different weight combinations to minimize errors:

```python
# Goal: Minimize this loss function
loss = -1/n * Σ(y_i * log(ŷ_i) + (1-y_i) * log(1-ŷ_i))

# Where ŷ_i is predicted probability for sample i
# y_i is actual label (0 or 1)
```

### Step 4: Class Balancing

```python
class_weight='balanced'
# Automatically adjusts for imbalanced classes
# Gives more importance to minority class (attacks)
```

## 📊 Model Output

### Predictions

```python
predictions = model.predict(X_test)
# Result: [0, 1, 0, 1, 0, ...] (400 predictions)
```

### Probabilities

```python
probabilities = model.predict_proba(X_test)
# Result: [[0.85, 0.15],    ← 85% normal, 15% attack
#          [0.23, 0.77],    ← 23% normal, 77% attack
#          [0.91, 0.09],    ← 91% normal, 9% attack
#          ...]
```

### Feature Importance

```python
coefficients = model.get_feature_importance()
# Shows which features influence predictions most
# Positive coefficient: Feature increases attack probability
# Negative coefficient: Feature decreases attack probability
```

## 📈 Understanding the Results

### Our Model Performance

```python
ROC AUC: 0.8036    # Excellent (0.8-0.9 is very good)
Accuracy: 0.7300   # Correct 73% of the time
Precision: 0.7212  # When we predict attack, 72% are real attacks
Recall: 0.7500     # We catch 75% of all attacks
F1-Score: 0.7353   # Balanced measure of precision & recall
```

### Confusion Matrix

```
Predicted:     Normal    Attack
Actually:
Normal          142        58     ← True Negatives | False Positives
Attack           50        150    ← False Negatives | True Positives
```

**Confusion Matrix Analysis:**

- **True Negatives (142)**: Normal requests correctly identified
- **False Positives (58)**: Normal requests incorrectly flagged as attacks
- **False Negatives (50)**: Attacks that slipped through undetected
- **True Positives (150)**: Attacks successfully detected

### ROC Curve Explanation

```
ROC AUC = 0.80 means:
- 80% chance model ranks random attack higher than random normal
- Excellent discrimination ability
- Better than random guessing (0.5)
```

### Training Performance

**Convergence and Stability:**

- Model converged successfully within 1000 iterations
- No signs of overfitting on training data
- Stable performance across multiple runs with fixed random seed

---

## 🎯 Why Logistic Regression for Security?

### Advantages

1. **Interpretable**: Can explain why predictions are made
2. **Fast**: Trains quickly on large datasets
3. **Probabilistic**: Gives confidence scores
4. **Feature Importance**: Shows what matters for decisions

### Perfect for Security

- **Explainability**: Need to know WHY something is flagged
- **Speed**: Must analyze requests in real-time
- **Confidence**: Different actions based on risk level
- **Features**: Works well with engineered security features

## 🔧 Model Configuration

### Hyperparameters

```python
LogisticRegression(
    random_state=42,           # Reproducible results
    max_iter=1000,             # Maximum training iterations
    C=1.0,                     # Regularization strength (inverse)
    penalty='l2',              # L2 regularization (ridge)
    class_weight='balanced',   # Handle class imbalance automatically
    solver='lbfgs'             # Optimization algorithm
)
```

**Hyperparameter Details:**

- **`C=1.0`**: Moderate regularization (lower = stronger regularization)
- **`penalty='l2'`**: Ridge regularization (prevents overfitting)
- **`solver='lbfgs'`**: Efficient for medium-sized datasets
- **`class_weight='balanced'`**: Automatically adjusts for class imbalance

### Feature Scaling

```python
StandardScaler()
# Makes all features have mean=0, std=1
# Prevents features with large ranges from dominating
```

## 🚨 Common Issues & Solutions

### Issue: "Convergence Warning"

**Cause**: Model didn't converge within max_iter
**Solution**: Increase max_iter or scale features better

### Issue: Poor Performance

**Cause**: Features not informative enough
**Solution**: Check feature engineering code

### Issue: Overfitting

**Cause**: Too many features, model memorizes training data
**Solution**: Simplify model, use more data, add regularization

## 🎮 Experimenting with the Model

### Try Different Algorithms

```python
# Instead of LogisticRegression, try:
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Random Forest: Good for complex patterns
model = RandomForestClassifier(n_estimators=100, random_state=42)

# SVM: Good for clear boundaries
model = SVC(probability=True, random_state=42)
```

### Adjust Hyperparameters

```python
# Stronger regularization (prevent overfitting)
model = LogisticRegression(C=0.1, penalty='l2', class_weight='balanced')

# L1 regularization (feature selection)
model = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', class_weight='balanced')

# Different solver for larger datasets
model = LogisticRegression(solver='saga', penalty='l2', class_weight='balanced')

# No regularization (may overfit)
model = LogisticRegression(C=100.0, penalty='l2', class_weight='balanced')
```

### Evaluate Changes

```python
# After training new model:
results = model.evaluate(X_test, y_test)
print(f"New ROC AUC: {results['roc_auc']:.4f}")

# Compare with original performance
```

## 💾 Model Persistence

### Saving Trained Models

```python
# After training, save for later use:
model.save_model('models/my_custom_model.pkl')

# File contains:
# - Trained weights/coefficients
# - Feature names
# - Scaling parameters
# - Configuration
```

### Loading Saved Models

```python
# Load previously trained model:
model.load_model('models/my_custom_model.pkl')

# Ready to make predictions on new data!
```

## 🔍 Understanding Predictions

### High Confidence Attack (Probability = 0.95)

**Features:**

- suspicious_pattern_score: 3
- url_has_suspicious_chars: 1
- has_sql_injection_patterns: 1

**Why?** Multiple attack indicators present

### Low Confidence Normal (Probability = 0.05)

**Features:**

- url_length: 25
- method_GET: 1
- user_agent_common_browser: 1

**Why?** Standard, normal-looking request

## 📊 Model Interpretability

### Feature Coefficients

```python
# Most important features for detecting attacks:
1. suspicious_pattern_score     +2.34  (Strong positive)
2. url_has_suspicious_chars     +1.87  (Strong positive)
3. has_sql_injection_patterns   +1.65  (Strong positive)
4. content_length               +0.45  (Moderate positive)
5. url_length                   +0.32  (Moderate positive)
```

### Decision Boundary

The model creates a **decision boundary** in 39-dimensional space:

- Points on one side → Normal (0)
- Points on other side → Attack (1)
- Distance from boundary = confidence

## 🎯 Real-World Usage

### Deployment

```python
# Load trained model
model = VulnerabilityDetector()
model.load_model('models/vulnerability_detector.pkl')

# Analyze new requests
while True:
    request = get_next_http_request()
    features = feature_engineer.extract_features(request)
    prediction = model.predict(features)
    probability = model.predict_proba(features)

    if prediction[0] == 1 and probability[0][1] > 0.8:
        block_request(request)
    else:
        allow_request(request)
```

### Monitoring Performance

```python
# Log predictions for analysis
logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}, "
           f"Features: {important_features}")
```

---

**Next**: [Training Pipeline](training-pipeline.md) - How all these pieces work together!
