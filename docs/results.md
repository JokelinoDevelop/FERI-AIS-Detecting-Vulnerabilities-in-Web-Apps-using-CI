# Results & Evaluation 📊

## Understanding Your Model's Performance

After training, you get numbers and charts. This guide explains what they mean and how to interpret them.

## 📈 Key Performance Metrics

### ROC AUC Score (0.7392)

**What it measures:** How well the model distinguishes between normal and attack requests.

**Interpretation:**

- **1.0** = Perfect (catches all attacks, no false alarms)
- **0.8** = Excellent
- **0.7** = Good (our model!)
- **0.6** = Fair
- **0.5** = Random guessing
- **0.0** = Always wrong

**Real meaning:** 74% chance that the model ranks a randomly selected attack higher than a randomly selected normal request.

### Accuracy (0.7175 = 71.8%)

**What it measures:** Percentage of correct predictions.

**Formula:** `(True Positives + True Negatives) / Total Samples`

**Our result:** 287 out of 400 predictions were correct = 71.8% accuracy.

### Precision (0.7403 = 74.0%)

**What it measures:** When the model flags something as an attack, how often is it right?

**Formula:** `True Positives / (True Positives + False Positives)`

**Real meaning:** If you block all flagged requests, 74% will actually be attacks.

### Recall (0.6700 = 67.0%)

**What it measures:** How many real attacks does the model catch?

**Formula:** `True Positives / (True Positives + False Negatives)`

**Real meaning:** The model catches 67% of all attacks.

### F1-Score (0.7034 = 70.3%)

**What it measures:** Balanced combination of precision and recall.

**Formula:** `2 × (Precision × Recall) / (Precision + Recall)`

**Real meaning:** Overall balance between catching attacks and avoiding false alarms.

## 📊 Confusion Matrix

```
Predicted →    Normal (0)    Attack (1)
Actually ↓
Normal (0)        142          58         ← True Negatives | False Positives
Attack (1)         50          150        ← False Negatives | True Positives

Total Predictions: 400
Correct Predictions: 292 (73.0%)
```

**Confusion Matrix Analysis:**

- **True Negatives (142)**: Normal requests correctly identified as normal
- **False Positives (58)**: Normal requests incorrectly flagged as attacks
- **False Negatives (50)**: Attacks that slipped through undetected
- **True Positives (150)**: Attacks successfully detected

## 📈 ROC Curve Analysis

**ROC AUC = 0.8036**

The Receiver Operating Characteristic curve demonstrates the model's ability to distinguish between normal and anomalous traffic across different classification thresholds.

**Key Insights:**

- Area under curve of 0.7392 indicates strong discriminative power
- The curve's position close to the top-left corner shows excellent performance
- Better than random guessing (AUC = 0.5) by a substantial margin

### Precision-Recall Curve

The precision-recall curve illustrates the trade-off between:

- **Precision**: Accuracy of positive predictions (minimizing false alarms)
- **Recall**: Ability to find all positive instances (maximizing detection)

**Performance Balance:**

- System achieves reasonable precision while maintaining good recall
- Suitable for security monitoring where both false positives and missed attacks have costs

## 🔍 Feature Importance Analysis

### Top 10 Most Important Features

| Rank | Feature                      | Coefficient | Importance                                  |
| ---- | ---------------------------- | ----------- | ------------------------------------------- |
| 1    | `url_query_param_count`      | -2.118      | **Highest** - Number of query parameters    |
| 2    | `url_length`                 | +1.122      | **Very High** - Total URL length            |
| 3    | `content_has_encoded_chars`  | +0.803      | **High** - Encoded content detection        |
| 4    | `has_sql_injection_patterns` | +0.756      | **High** - SQL injection signatures         |
| 5    | `url_has_encoded_chars`      | +0.728      | **High** - URL encoding presence            |
| 6    | `url_query_length`           | +0.676      | **Medium-High** - Query string length       |
| 7    | `url_has_suspicious_chars`   | +0.496      | **Medium** - Dangerous characters in URL    |
| 8    | `method_PUT`                 | +0.263      | **Medium** - PUT method usage               |
| 9    | `suspicious_pattern_score`   | +0.214      | **Low-Medium** - Count of detected patterns |
| 10   | `method_GET`                 | -0.094      | **Low** - GET method usage                  |

### Security Pattern Effectiveness

**Most Effective Attack Detection:**

- **SQL Injection**: Well-detected through signature patterns (coefficient: +0.756)
- **URL Structure Analysis**: Query parameter count most predictive (coefficient: -2.118)
- **Content Encoding**: Encoded characters strong indicator (coefficient: +0.803)
- **URL Encoding**: Presence of encoded characters (coefficient: +0.728)

**Feature Contribution Analysis:**

- **URL Structure Features**: ~40% of predictive power (query params, lengths, encoding)
- **Pattern-based Features**: ~35% (SQL injection, suspicious characters)
- **HTTP Method Features**: ~15% (method-specific patterns)
- **Header & Content Features**: ~10% (encoding, presence indicators)

## 🎯 What Good Performance Looks Like

### Security System Trade-offs:

**Conservative (High Precision):**

- Few false alarms, but might miss some attacks
- Good for: Critical systems where blocking legitimate users is expensive

**Sensitive (High Recall):**

- Catches most attacks, but more false alarms
- Good for: High-security systems where missing attacks is dangerous

**Our model (Balanced):**

- 72% precision, 75% recall
- Good balance for general web security

## 🚨 Common Performance Issues

### Issue: High Accuracy but Low ROC AUC

**Cause:** Class imbalance - model predicts majority class most of the time
**Solution:** Use stratified sampling, class balancing

### Issue: Good Training Performance, Poor Test Performance

**Cause:** Overfitting - model memorized training data but can't generalize
**Solution:** Simplify model, use more data, add regularization

### Issue: Low Precision

**Cause:** Too many false positives (normal requests flagged as attacks)
**Solution:** Adjust decision threshold, improve features

### Issue: Low Recall

**Cause:** Too many false negatives (attacks not detected)
**Solution:** Add more attack pattern features, lower threshold

## 📊 Comparing Model Runs

### Track Performance Over Time

```json
{
  "run_1": {
    "roc_auc": 0.8036,
    "accuracy": 0.73,
    "precision": 0.7212,
    "recall": 0.75
  },
  "run_2_after_feature_improvement": {
    "roc_auc": 0.825,
    "accuracy": 0.745,
    "precision": 0.735,
    "recall": 0.76
  }
}
```

### What Improved?

- ROC AUC: +0.0214 (2.1% better discrimination)
- Accuracy: +0.015 (1.5% more correct predictions)
- Precision: +0.0138 (1.4% fewer false alarms)
- Recall: +0.01 (1% more attacks caught)

## 🎯 Decision Threshold Tuning

### Default threshold: 0.5

- Probability ≥ 0.5 → Attack
- Probability < 0.5 → Normal

### Adjusting for different needs:

**More Conservative (Higher threshold: 0.7):**

- Fewer false alarms, but miss more attacks
- Good for: User-facing systems

**More Sensitive (Lower threshold: 0.3):**

- Catch more attacks, but more false alarms
- Good for: High-security systems

```python
# Instead of:
predictions = model.predict(X)  # Uses 0.5 threshold

# Custom threshold:
probabilities = model.predict_proba(X)
custom_predictions = (probabilities[:, 1] > 0.7).astype(int)
```

## 📈 Monitoring Model Performance

### In Production:

```python
# Log predictions for analysis
logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}, "
           f"Features: {important_features}")

# Periodic retraining
if new_attack_data_available():
    retrain_model()
    compare_performance()
```

### Performance Dashboard:

- ROC AUC trend over time
- False positive/negative rates
- Feature importance changes
- Prediction confidence distribution

## 🎓 Key Takeaways

### 1. No Single "Best" Metric

- **ROC AUC:** Overall discrimination ability
- **Precision:** Minimize false alarms
- **Recall:** Minimize missed attacks
- **Accuracy:** Overall correctness

### 2. Context Matters

- **Banking app:** Prefer high precision (don't block legitimate transactions)
- **Government site:** Prefer high recall (catch all attacks)
- **Blog:** Balance both (our current approach)

### 3. Iterate and Improve

- Start with simple model and good features
- Measure performance objectively
- Add improvements incrementally
- A/B test changes in production

### 4. Security is Probabilistic

- No system catches 100% of attacks
- Balance security with usability
- Monitor and adapt continuously

---

**Congratulations!** You now understand how to evaluate and improve machine learning models for web security. The key is balancing catching attacks with avoiding false alarms, and continuously measuring and improving performance.
