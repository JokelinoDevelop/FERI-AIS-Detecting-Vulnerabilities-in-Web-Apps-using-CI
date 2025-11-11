# Feature Engineering 🔧

## What is Feature Engineering?

Feature engineering is **converting raw data into numbers** that a machine learning algorithm can understand and learn from.

### The Problem

Computers can't understand text directly:

```
❌ Raw HTTP Request: "GET /admin.php?cmd=DROP TABLE users"
```

### The Solution

Convert to numbers the computer can learn:

```
✅ Features:
   url_length: 35
   has_sql_injection_patterns: 1
   method_is_standard: 1
   suspicious_pattern_score: 2
```

## 🧠 The FeatureEngineer Class

### File: `src/features/feature_engineer.py`

```python
class HTTPFeatureEngineer:
    def __init__(self):
        # Pre-compile attack patterns for speed
        self.suspicious_patterns = {
            'sql_injection': [...],
            'xss': [...],
            'path_traversal': [...],
            'command_injection': [...]
        }

    def extract_features(self, df):
        # Main method: converts DataFrame to feature matrix
        # Returns: DataFrame with 38 numeric features

    def _extract_url_features(self, df):
        # URL-specific features

    def _extract_content_features(self, df):
        # Content/body features

    def _extract_header_features(self, df):
        # HTTP header features

    def _extract_suspicious_features(self, df):
        # Pattern detection features
```

## 📊 The 38 Features We Create

### 1. HTTP Method Features (8 features)

One-hot encoded representation of HTTP methods:

- `method_GET`, `method_POST`, `method_PUT`, `method_DELETE`
- `method_HEAD`, `method_OPTIONS`, `method_PATCH`
- `method_is_standard`: Binary flag for standard HTTP methods

### 2. URL Analysis Features (6 features)

| Feature                    | Description                   | Type    | Range   |
| -------------------------- | ----------------------------- | ------- | ------- |
| `url_length`               | Total character count of URL  | Numeric | 10-200+ |
| `url_path_length`          | Length of URL path component  | Numeric | 5-150   |
| `url_query_param_count`    | Number of query parameters    | Numeric | 0-15    |
| `url_query_length`         | Total length of query string  | Numeric | 0-500   |
| `url_has_suspicious_chars` | Contains dangerous characters | Binary  | 0-1     |
| `url_has_encoded_chars`    | Contains URL encoding (`%XX`) | Binary  | 0-1     |

### 3. User-Agent Features (4 features)

| Feature                     | Description                     | Type    |
| --------------------------- | ------------------------------- | ------- |
| `user_agent_length`         | Length of User-Agent string     | Numeric |
| `user_agent_empty`          | Is User-Agent header empty      | Binary  |
| `user_agent_common_browser` | Matches common browser patterns | Binary  |
| `user_agent_suspicious`     | Contains attack tool signatures | Binary  |

### 4. Request Content Features (4 features)

| Feature                        | Description                   | Type    |
| ------------------------------ | ----------------------------- | ------- |
| `content_length`               | Size of request body          | Numeric |
| `content_empty`                | Is request body empty         | Binary  |
| `content_has_suspicious_chars` | Contains dangerous characters | Binary  |
| `content_has_encoded_chars`    | Contains encoded characters   | Binary  |

### 5. HTTP Header Features (11 features)

Presence indicators for important HTTP headers:

- `header_pragma_present`, `header_cache_control_present`
- `header_accept_present`, `header_accept_encoding_present`
- `header_accept_charset_present`, `header_language_present`
- `header_host_present`, `header_cookie_present`, `header_content_type_present`
- `host_localhost`: Host header contains localhost
- `language_english`: Language header indicates English

### 6. Attack Pattern Detection Features (5 features)

Advanced security pattern detection:

| Feature                          | Description                   | Detection Logic                   |
| -------------------------------- | ----------------------------- | --------------------------------- |
| `has_sql_injection_patterns`     | SQL injection signatures      | `UNION SELECT`, `1=1`, `--`, etc. |
| `has_xss_patterns`               | Cross-site scripting patterns | `<script>`, `javascript:`, etc.   |
| `has_path_traversal_patterns`    | Directory traversal attempts  | `../`, `..\\`, etc.               |
| `has_command_injection_patterns` | System command injection      | `;`, `\|`, `` ` ``, etc.          |
| `suspicious_pattern_score`       | Count of detected patterns    | Sum of pattern detection features |

## 🔍 Feature Extraction Examples

### Normal Request → Features

**Input:**

```
GET /search?q=cats HTTP/1.1
Host: example.com
User-Agent: Mozilla/5.0 (Windows NT 10.0)
```

**Features:**

```python
{
    'url_length': 15,
    'url_query_param_count': 1,
    'url_has_suspicious_chars': 0,
    'url_has_encoded_chars': 0,
    'method_GET': 1,
    'method_POST': 0,
    'method_PUT': 0,
    'method_DELETE': 0,
    'user_agent_length': 25,
    'user_agent_common_browser': 1,
    'content_length': 0,
    'content_empty': 1,
    'has_sql_injection_patterns': 0,
    'has_xss_patterns': 0,
    'suspicious_pattern_score': 0
}
```

### Attack Request → Features

**Input:**

```
GET /admin.php?id=1' UNION SELECT password FROM users-- HTTP/1.1
User-Agent: sqlmap/1.4.5
```

**Features:**

```python
{
    'url_length': 50,
    'url_query_param_count': 1,
    'url_has_suspicious_chars': 1,    # Has quotes
    'url_has_encoded_chars': 0,
    'method_GET': 1,
    'method_POST': 0,
    'user_agent_length': 12,
    'user_agent_empty': 0,
    'user_agent_common_browser': 0,   # Not a real browser
    'user_agent_suspicious': 1,       # sqlmap detected
    'content_length': 0,
    'content_empty': 1,
    'has_sql_injection_patterns': 1,  # UNION SELECT found
    'has_xss_patterns': 0,
    'has_path_traversal_patterns': 0,
    'has_command_injection_patterns': 0,
    'suspicious_pattern_score': 1     # One suspicious pattern detected
}
```

## 🛠️ How Pattern Detection Works

### Pre-compiled Regex Patterns

```python
# Instead of searching with strings each time:
sql_patterns = [
    r'union\s+select',
    r'1=1',
    r'--',
    r'/\*',
    r'\*/'
]

# We pre-compile them for speed:
self._compiled_patterns = {
    'sql_injection': [re.compile(pattern, re.IGNORECASE) for pattern in sql_patterns]
}
```

### Efficient Searching

```python
def _safe_search(self, text: str, patterns: List[re.Pattern]) -> bool:
    """Safely search for patterns in text"""
    if not isinstance(text, str) or not text.strip():
        return False
    return any(pattern.search(text) for pattern in patterns)
```

## 📈 Feature Importance

After training, we can see which features matter most:

```python
# Top 5 features by importance (from recent training):
1. url_query_param_count         (coefficient: -2.118)
2. url_length                    (coefficient: 1.122)
3. content_has_encoded_chars     (coefficient: 0.803)
4. has_sql_injection_patterns    (coefficient: 0.756)
5. url_has_encoded_chars         (coefficient: 0.728)
```

## 🎯 Why Feature Engineering Matters

### 1. **Computer Needs Numbers**

- ML algorithms work with numbers, not text
- Each feature becomes a column in your dataset

### 2. **Domain Knowledge**

- We know web attacks have specific patterns
- Features capture security expertise

### 3. **Model Performance**

- Good features = better model accuracy
- Wrong features = model can't learn

### 4. **Interpretability**

- Features explain why predictions are made
- Can debug and improve the system

## 🔧 Feature Engineering Best Practices

### 1. **Start Simple**

- Length, counts, presence/absence
- Easy to understand and debug

### 2. **Domain Expertise**

- Include security-specific knowledge
- Attack patterns, suspicious indicators

### 3. **Avoid Data Leakage**

- Features should be available at prediction time
- Don't use future information

### 4. **Handle Missing Values**

- What if URL is empty? Content missing?
- Robust handling prevents crashes

### 5. **Scale Appropriately**

- Some features (length) can be large numbers
- Model handles scaling automatically

## 🚨 Common Issues

### Problem: Too Many Features

**Symptom**: Model overfits, poor generalization
**Solution**: Select most important features

### Problem: Data Leakage

**Symptom**: Perfect training accuracy, poor test accuracy
**Solution**: Ensure features available at prediction time

### Problem: Sparse Features

**Symptom**: Many features are always 0
**Solution**: Remove useless features, focus on informative ones

## 🎮 Experimenting with Features

### Add New Features

```python
# In _extract_url_features method:
features['url_has_admin_word'] = df['URL'].str.contains(
    r'admin|config|setup|install', case=False
).astype(int)
```

### Modify Existing Features

```python
# Make suspicious chars more comprehensive:
suspicious_chars = ['<', '>', '"', "'", ';', '|', '$', '`', '\\', '\x00', '\n', '\r']
```

### Test Feature Impact

```python
# Add your feature, retrain, check if performance improves
# Look at feature importance to see if it's useful
```

## 📊 Feature Matrix

After feature engineering, we have:

```python
# Input: DataFrame (2000 rows × 18 columns)
# Output: Feature matrix (2000 rows × 38 columns)

print("Features shape:", X.shape)  # (2000, 38)
print("Feature names:", X.columns.tolist())
# ['method_GET', 'method_POST', 'url_length', 'url_path_length', ...]
```

## 🔄 Next Steps

The 38 features become the **input to the machine learning model**:

1. **Feature Matrix (2000 × 38)**: Numbers describing each request
2. **Target Vector (2000 × 1)**: 0=normal, 1=attack
3. **ML Algorithm**: Learns patterns from features → predictions

---

**Next**: [The ML Model](ml-model.md) - How the computer actually learns from these features!
