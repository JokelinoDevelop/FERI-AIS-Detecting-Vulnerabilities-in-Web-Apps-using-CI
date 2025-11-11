# Web Vulnerability Detection using Logistic Regression

A machine learning project for detecting web vulnerabilities in HTTP requests using logistic regression on the CSIC 2010 dataset.

## 🎯 Overview

This project implements a logistic regression model to classify HTTP requests as either normal or anomalous (potentially vulnerable to web attacks). The model is trained on the CSIC 2010 HTTP Dataset, which contains real HTTP requests labeled as normal or anomalous.

## 📊 Dataset

The project uses the **CSIC 2010 HTTP Dataset** which contains:

- **61,066 HTTP requests**
- **36,000 Normal requests**
- **25,065 Anomalous requests** (attacks)

### Features Available:

- HTTP Method
- User-Agent string
- Various HTTP headers (Pragma, Cache-Control, Accept, etc.)
- Content and content length
- URL and query parameters
- Classification label (Normal/Anomalous)

## 🏗️ Project Structure

```
.
├── data/                           # Dataset directory
│   └── csic_database.csv          # CSIC 2010 dataset
├── src/                           # Source code
│   ├── data/                      # Data processing
│   │   ├── __init__.py
│   │   └── data_loader.py         # Data loading and preprocessing
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineer.py    # HTTP request feature extraction
│   ├── models/                    # Model implementation
│   │   ├── __init__.py
│   │   ├── logistic_regression_model.py  # Logistic regression model
│   │   └── trainer.py             # Training pipeline
│   └── utils/                     # Utilities
│       ├── __init__.py
│       └── config.py              # Configuration utilities
├── results/                       # Results and plots (generated)
├── models/                        # Saved models (generated)
├── notebooks/                     # Jupyter notebooks (optional)
├── tests/                         # Unit tests
├── main.py                        # Main execution script
├── requirements.txt               # Python dependencies
├── config.json                    # Configuration file (optional)
└── README.md                      # This file
```

## 🚀 Features

### Feature Engineering

The model extracts comprehensive features from HTTP requests:

- **HTTP Method Analysis**: One-hot encoding of methods, standard method detection
- **URL Analysis**: Length, path length, query parameters, suspicious characters
- **User-Agent Analysis**: Length, browser detection, suspicious patterns
- **Content Analysis**: Length, suspicious characters, encoding detection
- **Header Analysis**: Presence of important headers, specific header values
- **Attack Pattern Detection**: SQL injection, XSS, path traversal, command injection patterns

### Model Capabilities

- **Logistic Regression** with balanced class weights
- **Feature Importance** analysis
- **Comprehensive Evaluation** metrics (ROC AUC, precision, recall, F1-score)
- **Visualization** of results (confusion matrix, ROC curves, feature importance)
- **Model Persistence** (save/load trained models)

## 📋 Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## 🛠️ Installation

1. **Clone or download the project**

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure dataset is in place:**
   The dataset `csic_database.csv` should be in the `data/` directory.

## 🎮 Usage

### Basic Usage

Run the complete training pipeline:

```bash
python main.py
```

### Advanced Usage

```bash
python main.py \
  --data-path data/csic_database.csv \
  --test-size 0.3 \
  --save-model models/my_model.pkl \
  --save-plots results/my_plots.png \
  --feature-importance
```

### Command Line Options

| Option                 | Default                             | Description                   |
| ---------------------- | ----------------------------------- | ----------------------------- |
| `--data-path`          | `data/csic_database.csv`            | Path to dataset               |
| `--test-size`          | `0.2`                               | Test set proportion           |
| `--random-state`       | `42`                                | Random seed                   |
| `--save-model`         | `models/vulnerability_detector.pkl` | Model save path               |
| `--save-plots`         | `results/evaluation_plots.png`      | Plots save path               |
| `--save-results`       | `results/evaluation_results.json`   | Results save path             |
| `--nrows`              | `None`                              | Limit data rows (for testing) |
| `--feature-importance` | `False`                             | Show top 20 features          |

## 📈 Model Performance

The logistic regression model typically achieves:

- **ROC AUC**: ~0.95-0.98
- **Accuracy**: ~90-95%
- **Precision (Anomalous)**: ~85-95%
- **Recall (Anomalous)**: ~85-95%
- **F1-Score**: ~85-95%

_Note: Actual performance may vary based on feature engineering and hyperparameter tuning._

## 🔍 Key Features Analysis

The most important features for vulnerability detection typically include:

1. **Suspicious pattern scores** (SQL injection, XSS, etc.)
2. **URL characteristics** (length, query parameters)
3. **Content analysis** (length, suspicious characters)
4. **HTTP method patterns**
5. **Header presence/absence**

## 🧪 Testing

Run the model with a subset of data for quick testing:

```bash
python main.py --nrows 1000
```

## 📊 Results Interpretation

### Confusion Matrix

```
              Predicted Normal    Predicted Anomalous
Actual Normal        TN                   FP
Actual Anomalous     FN                   TP
```

### Key Metrics

- **Precision**: TP / (TP + FP) - Of predicted attacks, how many are real?
- **Recall**: TP / (TP + FN) - Of real attacks, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall

## 🔧 Configuration

You can create a custom configuration file:

```bash
python -c "from src.utils.config import Config; Config.save_default_config('my_config.json')"
```

Then edit `my_config.json` and use it:

```bash
python main.py --config my_config.json
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational purposes. Please refer to the original CSIC dataset licensing terms.

## 📚 References

- [CSIC 2010 HTTP Dataset](http://www.isi.csic.es/dataset/)
- [Original CSIC Research Paper](https://www.isi.csic.es/~jalvareza/publications.html)

## 🆘 Troubleshooting

### Common Issues

1. **Memory Error**: Use `--nrows` to limit data size
2. **Import Error**: Ensure you're running from the project root
3. **File Not Found**: Check that `csic_database.csv` is in `data/` directory

### Getting Help

If you encounter issues:

1. Check the logs for detailed error messages
2. Verify your Python environment and dependencies
3. Try running with `--nrows 1000` for testing
