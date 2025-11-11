# Glossary 📚

## Machine Learning Terms

### A

**Accuracy**: Percentage of correct predictions out of total predictions.

**Algorithm**: A procedure or formula for solving a problem (e.g., Logistic Regression).

### C

**Classification**: Sorting data into categories (our task: safe vs dangerous requests).

**Class**: A category in classification (Class 0 = Normal, Class 1 = Anomalous).

**Confusion Matrix**: Table showing correct and incorrect predictions.

**Cross-Validation**: Testing model on multiple data splits to ensure reliability.

### D

**DataFrame**: A 2D table structure (like Excel) used in pandas.

**Dataset**: Collection of data used for training/testing.

### E

**Evaluation**: Measuring how well a model performs on unseen data.

**Extract**: To pull out useful information from raw data.

### F

**False Negative**: Attack that model failed to detect (dangerous!).

**False Positive**: Normal request incorrectly flagged as attack.

**Feature**: A measurable property used for prediction (e.g., URL length).

**Feature Engineering**: Converting raw data into features computers can learn from.

### H

**Hyperparameter**: Settings that control the learning algorithm (e.g., learning rate).

### I

**Imbalanced Data**: When one class has many more examples than another.

### L

**Label**: The "answer" we're trying to predict (0 = normal, 1 = attack).

**Logistic Regression**: The ML algorithm we use - predicts probabilities.

### M

**Machine Learning**: Computers learning patterns from data without explicit programming.

**Model**: A trained algorithm that can make predictions.

### O

**Overfitting**: Model memorizes training data but performs poorly on new data.

### P

**Pipeline**: A sequence of automated steps (data → features → model → evaluation).

**Precision**: When model flags attack, how often is it correct?

**Prediction**: What the model thinks the answer is.

**Probability**: Confidence score (0.0 to 1.0) of a prediction.

### R

**Recall**: How many real attacks does the model catch?

**Regression**: Predicting continuous values (vs classification for categories).

**ROC AUC**: Measure of how well model distinguishes between classes.

### S

**Sample**: One data point (one HTTP request in our case).

**Supervised Learning**: Learning from labeled examples (we have the answers).

**Scaler**: Tool that normalizes feature values to similar ranges.

### T

**Test Set**: Data used to evaluate trained model (never seen during training).

**Training Set**: Data used to teach the model.

**True Negative**: Normal request correctly identified as normal.

**True Positive**: Attack correctly identified as attack.

### U

**Underfitting**: Model too simple to learn patterns in the data.

### V

**Validation**: Checking model performance on data it hasn't seen.

---

## Web Security Terms

### A

**Anomalous**: Unusual or suspicious (our attacks).

**Attack**: Malicious attempt to compromise a system.

### C

**Command Injection**: Attacker runs system commands through web forms.

**Cross-Site Scripting (XSS)**: Attacker injects malicious JavaScript.

### H

**HTTP**: Protocol for web communication (HyperText Transfer Protocol).

**HTTP Request**: Message sent from browser to server.

### N

**Normal**: Regular, safe behavior (our legitimate requests).

### P

**Path Traversal**: Attacker accesses files outside allowed directories.

### R

**Request**: See HTTP Request.

### S

**SQL Injection**: Attacker runs database commands through web forms.

**Suspicious**: Potentially dangerous or unusual.

### U

**URL**: Web address (Uniform Resource Locator).

### W

**Web Application**: Software that runs in web browsers.

---

## Python Programming Terms

### A

**Argument**: Value passed to a function.

### C

**Class**: Blueprint for creating objects.

**CSV**: Comma-Separated Values file format.

### D

**Dictionary**: Key-value data structure (like a lookup table).

### F

**Function**: Reusable block of code.

### I

**Import**: Loading external code/libraries.

### L

**List**: Ordered collection of items.

### M

**Method**: Function that belongs to a class.

### O

**Object**: Instance of a class.

### P

**Parameter**: Variable in function definition.

**Pandas**: Library for data manipulation.

### R

**Return**: What a function gives back.

### S

**String**: Text data type.

**Scikit-learn**: Machine learning library for Python.

### V

**Variable**: Named storage for data.

---

## File Structure Terms

### C

**Configuration**: Settings that control program behavior.

### D

**Data**: Information used by the program.

### L

**Logging**: Recording program activity for debugging.

### M

**Model**: Saved trained machine learning model.

### R

**Results**: Output files (charts, metrics, reports).

### S

**Source**: Program code files.

### U

**Utils**: Utility/helper functions.

---

## Performance Terms

### A

**Area Under Curve (AUC)**: See ROC AUC.

### B

**Balanced Accuracy**: Accuracy that accounts for class imbalance.

### F

**F1-Score**: Harmonic mean of precision and recall.

### R

**Receiver Operating Characteristic (ROC)**: Plot of true vs false positive rates.

### T

**Threshold**: Decision boundary (e.g., probability > 0.5 = attack).

---

_Tip: When you encounter an unfamiliar term, search for it in this glossary or ask specific questions about it!_
