"""
Logistic Regression model for web vulnerability detection
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VulnerabilityDetector:
    """Logistic Regression model for detecting web vulnerabilities"""

    def __init__(self,
                 random_state: int = 42,
                 max_iter: int = 1000,
                 C: float = 1.0,
                 penalty: str = 'l2',
                 class_weight: str = 'balanced',
                 solver: str = 'lbfgs'):
        """
        Initialize the vulnerability detector

        Args:
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for logistic regression
            C: Regularization strength (smaller = stronger regularization)
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            class_weight: Class weighting strategy
            solver: Optimization algorithm
        """
        self.random_state = random_state
        self.max_iter = max_iter
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.solver = solver
        self.model = None
        self.feature_names = None

        # Create the pipeline
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=random_state,
                max_iter=max_iter,
                C=C,
                penalty=penalty,
                class_weight=class_weight,
                solver=solver
            ))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'VulnerabilityDetector':
        """
        Train the model

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Self for method chaining
        """
        logger.info("Training logistic regression model...")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values

        # Train the model
        self.pipeline.fit(X_array, y_array)

        logger.info("Model training completed")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix

        Returns:
            Predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_array = X.values
        return self.pipeline.predict(X_array)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities

        Args:
            X: Feature matrix

        Returns:
            Predicted probabilities for each class
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_array = X.values
        return self.pipeline.predict_proba(X_array)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate model performance

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Evaluating model performance...")

        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        # Calculate metrics
        metrics = {
            'classification_report': classification_report(y, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
            'roc_auc': roc_auc_score(y, y_proba[:, 1])
        }

        # Log key metrics
        logger.info(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {metrics['classification_report']['accuracy']:.4f}")
        logger.info(f"Precision (Anomalous): {metrics['classification_report']['1']['precision']:.4f}")
        logger.info(f"Recall (Anomalous): {metrics['classification_report']['1']['recall']:.4f}")
        logger.info(f"F1-Score (Anomalous): {metrics['classification_report']['1']['f1-score']:.4f}")

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from logistic regression coefficients

        Returns:
            DataFrame with feature names and their importance scores
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Get coefficients from the classifier
        coefficients = self.pipeline.named_steps['classifier'].coef_[0]

        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })

        # Sort by absolute coefficient value
        feature_importance = feature_importance.sort_values(
            'abs_coefficient', ascending=False
        ).reset_index(drop=True)

        return feature_importance

    def plot_evaluation_metrics(self,
                              X: pd.DataFrame,
                              y: pd.Series,
                              save_path: Optional[str] = None) -> None:
        """
        Plot evaluation metrics

        Args:
            X: Feature matrix
            y: Target vector
            save_path: Path to save plots (optional)
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_proba[:, 1])
        axes[0, 1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y, y_proba[:, 1]):.4f}')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_proba[:, 1])
        axes[1, 0].plot(recall, precision)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')

        # Feature Importance (top 20, excluding zero coefficients)
        feature_imp = self.get_feature_importance()
        # Filter out zero coefficients and get top 20
        feature_imp = feature_imp[feature_imp['abs_coefficient'] > 0].head(20)
        # Reverse order so largest values are at top
        feature_imp = feature_imp.iloc[::-1]
        
        # Color bars based on sign (red for negative, green for positive)
        colors = ['#d62728' if x < 0 else '#2ca02c' for x in feature_imp['coefficient']]
        bars = axes[1, 1].barh(feature_imp['feature'], feature_imp['coefficient'], color=colors)
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[1, 1].set_xlabel('Coefficient')
        axes[1, 1].set_title('Top 20 Feature Importance')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        # Add legend
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Positive (increases vulnerability risk)'),
            Patch(facecolor='#d62728', label='Negative (decreases vulnerability risk)')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='lower right', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to {save_path}")
        else:
            plt.show()

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model

        Args:
            filepath: Path to save the model
        """
        if self.pipeline is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'C': self.C,
            'penalty': self.penalty,
            'class_weight': self.class_weight,
            'solver': self.solver
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> 'VulnerabilityDetector':
        """
        Load a trained model

        Args:
            filepath: Path to the saved model

        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)

        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.random_state = model_data['random_state']
        self.max_iter = model_data['max_iter']
        self.C = model_data.get('C', 1.0)
        self.penalty = model_data.get('penalty', 'l2')
        self.class_weight = model_data.get('class_weight', 'balanced')
        self.solver = model_data.get('solver', 'lbfgs')

        logger.info(f"Model loaded from {filepath}")
        return self
