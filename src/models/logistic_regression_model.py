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
import seaborn as sns
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

from ..utils.config import get_logger

logger = get_logger(__name__)


class VulnerabilityDetector:
    """Logistic Regression model for detecting web vulnerabilities"""

    def __init__(self, random_state: int = 42, max_iter: int = 1000):
        """
        Initialize the vulnerability detector

        Args:
            random_state: Random state for reproducibility
            max_iter: Maximum iterations for logistic regression
        """
        self.random_state: int = random_state
        self.max_iter: int = max_iter
        self.feature_names: Optional[List[str]] = None

        # Create the pipeline
        self.pipeline: Pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=random_state,
                max_iter=max_iter,
                class_weight='balanced'  # Handle class imbalance
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

        # Feature Importance (top 20)
        feature_imp = self.get_feature_importance().head(20)
        axes[1, 1].barh(feature_imp['feature'], feature_imp['coefficient'])
        axes[1, 1].set_xlabel('Coefficient')
        axes[1, 1].set_title('Top 20 Feature Importance')

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
            'max_iter': self.max_iter
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

        logger.info(f"Model loaded from {filepath}")
        return self
