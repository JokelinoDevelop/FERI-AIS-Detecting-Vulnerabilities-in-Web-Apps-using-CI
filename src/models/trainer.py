"""
Training pipeline for web vulnerability detection
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from ..data.data_loader import load_and_preprocess_data
from ..features.feature_engineer import HTTPFeatureEngineer
from .logistic_regression_model import VulnerabilityDetector

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Training pipeline for vulnerability detection model"""

    def __init__(self,
                 data_path: str = "data/csic_database.csv",
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialize trainer

        Args:
            data_path: Path to dataset
            test_size: Proportion for test set
            random_state: Random state for reproducibility
        """
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state

        self.train_df = None
        self.test_df = None
        self.feature_engineer = HTTPFeatureEngineer()
        self.model = VulnerabilityDetector(random_state=random_state)

    def load_data(self, nrows: Optional[int] = None) -> 'ModelTrainer':
        """
        Load and preprocess data

        Args:
            nrows: Number of rows to load (for testing)

        Returns:
            Self for method chaining
        """
        logger.info("Loading and preprocessing data...")
        self.train_df, self.test_df = load_and_preprocess_data(
            self.data_path, self.test_size, nrows
        )
        return self

    def extract_features(self) -> 'ModelTrainer':
        """
        Extract features from training and test data

        Returns:
            Self for method chaining
        """
        if self.train_df is None or self.test_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        logger.info("Extracting features...")

        # Extract features for training data
        self.X_train = self.feature_engineer.extract_features(self.train_df)
        self.y_train = self.train_df['target']

        # Extract features for test data
        self.X_test = self.feature_engineer.extract_features(self.test_df)
        self.y_test = self.test_df['target']

        logger.info(f"Training features shape: {self.X_train.shape}")
        logger.info(f"Test features shape: {self.X_test.shape}")

        return self

    def train_model(self) -> 'ModelTrainer':
        """
        Train the model

        Returns:
            Self for method chaining
        """
        if not hasattr(self, 'X_train') or not hasattr(self, 'y_train'):
            raise ValueError("Features not extracted. Call extract_features() first.")

        logger.info("Training model...")
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model.pipeline is None:
            raise ValueError("Model not trained. Call train_model() first.")

        logger.info("Evaluating model on test set...")
        return self.model.evaluate(self.X_test, self.y_test)

    def save_model(self, filepath: str = "models/vulnerability_detector.pkl") -> 'ModelTrainer':
        """
        Save the trained model

        Args:
            filepath: Path to save the model

        Returns:
            Self for method chaining
        """
        self.model.save_model(filepath)
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance

        Returns:
            DataFrame with feature importance
        """
        return self.model.get_feature_importance()

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot evaluation results

        Args:
            save_path: Path to save plots
        """
        if save_path is None:
            save_path = "results/evaluation_plots.png"

        # Create results directory
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        self.model.plot_evaluation_metrics(self.X_test, self.y_test, save_path)

    def run_full_pipeline(self,
                         save_model_path: Optional[str] = None,
                         save_plots_path: Optional[str] = None,
                         nrows: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete training pipeline

        Args:
            save_model_path: Path to save trained model
            save_plots_path: Path to save evaluation plots
            nrows: Number of rows to load (for testing)

        Returns:
            Dictionary with evaluation results
        """
        logger.info("Starting full training pipeline...")

        # Execute pipeline
        self.load_data(nrows)
        self.extract_features()
        self.train_model()

        # Evaluate
        evaluation_results = self.evaluate_model()

        # Save model if requested
        if save_model_path:
            self.save_model(save_model_path)

        # Generate plots if requested
        if save_plots_path:
            self.plot_results(save_plots_path)

        # Get feature importance
        feature_importance = self.get_feature_importance()

        logger.info("Training pipeline completed successfully!")

        return {
            'evaluation_metrics': evaluation_results,
            'feature_importance': feature_importance,
            'model': self.model,
            'trainer': self
        }
