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
from ..utils.config import get_logger, DataLoadError, FeatureEngineeringError, ModelTrainingError, PipelineError

logger = get_logger(__name__)


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
        self.data_path: str = data_path
        self.test_size: float = test_size
        self.random_state: int = random_state

        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.feature_engineer: HTTPFeatureEngineer = HTTPFeatureEngineer()
        self.model: VulnerabilityDetector = VulnerabilityDetector(random_state=random_state)

    def validate_pipeline_state(self, required_stage: str) -> None:
        """
        Validate that the pipeline is in the required state

        Args:
            required_stage: The minimum pipeline stage required
                ('data_loaded', 'features_extracted', 'model_trained')

        Raises:
            ValueError: If the pipeline is not in the required state
        """
        stages = {
            'data_loaded': self.train_df is not None and self.test_df is not None,
            'features_extracted': hasattr(self, 'X_train') and self.X_train is not None,
            'model_trained': self.model.pipeline is not None
        }

        stage_order = ['data_loaded', 'features_extracted', 'model_trained']
        required_index = stage_order.index(required_stage)

        for i, stage in enumerate(stage_order[:required_index + 1]):
            if not stages[stage]:
                raise ValueError(f"Pipeline stage '{stage}' not completed. "
                               f"Call the appropriate method first.")

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

        Raises:
            ValueError: If data not loaded first
        """
        self.validate_pipeline_state('data_loaded')

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

        Raises:
            ValueError: If features not extracted first
        """
        self.validate_pipeline_state('features_extracted')

        logger.info("Training model...")
        self.model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self) -> Dict[str, Any]:
        """
        Evaluate the trained model

        Returns:
            Dictionary with evaluation metrics

        Raises:
            ValueError: If model not trained first
        """
        self.validate_pipeline_state('model_trained')

        logger.info("Evaluating model on test set...")
        return self.model.evaluate(self.X_test, self.y_test)

    def save_model(self, filepath: str = "models/vulnerability_detector.pkl") -> 'ModelTrainer':
        """
        Save the trained model

        Args:
            filepath: Path to save the model

        Returns:
            Self for method chaining

        Raises:
            ValueError: If model not trained first
        """
        self.validate_pipeline_state('model_trained')

        self.model.save_model(filepath)
        return self

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance

        Returns:
            DataFrame with feature importance

        Raises:
            ValueError: If model not trained first
        """
        self.validate_pipeline_state('model_trained')

        return self.model.get_feature_importance()

    def plot_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot evaluation results

        Args:
            save_path: Path to save plots

        Raises:
            ValueError: If model not trained first
        """
        self.validate_pipeline_state('model_trained')

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
            Dictionary with evaluation results and metadata

        Raises:
            RuntimeError: If any pipeline step fails
        """
        logger.info("Starting full training pipeline...")

        try:
            # Step 1: Load and preprocess data
            logger.info("Step 1/5: Loading and preprocessing data...")
            self.load_data(nrows)

            # Step 2: Extract features
            logger.info("Step 2/5: Extracting features...")
            self.extract_features()

            # Step 3: Train model
            logger.info("Step 3/5: Training model...")
            self.train_model()

            # Step 4: Evaluate model
            logger.info("Step 4/5: Evaluating model...")
            evaluation_results = self.evaluate_model()

            # Step 5: Save artifacts
            logger.info("Step 5/5: Saving artifacts...")

            # Save model if requested
            if save_model_path:
                self.save_model(save_model_path)
                logger.info(f"Model saved to {save_model_path}")

            # Generate plots if requested
            if save_plots_path:
                self.plot_results(save_plots_path)
                logger.info(f"Plots saved to {save_plots_path}")

            # Get feature importance
            feature_importance = self.get_feature_importance()

            # Calculate additional metrics
            pipeline_metadata = {
                'training_samples': len(self.train_df) if self.train_df is not None else 0,
                'test_samples': len(self.test_df) if self.test_df is not None else 0,
                'num_features': self.X_train.shape[1] if self.X_train is not None else 0,
                'model_type': 'LogisticRegression',
                'feature_engineering_version': '2.0'  # Track feature engineering changes
            }

            logger.info("Training pipeline completed successfully!")
            logger.info(f"Model ROC-AUC: {evaluation_results.get('roc_auc', 'N/A'):.4f}")

            return {
                'evaluation_metrics': evaluation_results,
                'feature_importance': feature_importance,
                'pipeline_metadata': pipeline_metadata,
                'model': self.model,
                'trainer': self
            }

        except (DataLoadError, FeatureEngineeringError, ModelTrainingError) as e:
            logger.error(f"Training pipeline failed: {e}")
            raise PipelineError(f"Training pipeline failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in training pipeline: {e}")
            raise PipelineError(f"Training pipeline failed with unexpected error: {e}") from e
