"""
Configuration utilities
"""

from pathlib import Path
from typing import Dict, Any
import json


class Config:
    """Configuration class for the project"""

    # Default configuration
    DEFAULT_CONFIG = {
        'data': {
            'path': 'data/csic_database.csv',
            'test_size': 0.2,
            'random_state': 42
        },
        'model': {
            'save_path': 'models/vulnerability_detector.pkl',
            'max_iter': 1000,
            'C': 1.0,  # Regularization strength (smaller = stronger regularization)
            'penalty': 'l2',  # L2 regularization
            'class_weight': 'balanced',  # Handle class imbalance
            'solver': 'lbfgs'  # Efficient solver for small datasets
        },
        'training': {
            'nrows': None,  # Load all data by default
            'stratified_sample': True  # Ensure balanced classes
        },
        'results': {
            'plots_path': 'results/evaluation_plots.png',
            'metrics_path': 'results/evaluation_results.json'
        },
        'cross_validation': {
            'cv_folds': 5,
            'scoring': 'roc_auc'  # Primary metric for CV
        }
    }

    @classmethod
    def get_config(cls, config_file: str = None) -> Dict[str, Any]:
        """
        Get configuration, optionally loading from file

        Args:
            config_file: Path to config file (optional)

        Returns:
            Configuration dictionary
        """
        config = cls.DEFAULT_CONFIG.copy()

        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                # Deep merge user config with defaults
                config = cls._deep_merge(config, user_config)

        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = Config._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @classmethod
    def save_default_config(cls, filepath: str = 'config.json') -> None:
        """
        Save default configuration to file

        Args:
            filepath: Path to save config file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(cls.DEFAULT_CONFIG, f, indent=2)

        print(f"Default configuration saved to {filepath}")
