"""
Configuration utilities and logging setup
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import sys


class VulnerabilityDetectionError(Exception):
    """Base exception for vulnerability detection errors"""
    pass


class DataLoadError(VulnerabilityDetectionError):
    """Exception raised when data loading fails"""
    pass


class FeatureEngineeringError(VulnerabilityDetectionError):
    """Exception raised when feature engineering fails"""
    pass


class ModelTrainingError(VulnerabilityDetectionError):
    """Exception raised when model training fails"""
    pass


class PipelineError(VulnerabilityDetectionError):
    """Exception raised when pipeline execution fails"""
    pass


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
            'max_iter': 1000
        },
        'results': {
            'plots_path': 'results/evaluation_plots.png',
            'metrics_path': 'results/evaluation_results.json'
        },
        'training': {
            'nrows': None  # Load all data by default
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


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the project

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Prevent duplicate messages in case of imports
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
