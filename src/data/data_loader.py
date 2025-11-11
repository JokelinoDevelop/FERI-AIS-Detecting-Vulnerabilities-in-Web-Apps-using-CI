"""
Data loading and preprocessing for CSIC 2010 HTTP Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging
from sklearn.model_selection import train_test_split

from ..utils.config import get_logger

logger = get_logger(__name__)


class CSICDataLoader:
    """Data loader for CSIC 2010 HTTP Dataset"""

    # Expected column names (CSV has an empty first column that we skip)
    COLUMN_NAMES = [
        'Method', 'User-Agent', 'Pragma', 'Cache-Control', 'Accept',
        'Accept-encoding', 'Accept-charset', 'language', 'host', 'cookie',
        'content-type', 'connection', 'length', 'content', 'classification', 'URL'
    ]

    def __init__(self, data_path: str = "data/csic_database.csv"):
        """
        Initialize data loader

        Args:
            data_path: Path to the CSIC dataset CSV file
        """
        self.data_path = Path(data_path)

    def load_data(self, nrows: Optional[int] = None, stratified_sample: bool = True) -> pd.DataFrame:
        """
        Load the CSIC dataset

        Args:
            nrows: Number of rows to load (for testing). If None, loads all data.
            stratified_sample: If True and nrows is specified, ensures balanced classes.

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the loaded data doesn't have the expected structure
        """
        logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")

        try:
            # Load CSV data - it has headers
            df = pd.read_csv(
                self.data_path,
                nrows=None if stratified_sample and nrows else nrows,  # Load all if we need stratified sampling
                low_memory=False,
                quoting=1,  # QUOTE_ALL - handles quoted fields properly
                on_bad_lines='skip'  # Skip malformed lines
            )

            # Handle column mapping - CSV has 'Unnamed: 0' first column and 'lenght' instead of 'length'
            column_mapping = {
                'Unnamed: 0': None,  # Drop this column
                'lenght': 'length'   # Fix typo
            }
            df = df.rename(columns=column_mapping)

            # Drop the unnamed column if it exists
            if 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)

            # Ensure we have the expected columns
            expected_cols = self.COLUMN_NAMES
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing expected columns: {missing_cols}")

            # Validate essential columns exist
            required_cols = ['Method', 'URL', 'classification']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Perform stratified sampling if requested and nrows is specified
            if stratified_sample and nrows is not None and len(df) > nrows:
                logger.info(f"Performing stratified sampling to get {nrows} samples with balanced classes")
                # Sample proportionally from each class
                samples_per_class = nrows // 2  # Roughly balanced

                df_normal = df[df['classification'] == 0]
                df_anomalous = df[df['classification'] == 1]

                # Sample from each class
                n_normal = min(samples_per_class, len(df_normal))
                n_anomalous = min(samples_per_class, len(df_anomalous))

                # If we don't have enough anomalous samples, adjust
                if n_normal + n_anomalous < nrows:
                    n_normal = min(nrows - n_anomalous, len(df_normal))
                    n_anomalous = min(nrows - n_normal, len(df_anomalous))

                sampled_normal = df_normal.sample(n=n_normal, random_state=42) if n_normal > 0 else pd.DataFrame()
                sampled_anomalous = df_anomalous.sample(n=n_anomalous, random_state=42) if n_anomalous > 0 else pd.DataFrame()

                df = pd.concat([sampled_normal, sampled_anomalous], ignore_index=True)

            logger.info(f"Successfully loaded {len(df)} samples")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Classification distribution:\n{df['classification'].value_counts()}")

            return df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic preprocessing of the data

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")

        # Create a copy to avoid modifying original
        df_processed = df.copy()

        # Convert classification to binary (0 = Normal, 1 = Anomalous)
        # Handle both string and numeric classification values
        if df_processed['classification'].dtype == 'object':
            # String classification ('Normal', 'Anomalous')
            df_processed['target'] = df_processed['classification'].map({
                'Normal': 0,
                'Anomalous': 1
            }).fillna(df_processed['classification'].astype(int))  # Fallback to numeric
        else:
            # Already numeric (0, 1)
            df_processed['target'] = df_processed['classification'].astype(int)

        # Rename length column to content_length for consistency
        df_processed.rename(columns={'length': 'content_length'}, inplace=True)

        # Handle missing values - fill with empty strings for text columns
        text_columns = ['Method', 'User-Agent', 'Pragma', 'Cache-Control', 'Accept',
                       'Accept-encoding', 'Accept-charset', 'language', 'host', 'cookie',
                       'content-type', 'connection', 'content', 'URL']
        df_processed[text_columns] = df_processed[text_columns].fillna('')

        # Convert content_length to numeric, handling empty strings
        df_processed['content_length'] = pd.to_numeric(
            df_processed['content_length'], errors='coerce'
        ).fillna(0).astype(int)

        logger.info("Preprocessing completed")
        logger.info(f"Final shape: {df_processed.shape}")
        logger.info(f"Target distribution:\n{df_processed['target'].value_counts()}")

        return df_processed

    def get_train_test_split(self,
                           df: pd.DataFrame,
                           test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets

        Args:
            df: Preprocessed DataFrame with 'target' column
            test_size: Proportion of data for testing (0 < test_size < 1)
            random_state: Random state for reproducibility

        Returns:
            Tuple of (train_df, test_df)

        Raises:
            ValueError: If 'target' column is missing or test_size is invalid
        """
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column. Call preprocess_data() first.")

        if not (0 < test_size < 1):
            raise ValueError("test_size must be between 0 and 1")

        logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}")

        # Stratified split to maintain class distribution
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['target']
        )

        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        logger.info(f"Train class distribution:\n{train_df['target'].value_counts(normalize=True)}")
        logger.info(f"Test class distribution:\n{test_df['target'].value_counts(normalize=True)}")

        return train_df, test_df


def load_and_preprocess_data(data_path: str = "data/csic_database.csv",
                           test_size: float = 0.2,
                           nrows: Optional[int] = None,
                           stratified_sample: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and preprocess data in one call

    Args:
        data_path: Path to the dataset
        test_size: Proportion for test set
        nrows: Number of rows to load (for testing)
        stratified_sample: Whether to use stratified sampling for balanced classes

    Returns:
        Tuple of (train_df, test_df)
    """
    loader = CSICDataLoader(data_path)
    df = loader.load_data(nrows, stratified_sample)
    df_processed = loader.preprocess_data(df)
    train_df, test_df = loader.get_train_test_split(df_processed, test_size)

    return train_df, test_df
