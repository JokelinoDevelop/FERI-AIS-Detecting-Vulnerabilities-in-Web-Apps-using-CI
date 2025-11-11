#!/usr/bin/env python3
"""
Main script for Web Vulnerability Detection using Logistic Regression

This script provides a command-line interface to train and evaluate
a logistic regression model for detecting web vulnerabilities using
the CSIC 2010 HTTP Dataset.
"""

import argparse
import sys
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.models.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Web Vulnerability Detection using Logistic Regression'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='data/csic_database.csv',
        help='Path to the CSIC dataset CSV file'
    )

    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random state for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--save-model',
        type=str,
        default='models/vulnerability_detector.pkl',
        help='Path to save the trained model'
    )

    parser.add_argument(
        '--save-plots',
        type=str,
        default='results/evaluation_plots.png',
        help='Path to save evaluation plots'
    )

    parser.add_argument(
        '--save-results',
        type=str,
        default='results/evaluation_results.json',
        help='Path to save evaluation results as JSON'
    )

    parser.add_argument(
        '--nrows',
        type=int,
        default=None,
        help='Number of rows to load from dataset (for testing, default: all)'
    )

    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Display top 20 most important features'
    )

    args = parser.parse_args()

    try:
        # Initialize trainer
        logger.info("Initializing model trainer...")
        trainer = ModelTrainer(
            data_path=args.data_path,
            test_size=args.test_size,
            random_state=args.random_state
        )

        # Run full pipeline
        logger.info("Starting training pipeline...")
        results = trainer.run_full_pipeline(
            save_model_path=args.save_model,
            save_plots_path=args.save_plots,
            nrows=args.nrows
        )

        # Save evaluation results
        Path(args.save_results).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_results, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {
                'evaluation_metrics': results['evaluation_metrics'],
                'feature_importance_top_20': results['feature_importance'].head(20).to_dict('records')
            }
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Evaluation results saved to {args.save_results}")

        # Display feature importance if requested
        if args.feature_importance:
            print("\n" + "="*50)
            print("TOP 20 MOST IMPORTANT FEATURES")
            print("="*50)
            top_features = results['feature_importance'].head(20)
            for idx, row in top_features.iterrows():
                print(f"{idx+1:3d}. {row['feature']:<30} {row['coefficient']:+.4f}")

        # Display summary metrics
        eval_metrics = results['evaluation_metrics']
        print("\n" + "="*50)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*50)
        print(f"ROC AUC Score: {eval_metrics['roc_auc']:.4f}")
        print(f"Accuracy: {eval_metrics['classification_report']['accuracy']:.4f}")
        print(f"Precision (Anomalous): {eval_metrics['classification_report']['1']['precision']:.4f}")
        print(f"Recall (Anomalous): {eval_metrics['classification_report']['1']['recall']:.4f}")
        print(f"F1-Score (Anomalous): {eval_metrics['classification_report']['1']['f1-score']:.4f}")
        print("\nConfusion Matrix:")
        cm = eval_metrics['confusion_matrix']
        print(f"  True Normal:     {cm[0][0]:6d}    False Positive: {cm[0][1]:6d}")
        print(f"  False Negative:  {cm[1][0]:6d}    True Anomalous: {cm[1][1]:6d}")

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
