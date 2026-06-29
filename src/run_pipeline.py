import argparse
import os
import sys

from zenml.client import Client

from pipelines.training_pipeline import train_pipeline
from steps.config import ModelNameConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Hydrogen Production Rate training pipeline"
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset file (.csv or .xlsx)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='RandomForest',
        choices=['RandomForest', 'SVM', 'KNN', 'DecisionTree', 'XGBoost'],
        help='Model type to train (default: RandomForest)'
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        sys.exit(1)

    # Build model config
    model_map = {
        'RandomForest': ModelNameConfig.get_random_forest_config(n_estimators=100),
        'SVM': ModelNameConfig.get_svm_config(),
        'KNN': ModelNameConfig.get_knn_config(n_neighbors=5),
        'DecisionTree': ModelNameConfig.get_decision_tree_config(),
        'XGBoost': ModelNameConfig.get_xgboost_config(),
    }
    model_config = model_map.get(args.model, ModelNameConfig.get_random_forest_config())

    tracker = Client().active_stack.experiment_tracker
    print(f"ZenML MLflow Tracker URI: {tracker.get_tracking_uri()}")
    print(f"Starting training pipeline: model={args.model}, data={args.data}")

    run = train_pipeline(data_path=args.data, model_config=model_config)
    print(f"Pipeline completed! Run: {run.name}")
