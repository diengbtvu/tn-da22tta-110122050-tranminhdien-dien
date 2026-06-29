from typing import Optional

from zenml import pipeline

from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model
from steps.config import ModelNameConfig


@pipeline(
    enable_cache=False,
    settings={
        "experiment_tracker": {
            "name": "mlflow_tracker_hydrogen"
        }
    }
)
def train_pipeline(data_path: str, model_config: Optional[ModelNameConfig] = None):
    """
    ZenML training pipeline for Hydrogen Production Rate prediction.

    Steps:
    1. ingest_df    — load CSV or Excel file → DataFrame
    2. clean_df     — preprocess + MinMax normalise + train/test split
    3. train_model  — train chosen model (RF, SVM, KNN, DT, XGBoost)
    4. evaluate_model — compute R², RMSE, MSE, MAE and log to MLflow
    """
    # 1. Ingest
    df = ingest_df(data_path)

    # 2. Preprocess + split
    X_train, X_test, y_train, y_test = clean_df(df)

    # 3. Train
    if model_config is None:
        model_config = ModelNameConfig()  # default: RandomForest
    model = train_model(X_train, X_test, y_train, y_test, model_config)

    # 4. Evaluate
    r2_score, rmse, mse, mae = evaluate_model(model, X_test, y_test)

    print("\n=== Hydrogen Production Rate — Evaluation Results ===")
    print(f"  R² Score  : {r2_score:.4f}")
    print(f"  RMSE      : {rmse:.6f}")
    print(f"  MSE       : {mse:.6f}")
    print(f"  MAE       : {mae:.6f}")
