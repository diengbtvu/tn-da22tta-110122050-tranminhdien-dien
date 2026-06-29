import logging
from typing import Tuple

import mlflow
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from src.evaluation import MAE, MSE, RMSE, R2Score

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[
    Annotated[float, "r2_score"],
    Annotated[float, "rmse"],
    Annotated[float, "mse"],
    Annotated[float, "mae"],
]:
    """
    Evaluate regression model using R², RMSE, MSE, MAE.

    Metrics are logged to MLflow and returned for ZenML artifact tracking.
    """
    try:
        y_pred = model.predict(X_test)

        r2 = R2Score().calculate_score(y_test, y_pred)
        rmse = RMSE().calculate_score(y_test, y_pred)
        mse = MSE().calculate_score(y_test, y_pred)
        mae = MAE().calculate_score(y_test, y_pred)

        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)

        logging.info(
            f"Evaluation complete — R²={r2:.4f}, RMSE={rmse:.6f}, "
            f"MSE={mse:.6f}, MAE={mae:.6f}"
        )
        return r2, rmse, mse, mae

    except Exception as e:
        logging.error(f"Error in evaluate_model step: {e}")
        raise