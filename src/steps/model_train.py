import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step
from zenml.client import Client

from .config import ModelNameConfig
from src.model_dev import (
    DecisionTreeModel,
    KNNModel,
    RandomForestModel,
    SVMModel,
    XGBoostModel,
)

experiment_tracker = Client().active_stack.experiment_tracker

MODEL_REGISTRY = {
    "RandomForest": RandomForestModel,
    "SVM": SVMModel,
    "KNN": KNNModel,
    "DecisionTree": DecisionTreeModel,
    "XGBoost": XGBoostModel,
}


@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> Annotated[RegressorMixin, "model"]:
    """
    Train a regression model for Hydrogen Production Rate prediction.

    Supports: RandomForest, SVM, KNN, DecisionTree, XGBoost.
    Uses mlflow.sklearn.autolog() for automatic experiment tracking.
    """
    try:
        mlflow.sklearn.autolog()

        if config.model_name not in MODEL_REGISTRY:
            raise ValueError(
                f"Model '{config.model_name}' not supported. "
                f"Available: {list(MODEL_REGISTRY.keys())}"
            )

        model_class = MODEL_REGISTRY[config.model_name]
        model_instance = model_class()

        hyperparams = getattr(config, 'hyperparameters', {}) or {}
        trained_model = model_instance.train(X_train, y_train, **hyperparams)

        logging.info(f"Successfully trained {config.model_name}")
        return trained_model

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise
