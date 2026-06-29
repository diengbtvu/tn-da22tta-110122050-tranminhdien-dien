from typing import Any, Dict, List, Optional

from pydantic import BaseModel

SUPPORTED_MODELS = [
    "RandomForest",
    "SVM",
    "KNN",
    "DecisionTree",
    "XGBoost",
]


class ModelNameConfig(BaseModel):
    """Configuration for a single model training run."""

    model_name: str = "RandomForest"
    hyperparameters: Optional[Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------
    @classmethod
    def get_random_forest_config(
        cls,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> "ModelNameConfig":
        return cls(
            model_name="RandomForest",
            hyperparameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "random_state": random_state,
            },
        )

    @classmethod
    def get_svm_config(
        cls, C: float = 1.0, gamma: str = "scale", kernel: str = "rbf"
    ) -> "ModelNameConfig":
        """SVM (SVR) with RBF kernel — as used in Wang et al. (2024)."""
        return cls(
            model_name="SVM",
            hyperparameters={"C": C, "gamma": gamma, "kernel": kernel},
        )

    @classmethod
    def get_knn_config(cls, n_neighbors: int = 5) -> "ModelNameConfig":
        """KNN with k=5 — as used in Wang et al. (2024)."""
        return cls(
            model_name="KNN",
            hyperparameters={"n_neighbors": n_neighbors},
        )

    @classmethod
    def get_decision_tree_config(
        cls, max_depth: Optional[int] = None, random_state: int = 42
    ) -> "ModelNameConfig":
        return cls(
            model_name="DecisionTree",
            hyperparameters={"random_state": random_state, "max_depth": max_depth},
        )

    @classmethod
    def get_xgboost_config(
        cls, n_estimators: int = 100, learning_rate: float = 0.1
    ) -> "ModelNameConfig":
        return cls(
            model_name="XGBoost",
            hyperparameters={
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "random_state": 42,
            },
        )


# ---------------------------------------------------------------------------
class HydrogenExperimentConfig(BaseModel):
    """
    Hydrogen Production Rate experiment configuration.

    Based on Wang et al. (2024).
    """

    # 11 input features used in the paper
    feature_names: List[str] = [
        "pH",
        "VSS",
        "Ethanol",
        "Acetate",
        "Propionate",
        "Butyrate",
        "Sucrose_Degradation",
        "ORP_Mid",
        "ORP_Low",
        "VFA",
        "COD-O",
    ]

    # Target variable
    target_name: str = "HPR"  # Hydrogen Production Rate (L/h/L)

    # Input value ranges for validation
    feature_ranges: Dict[str, Dict[str, float]] = {
        "pH":                  {"min": 4.5,   "max": 9.0},
        "VSS":                 {"min": 0.0,   "max": 10.0},
        "Ethanol":             {"min": 0.0,   "max": 6000.0},
        "Acetate":             {"min": 0.0,   "max": 8000.0},
        "Propionate":          {"min": 0.0,   "max": 4000.0},
        "Butyrate":            {"min": 0.0,   "max": 120.0},
        "Sucrose_Degradation": {"min": 0.0,   "max": 100.0},
        "ORP_Mid":             {"min": -700.0, "max": -150.0},
        "ORP_Low":             {"min": -700.0, "max": -200.0},
        "VFA":                 {"min": 0.0,   "max": 30000.0},
        "COD-O":               {"min": 0.0,   "max": 50.0},
    }

    # Models to compare in experiment
    models_to_compare: List[str] = ["SVM", "DecisionTree", "RandomForest", "KNN", "XGBoost"]

    # Evaluation settings
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5

    # Deployment criteria (R² must exceed threshold)
    min_r2_for_deployment: float = 0.85

    # GRA settings
    gra_discrimination_coefficient: float = 0.5  # ξ = 0.5 (optimal)
