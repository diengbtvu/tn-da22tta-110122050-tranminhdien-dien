"""
Paper-aligned hydrogen training pipeline for the web demo product.

This module is intentionally based on the original implementation in:
    /tmp/Hydrogen-Prediction/hydrogen_production_prediction.py

The goal here is not to invent a new workflow. It preserves the original
methodology and repackages it into a reusable function for Flask:

1. Load uploaded CSV/XLS/XLSX data.
2. Apply the same preprocessing logic as the original file.
3. Normalise X and y with MinMaxScaler.
4. Run Grey Relational Analysis (GRA) with xi=0.5.
5. Train the five paper models.
6. Run sequential feature addition using the GRA ranking.
7. Pick the best model by highest R2, then lowest MSE.
8. Save the best model and related artifacts for downstream inference.
9. Compute SHAP values for the winning model.
"""

from __future__ import annotations

import argparse
import json
from io import BytesIO
import re
import shutil
from datetime import datetime
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .chart_explanation_payload import (
        CHART_EXPLANATION_PAYLOAD_FILENAME,
        build_chart_explanation_payload,
    )
except ImportError:
    from chart_explanation_payload import (  # type: ignore
        CHART_EXPLANATION_PAYLOAD_FILENAME,
        build_chart_explanation_payload,
    )

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBRegressor = None
    XGBOOST_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "ml_model"
LEGACY_MODEL_DIR = PROJECT_ROOT / "app" / "ml_model"
ARTIFACT_DIR = PROJECT_ROOT / "ml_train" / "artifacts"
LARAVEL_PUBLIC_MODEL_DIRS = [
    Path("/var/www/html/public/models"),
    PROJECT_ROOT.parent / "WebApp" / "public" / "models",
]

BEST_MODEL_FILENAME = "best_model.pkl"
BEST_SCALER_X_FILENAME = "best_scaler_X.pkl"
BEST_SCALER_Y_FILENAME = "best_scaler_Y.pkl"
SHARED_SCALER_FILENAME = "scaler.pkl"
BEST_MODEL_INFO_FILENAME = "best_model_info.txt"
GRA_RANKING_FILENAME = "gra_ranking.json"
INCREMENTAL_RESULTS_FILENAME = "table1_incremental_results.csv"
MODEL_COMPARISON_TABLE_FILENAME = "table_model_comparison.csv"
SHAP_VALUES_FILENAME = "best_model_shap_values.npy"
SHAP_IMPORTANCE_FILENAME = "best_model_shap_importance.csv"
TRAINING_SUMMARY_FILENAME = "best_model_summary.json"
FIG3A_GRA_RANKING_FILENAME = "fig3a_gra_ranking.png"
FIG3B_SHAP_ANALYSIS_FILENAME = "fig3b_shap_analysis.png"
FIG3_FEATURE_ANALYSIS_FILENAME = "fig3_feature_analysis.png"
FIG4_UNIVARIATE_ANALYSIS_FILENAME = "fig4_univariate_analysis.png"
FIG5_MODEL_COMPARISON_FILENAME = "fig5_model_comparison.png"
FIG6AB_MSE_R2_FEATURES_FILENAME = "fig6ab_mse_r2_features.png"
FIG6C_PREDICTION_TIME_FILENAME = "fig6c_prediction_time.png"
RESULTS_SUMMARY_FILENAME = "results_summary.txt"
CSV_SHEET_LABEL = "CSV"

TARGET_COLUMN = "HPR"
MIN_REQUIRED_ROWS = 6

EXCEL_COLUMN_RENAME = {
    "Unnamed: 0": "DateTime",
    "Unnamed: 1": "Day",
    "Unnamed: 2": "HRT",
    "TS degradation": "Sucrose_Degradation",
    "EtOH_Avg": "Ethanol",
    "HAc_Avg": "Acetate",
    "HPr_AVG": "Propionate",
    "HBu_Total": "Butyrate",
    "VFA_avg": "VFA",
    "ORP_中": "ORP_Mid",
    "ORP_下": "ORP_Low",
    "COD_O": "COD-O",
}

PAPER_FEATURE_COLUMNS = [
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

MODEL_NAME_TO_API_TYPE = {
    "SVM": "svm",
    "DT": "decision_tree",
    "RF": "random_forest",
    "KNN": "knn",
    "XGBoost": "xgboost",
}

CANONICAL_COLUMN_ALIASES = {
    "ph": "pH",
    "vss": "VSS",
    "ethanol": "Ethanol",
    "etoh": "Ethanol",
    "etoh_avg": "Ethanol",
    "acetate": "Acetate",
    "hac": "Acetate",
    "hac_avg": "Acetate",
    "propionate": "Propionate",
    "hpr_avg": "Propionate",
    "butyrate": "Butyrate",
    "hbu_total": "Butyrate",
    "sucrose_degradation": "Sucrose_Degradation",
    "sucrosedegradation": "Sucrose_Degradation",
    "ts_degradation": "Sucrose_Degradation",
    "orp_mid": "ORP_Mid",
    "orp_middle": "ORP_Mid",
    "orp_low": "ORP_Low",
    "orp_lower": "ORP_Low",
    "vfa": "VFA",
    "vfa_avg": "VFA",
    "cod_o": "COD-O",
    "codo": "COD-O",
    "cod_in": "COD-O",
    "hpr": "HPR",
}

ProgressCallback = Callable[[float, str], None]


class GreyRelationalAnalysis:
    """Original Grey Relational Analysis implementation from the source repo."""

    def __init__(self, xi: float = 0.5):
        self.xi = xi
        self.grey_relations: Dict[str, float] | None = None
        self.ranking: List[Tuple[str, float]] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str] | None = None,
    ) -> "GreyRelationalAnalysis":
        if feature_names is None:
            feature_names = [f"Feature_{idx}" for idx in range(X.shape[1])]

        # Step 1: MinMax normalisation to [0, 1] (paper Section 2.2)
        def _minmax(arr: np.ndarray) -> np.ndarray:
            if arr.ndim == 1:
                lo, hi = arr.min(), arr.max()
                return (arr - lo) / (hi - lo) if hi != lo else np.zeros_like(arr)
            lo = arr.min(axis=0)
            hi = arr.max(axis=0)
            denom = hi - lo
            with np.errstate(divide='ignore', invalid='ignore'):
                out = np.where(denom != 0, (arr - lo) / denom, 0.0)
            return np.nan_to_num(out, nan=0.0)

        y_normalized = _minmax(y)
        X_normalized = _minmax(X)

        # Steps 2-3: Deviation sequences & Grey Relational Coefficients
        # Compute all deviation sequences at once: Δ₀ᵢ(k) = |y*(k) − xᵢ*(k)|
        deltas = np.abs(y_normalized[:, None] - X_normalized)  # (n_samples, n_features)

        # Global Δmin and Δmax across ALL features and ALL samples
        # Δmin = min∀i min∀k Δ₀ᵢ(k),  Δmax = max∀i max∀k Δ₀ᵢ(k)
        delta_min = deltas.min()
        delta_max = deltas.max()

        grey_relations: Dict[str, float] = {}
        if delta_max == 0:
            for idx, feature_name in enumerate(feature_names):
                grey_relations[str(feature_name)] = 1.0
        else:
            coefficients = (delta_min + self.xi * delta_max) / (
                deltas + self.xi * delta_max
            )  # (n_samples, n_features)
            for idx, feature_name in enumerate(feature_names):
                grey_relations[str(feature_name)] = float(coefficients[:, idx].mean())

        self.grey_relations = grey_relations
        self.ranking = sorted(
            grey_relations.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return self

    def get_ranking(self) -> List[Tuple[str, float]]:
        if self.ranking is None:
            raise ValueError("GRA must be fitted before reading the ranking.")
        return self.ranking

    def plot_ranking(self, save_path: Path | str | None = None) -> plt.Figure:
        """Plot grey relation ranking with the same visual structure as the source repo."""
        ranking = self.get_ranking()
        features = [feature for feature, _score in ranking]
        values = [score for _feature, score in ranking]

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(features)))[::-1]

        bars = ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel("Grey Relation Degree", fontsize=12)
        ax.set_title("(a) Feature Grey Relation Ranking", fontsize=14)

        for bar, value in zip(bars, values):
            ax.text(
                value + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return fig


def _ensure_runtime_dependencies() -> None:
    if not XGBOOST_AVAILABLE:
        raise RuntimeError(
            "xgboost is required for the paper workflow but is not installed."
        )
    if not SHAP_AVAILABLE:
        raise RuntimeError(
            "shap is required for the paper workflow but is not installed."
        )


def _ensure_output_directories() -> None:
    for directory in (MODEL_DIR, LEGACY_MODEL_DIR, ARTIFACT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def _notify_progress(
    progress_callback: ProgressCallback | None,
    progress: float,
    message: str,
) -> None:
    if progress_callback is None:
        return
    progress_callback(float(progress), message)


def _get_file_source(file_input: Any) -> Tuple[Path | bytes, str]:
    if isinstance(file_input, (str, Path)):
        file_path = Path(file_input)
        return file_path, file_path.suffix.lower()

    filename = getattr(file_input, "filename", "") or getattr(file_input, "name", "")
    suffix = Path(filename).suffix.lower()
    stream = getattr(file_input, "stream", file_input)

    if hasattr(stream, "seek"):
        stream.seek(0)
    file_bytes = stream.read()
    if hasattr(stream, "seek"):
        stream.seek(0)

    return file_bytes, suffix


def _open_file_source(source: Path | bytes) -> Path | BytesIO:
    if isinstance(source, Path):
        return source
    return BytesIO(source)


def get_sheet_names(file_input: Any) -> List[str]:
    source, suffix = _get_file_source(file_input)
    if suffix == ".csv" or suffix == "":
        return [CSV_SHEET_LABEL]
    if suffix not in {".xls", ".xlsx"}:
        raise ValueError(f"Unsupported file format: {suffix}")
    excel_file = pd.ExcelFile(_open_file_source(source))
    return [str(sheet_name) for sheet_name in excel_file.sheet_names]


def _resolve_sheet_name(
    file_input: Any,
    sheet_name: str | None = None,
) -> Tuple[str | None, List[str], str]:
    source, suffix = _get_file_source(file_input)
    if suffix == ".csv" or suffix == "":
        return None, [CSV_SHEET_LABEL], suffix
    if suffix not in {".xls", ".xlsx"}:
        raise ValueError(f"Unsupported file format: {suffix}")

    excel_file = pd.ExcelFile(_open_file_source(source))
    sheet_names = [str(name) for name in excel_file.sheet_names]
    requested_sheet = (sheet_name or "").strip()
    if requested_sheet:
        if requested_sheet not in sheet_names:
            raise ValueError(
                f"Sheet '{requested_sheet}' was not found. Available sheets: {', '.join(sheet_names)}"
            )
        return requested_sheet, sheet_names, suffix

    if len(sheet_names) == 1:
        return sheet_names[0], sheet_names, suffix
    return None, sheet_names, suffix


def _read_uploaded_dataframe(
    file_input: Any,
    sheet_name: str | None = None,
) -> Tuple[pd.DataFrame, str | None, List[str], str]:
    """Read a CSV/XLS/XLSX upload from a path or Flask file object."""
    source, suffix = _get_file_source(file_input)
    if suffix == ".csv" or suffix == "":
        return pd.read_csv(_open_file_source(source)), None, [CSV_SHEET_LABEL], suffix

    if suffix in {".xls", ".xlsx"}:
        resolved_sheet, sheet_names, _ = _resolve_sheet_name(file_input, sheet_name)
        if resolved_sheet is None:
            raise ValueError(
                "This workbook contains multiple sheets. Please choose one sheet before continuing."
            )
        return (
            pd.read_excel(_open_file_source(source), sheet_name=resolved_sheet),
            resolved_sheet,
            sheet_names,
            suffix,
        )

    raise ValueError(f"Unsupported uploaded file format: {suffix}")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    standardized = df.copy()
    standardized.columns = [str(col).strip() for col in standardized.columns]
    standardized = standardized.rename(columns=EXCEL_COLUMN_RENAME)

    rename_map: Dict[str, str] = {}
    for column in standardized.columns:
        normalized_key = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        canonical = CANONICAL_COLUMN_ALIASES.get(normalized_key)
        if canonical:
            rename_map[column] = canonical

    standardized = standardized.rename(columns=rename_map)

    if standardized.columns.duplicated().any():
        deduplicated = pd.DataFrame(index=standardized.index)
        for column in pd.unique(standardized.columns):
            same_name = standardized.loc[:, standardized.columns == column]
            if same_name.shape[1] == 1:
                deduplicated[column] = same_name.iloc[:, 0]
            else:
                deduplicated[column] = same_name.bfill(axis=1).iloc[:, 0]
        standardized = deduplicated

    return standardized


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if value is None:
        return None
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if pd.isna(value):
        return None
    return value


def _preview_to_records(df: pd.DataFrame, preview_rows: int) -> List[Dict[str, Any]]:
    preview_df = df.head(preview_rows).astype(object)
    preview_df = preview_df.where(pd.notna(preview_df), None)
    return _json_safe_value(preview_df.to_dict(orient="records"))


def _filter_observation_rows(
    standardized_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Keep only actual observation rows before interpolation.

    Some Excel workbooks append footer/statistics rows below the training data.
    Those rows should never be interpolated into the training set. When the
    source workbook contains timeline columns from the original experiment
    sheets, use them to isolate the real observations first.
    """
    source_row_count = int(len(standardized_df))
    working_df = standardized_df.dropna(how="all").reset_index(drop=True)

    filter_strategy: str | None = None
    row_mask = pd.Series(True, index=working_df.index, dtype=bool)

    if "DateTime" in working_df.columns:
        datetime_values = pd.to_datetime(working_df["DateTime"], errors="coerce")
        datetime_mask = datetime_values.notna()
        if int(datetime_mask.sum()) >= MIN_REQUIRED_ROWS:
            row_mask = datetime_mask
            filter_strategy = "DateTime"

    if filter_strategy is None and {"Day", "HRT"}.issubset(working_df.columns):
        day_values = pd.to_numeric(working_df["Day"], errors="coerce")
        hrt_values = pd.to_numeric(working_df["HRT"], errors="coerce")
        timeline_mask = day_values.notna() & hrt_values.notna()
        if int(timeline_mask.sum()) >= MIN_REQUIRED_ROWS:
            row_mask = timeline_mask
            filter_strategy = "Day+HRT"

    filtered_df = working_df.loc[row_mask].reset_index(drop=True)
    metadata = {
        "source_rows": source_row_count,
        "candidate_training_rows": int(len(filtered_df)),
        "filtered_out_rows": int(source_row_count - len(filtered_df)),
        "row_filter_strategy": filter_strategy,
    }
    return filtered_df, metadata


def _preprocess_training_dataframe(
    standardized_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    observation_df, metadata = _filter_observation_rows(standardized_df)

    missing_columns = [
        column
        for column in [*PAPER_FEATURE_COLUMNS, TARGET_COLUMN]
        if column not in observation_df.columns
    ]
    if missing_columns:
        available = ", ".join(observation_df.columns.tolist())
        missing = ", ".join(missing_columns)
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {available}"
        )

    df_subset = observation_df[[*PAPER_FEATURE_COLUMNS, TARGET_COLUMN]].copy()
    for column in df_subset.columns:
        df_subset[column] = pd.to_numeric(df_subset[column], errors="coerce")

    df_subset = df_subset.interpolate(method="linear", limit_direction="both")
    df_subset = df_subset.dropna().reset_index(drop=True)

    if len(df_subset) < MIN_REQUIRED_ROWS:
        raise ValueError(
            "Training requires at least 6 valid rows after preprocessing so the "
            "80/20 split can compute R2 on the test set."
        )

    metadata = {
        **metadata,
        "rows_after_preprocessing": int(len(df_subset)),
    }
    return df_subset, metadata


def load_and_preprocess_data(
    file_input: Any,
    sheet_name: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    """
    Load and preprocess the training dataset following the original script.

    This keeps the original feature set:
        pH, VSS, Ethanol, Acetate, Propionate, Butyrate,
        Sucrose_Degradation, ORP_Mid, ORP_Low, VFA, COD-O
    """
    print("Loading dataset...")
    raw_df, _resolved_sheet, _sheet_names, _suffix = _read_uploaded_dataframe(
        file_input,
        sheet_name=sheet_name,
    )
    df = _standardize_columns(raw_df)
    df_subset, preprocessing_metadata = _preprocess_training_dataframe(df)

    filtered_out_rows = preprocessing_metadata["filtered_out_rows"]
    row_filter_strategy = preprocessing_metadata["row_filter_strategy"]
    if filtered_out_rows > 0:
        strategy_label = row_filter_strategy or "non-empty-row"
        print(
            f"Ignoring {filtered_out_rows} non-training/footer rows using "
            f"the {strategy_label} observation filter."
        )

    print(f"Dataset shape after preprocessing: {df_subset.shape}")
    X = df_subset[PAPER_FEATURE_COLUMNS].values
    y = df_subset[TARGET_COLUMN].values
    return X, y, list(PAPER_FEATURE_COLUMNS), df_subset


def inspect_dataset_file(
    file_input: Any,
    sheet_name: str | None = None,
    preview_rows: int = 10,
) -> Dict[str, Any]:
    source, suffix = _get_file_source(file_input)
    if suffix in {".xls", ".xlsx"}:
        resolved_sheet, sheet_names, _ = _resolve_sheet_name(file_input, sheet_name)
        if resolved_sheet is None:
            default_preview_sheet = sheet_names[0]
            preview_df = pd.read_excel(
                _open_file_source(source),
                sheet_name=default_preview_sheet,
            )
            standardized_preview_df = _standardize_columns(preview_df)
            filtered_preview_df, preview_metadata = _filter_observation_rows(
                standardized_preview_df
            )
            return _json_safe_value({
                "status": "success",
                "detected_format": suffix.lstrip("."),
                "sheet_names": sheet_names,
                "requires_sheet_selection": True,
                "selected_sheet": None,
                "preview_sheet": default_preview_sheet,
                "preview_columns": [str(column) for column in filtered_preview_df.columns],
                "preview_rows": _preview_to_records(filtered_preview_df, preview_rows),
                "required_columns": [*PAPER_FEATURE_COLUMNS, TARGET_COLUMN],
                "missing_columns": [],
                "rows_after_preprocessing": 0,
                "source_rows": preview_metadata["source_rows"],
                "candidate_training_rows": preview_metadata["candidate_training_rows"],
                "filtered_out_rows": preview_metadata["filtered_out_rows"],
                "row_filter_strategy": preview_metadata["row_filter_strategy"],
                "minimum_required_rows": MIN_REQUIRED_ROWS,
                "is_valid": False,
                "validation_error": "Please choose a sheet before uploading this workbook.",
            })
    else:
        resolved_sheet = None
        sheet_names = [CSV_SHEET_LABEL]

    raw_df, _resolved_sheet, _sheet_names, detected_suffix = _read_uploaded_dataframe(
        file_input,
        sheet_name=resolved_sheet,
    )
    standardized_df = _standardize_columns(raw_df)
    filtered_preview_df, preview_metadata = _filter_observation_rows(standardized_df)

    required_columns = [*PAPER_FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in filtered_preview_df.columns
    ]

    rows_after_preprocessing = 0
    validation_error = None
    is_valid = False
    try:
        _X, _y, _feature_names, cleaned_df = load_and_preprocess_data(
            file_input,
            sheet_name=resolved_sheet,
        )
        rows_after_preprocessing = int(len(cleaned_df))
        is_valid = rows_after_preprocessing >= MIN_REQUIRED_ROWS and not missing_columns
    except Exception as exc:
        validation_error = str(exc)

    return _json_safe_value({
        "status": "success",
        "detected_format": detected_suffix.lstrip(".") or "csv",
        "sheet_names": sheet_names,
        "requires_sheet_selection": False,
        "selected_sheet": resolved_sheet,
        "preview_sheet": resolved_sheet or CSV_SHEET_LABEL,
        "preview_columns": [str(column) for column in filtered_preview_df.columns],
        "preview_rows": _preview_to_records(filtered_preview_df, preview_rows),
        "required_columns": required_columns,
        "missing_columns": missing_columns,
        "rows_after_preprocessing": rows_after_preprocessing,
        "source_rows": preview_metadata["source_rows"],
        "candidate_training_rows": preview_metadata["candidate_training_rows"],
        "filtered_out_rows": preview_metadata["filtered_out_rows"],
        "row_filter_strategy": preview_metadata["row_filter_strategy"],
        "minimum_required_rows": MIN_REQUIRED_ROWS,
        "is_valid": is_valid,
        "validation_error": validation_error,
    })


def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Apply MinMaxScaler to X and y exactly like the original script."""
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_normalized = scaler_X.fit_transform(X)
    y_normalized = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    return X_normalized, y_normalized, scaler_X, scaler_y


def build_models() -> Dict[str, Any]:
    """Build the five paper models with the same defaults as the source repo."""
    models: Dict[str, Any] = {
        "SVM": SVR(kernel="rbf", C=1.0, gamma="scale"),
        "DT": DecisionTreeRegressor(random_state=42),
        "RF": RandomForestRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=42,
            verbosity=0,
        ),
    }
    return models


def train_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Train and evaluate the five paper models on the 80/20 split."""
    models = build_models()
    results: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}

    print("=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse_train = float(mean_squared_error(y_train, y_pred_train))
        mse_test = float(mean_squared_error(y_test, y_pred_test))
        mae_train = float(mean_absolute_error(y_train, y_pred_train))
        mae_test = float(mean_absolute_error(y_test, y_pred_test))
        r2_train = float(r2_score(y_train, y_pred_train))
        r2_test = float(r2_score(y_test, y_pred_test))

        results[model_name] = {
            "MSE_train": mse_train,
            "MSE_test": mse_test,
            "MAE_train": mae_train,
            "MAE_test": mae_test,
            "RMSE_train": float(np.sqrt(mse_train)),
            "RMSE_test": float(np.sqrt(mse_test)),
            "R2_train": r2_train,
            "R2_test": r2_test,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
        }

        print(
            f"  Train - MSE: {mse_train:.6f}, R2: {r2_train:.4f}\n"
            f"  Test  - MSE: {mse_test:.6f}, R2: {r2_test:.4f}"
        )

    return results, trained_models


def _save_figure(fig: plt.Figure, save_path: Path | str) -> None:
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _paper_incremental_results_view(results_table: pd.DataFrame) -> pd.DataFrame:
    """Return the paper-style R2-only table used by the original plots and summary."""
    paper_table = pd.DataFrame({"n_features": results_table["n_features"].astype(int)})
    for model_name in build_models().keys():
        r2_column = f"{model_name}_R2"
        if r2_column in results_table.columns:
            paper_table[model_name] = results_table[r2_column]
    return paper_table


def plot_model_comparison(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot the paper-style model comparison figure and scatter panels."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = np.asarray(axes).flatten()

    for index, (model_name, result) in enumerate(results.items()):
        if index >= len(axes) - 1:
            break

        ax = axes[index]
        y_pred_test = np.asarray(result["y_pred_test"])
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=20, label="Data Points")

        if len(y_test) > 1:
            z = np.polyfit(y_test, y_pred_test, 1)
            p = np.poly1d(z)
            x_line = np.linspace(float(np.min(y_test)), float(np.max(y_test)), 100)
            ax.plot(x_line, p(x_line), "r-", linewidth=2, label="Best Linear Fit")

        ax.set_xlabel("Measured", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.set_title(f"{model_name}-test\nR²={result['R2_test']:.5f}", fontsize=12)
        ax.legend(fontsize=8)

    ax = axes[-1]
    model_names = list(results.keys())
    r2_values = [float(results[name]["R2_test"]) for name in model_names]
    colors = ["steelblue", "coral", "seagreen", "mediumpurple", "goldenrod"][: len(model_names)]
    bars = ax.bar(model_names, r2_values, color=colors)
    ax.set_ylabel("R squared", fontsize=10)
    ax.set_xlabel("Kinds of Machine Learning", fontsize=10)
    ax.set_title("(f) R² Comparison", fontsize=12)
    ax.set_ylim(0.0, max(1.0, max(r2_values) + 0.1))

    for bar, value in zip(bars, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.5f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_mse_r2_vs_features(
    results_table: pd.DataFrame,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot the paper's Figure 6a/6b using the real incremental MSE and R2 values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    feature_counts = results_table["n_features"].values
    model_names = [model_name for model_name in build_models().keys() if f"{model_name}_R2" in results_table.columns]
    markers = ["s", "^", "v", "D", "o"]
    colors = ["blue", "orange", "green", "red", "purple"]

    ax = axes[0]
    for index, model_name in enumerate(model_names):
        ax.plot(
            feature_counts,
            results_table[f"{model_name}_MSE"].values,
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            label=model_name,
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Feature Number", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("(a) MSE vs Feature Number", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for index, model_name in enumerate(model_names):
        ax.plot(
            feature_counts,
            results_table[f"{model_name}_R2"].values,
            marker=markers[index % len(markers)],
            color=colors[index % len(colors)],
            label=model_name,
            linewidth=2,
            markersize=8,
        )
    ax.set_xlabel("Feature Number", fontsize=12)
    ax.set_ylabel("R Squared", fontsize=12)
    ax.set_title("(b) R² vs Feature Number", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(min(0.0, float(results_table[[f"{name}_R2" for name in model_names]].min().min()) - 0.05), 1.0)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_univariate_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot the univariate feature-vs-target analysis from the source repo."""
    n_features = min(8, len(feature_cols))
    n_rows = (n_features + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for index, feature in enumerate(feature_cols[:n_features]):
        ax = axes[index]
        x = pd.to_numeric(df[feature], errors="coerce")
        y = pd.to_numeric(df[target_col], errors="coerce")
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        ax.scatter(x_clean, y_clean, alpha=0.5, s=30, c="steelblue")
        if len(x_clean) > 2:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_line = np.linspace(float(x_clean.min()), float(x_clean.max()), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=2)

        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel("HPR (L/h/L)", fontsize=10)
        ax.set_title(f"({chr(97 + index)})", fontsize=12)
        ax.grid(True, alpha=0.3)

    for index in range(n_features, len(axes)):
        axes[index].set_visible(False)

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_prediction_over_time(
    y_actual: np.ndarray,
    y_predicted: np.ndarray,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Plot the paper-style prediction-over-time chart."""
    fig, ax = plt.subplots(figsize=(14, 6))
    days = np.arange(len(y_actual))

    ax.plot(days, y_actual, "b-", label="Experimental", alpha=0.7, linewidth=1)
    ax.fill_between(days, 0, y_actual, alpha=0.3, color="blue")
    ax.plot(days, y_predicted, "r-", label="Predicted", alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Time (day)", fontsize=12)
    ax.set_ylabel("Hydrogen Production Rate (L/h/L)", fontsize=12)
    ax.set_title("(c) Predicted Hydrogen Production Rate", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel("Hydrogen Production Rate (L/h/L)", fontsize=12)

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def incremental_feature_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    feature_ranking: Sequence[Tuple[str, float]],
) -> pd.DataFrame:
    """
    Run sequential feature addition following the GRA order.

    The original script only kept R2 in the wide table.
    For the web product we also keep MSE for each model at each feature step.
    """
    print("=" * 60)
    print("INCREMENTAL FEATURE ANALYSIS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    ranked_features = [feature for feature, _score in feature_ranking]
    feature_indices = [feature_names.index(feature) for feature in ranked_features]

    rows: List[Dict[str, Any]] = []
    models = build_models()

    for n_features in range(1, len(feature_indices) + 1):
        selected_features = ranked_features[:n_features]
        selected_indices = feature_indices[:n_features]
        X_train_subset = X_train_norm[:, selected_indices]
        X_test_subset = X_test_norm[:, selected_indices]

        row: Dict[str, Any] = {
            "n_features": n_features,
            "feature_subset": " -> ".join(selected_features),
        }

        summary_parts: List[str] = []
        for model_name, model in models.items():
            model_clone = clone(model)
            model_clone.fit(X_train_subset, y_train)
            y_pred = model_clone.predict(X_test_subset)

            mse_value = float(mean_squared_error(y_test, y_pred))
            r2_value = float(r2_score(y_test, y_pred))

            row[f"{model_name}_R2"] = r2_value
            row[f"{model_name}_MSE"] = mse_value
            summary_parts.append(
                f"{model_name}: R2={r2_value:.4f}, MSE={mse_value:.6f}"
            )

        rows.append(row)
        print(f"Features {n_features}/{len(feature_indices)}: {' | '.join(summary_parts)}")

    return pd.DataFrame(rows)


def _sample_rows(array: np.ndarray, sample_size: int) -> np.ndarray:
    if array.shape[0] <= sample_size:
        return array

    rng = np.random.default_rng(42)
    sampled_indices = np.sort(
        rng.choice(array.shape[0], size=sample_size, replace=False)
    )
    return array[sampled_indices]


def perform_shap_analysis(
    model: Any,
    X_train: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    """Compute SHAP values for the winning model and return the exact explained matrix."""
    if not SHAP_AVAILABLE:
        raise RuntimeError("SHAP is not installed. Add shap to predict-service/requirements.txt.")

    print("Computing SHAP values for the winning model...")
    if hasattr(model, "feature_importances_"):
        shap_input = X_train
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(shap_input)
    else:
        background_size = min(25, X_train.shape[0])
        explain_size = min(25, X_train.shape[0])
        background = _sample_rows(X_train, background_size)
        shap_input = _sample_rows(X_train, explain_size)
        print(
            "Using sampled Kernel SHAP for the winning non-tree model "
            f"(background={background.shape[0]}, explained_rows={shap_input.shape[0]})."
        )
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(shap_input, nsamples=100)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    shap_importance = (
        pd.DataFrame(
            {
                "feature": list(feature_names),
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_array, shap_importance, np.asarray(shap_input)


def plot_shap_analysis(
    shap_values: np.ndarray,
    shap_input: np.ndarray,
    feature_names: Sequence[str],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Render the SHAP summary plot with the original output filename."""
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_array,
        np.asarray(shap_input),
        feature_names=list(feature_names),
        plot_type="dot",
        show=False,
    )
    fig = plt.gcf()
    plt.title("(b) Shapley Value Analysis", fontsize=14)
    plt.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def plot_combined_feature_analysis(
    gra_ranking: Sequence[Tuple[str, float]],
    shap_values: np.ndarray,
    shap_input: np.ndarray,
    feature_names: Sequence[str],
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Render the combined Figure 3 panel matching the original repo outputs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax = axes[0]
    features = [feature for feature, _score in gra_ranking]
    values = [float(score) for _feature, score in gra_ranking]
    y_pos = np.arange(len(features))
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(features)))[::-1]

    bars = ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Grey Relation Degree", fontsize=12)
    ax.set_title("(a) Feature Grey Relation Ranking", fontsize=14)

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    ax = axes[1]
    shap_array = np.asarray(shap_values)
    input_array = np.asarray(shap_input)
    mean_shap = np.abs(shap_array).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[::-1]
    sorted_features = [feature_names[index] for index in sorted_idx]

    rng = np.random.default_rng(42)
    for plot_index, feature_index in enumerate(sorted_idx):
        feature_shap = shap_array[:, feature_index]
        feature_vals = input_array[:, feature_index]
        norm_vals = (feature_vals - feature_vals.min()) / (
            feature_vals.max() - feature_vals.min() + 1e-10
        )
        colors = plt.cm.RdBu_r(norm_vals)
        jitter = rng.normal(0, 0.1, len(feature_shap))
        ax.scatter(
            feature_shap,
            np.full(len(feature_shap), plot_index) + jitter,
            c=colors,
            alpha=0.5,
            s=10,
        )

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features)
    ax.set_xlabel("SHAP value", fontsize=12)
    ax.set_title("(b) Shapley Analysis", fontsize=14)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=ax, label="Feature value")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    if save_path:
        _save_figure(fig, save_path)
    return fig


def _save_model_scatter_plots(
    results: Dict[str, Dict[str, Any]],
    y_test: np.ndarray,
    output_dir: Path,
) -> Dict[str, str]:
    """Save the individual per-model scatter plots produced by the source repo."""
    saved_paths: Dict[str, str] = {}

    for model_name, result in results.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        y_pred_test = np.asarray(result["y_pred_test"])
        ax.scatter(y_test, y_pred_test, alpha=0.6, s=40, c="steelblue")

        if len(y_test) > 1:
            z = np.polyfit(y_test, y_pred_test, 1)
            p = np.poly1d(z)
            x_line = np.linspace(float(np.min(y_test)), float(np.max(y_test)), 100)
            ax.plot(
                x_line,
                p(x_line),
                "r-",
                linewidth=2,
                label=f"Linear Fit: y={z[0]:.2f}x+{z[1]:.3f}",
            )

        ax.set_xlabel("Measured", fontsize=12)
        ax.set_ylabel("Predicted", fontsize=12)
        ax.set_title(f"{model_name} Model\nR² = {result['R2_test']:.5f}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        scatter_path = output_dir / f"model_{model_name.lower()}_scatter.png"
        fig.tight_layout()
        _save_figure(fig, scatter_path)
        saved_paths[f"model_{model_name.lower()}_scatter_path"] = str(scatter_path)

    return saved_paths


def create_results_summary(
    results: Dict[str, Dict[str, Any]],
    gra_ranking: Sequence[Tuple[str, float]],
    incremental_results: pd.DataFrame,
) -> str:
    """Create the text summary file using the same sections as the source repo."""
    summary_lines = ["=" * 70, "RESULTS SUMMARY", "=" * 70]

    summary_lines.append("\n1. GREY RELATIONAL ANALYSIS - FEATURE RANKING")
    summary_lines.append("-" * 50)
    for index, (feature, value) in enumerate(gra_ranking, 1):
        summary_lines.append(f"   {index}. {feature}: {value:.4f}")

    summary_lines.append("\n2. MODEL PERFORMANCE (Test Set)")
    summary_lines.append("-" * 50)
    summary_lines.append(f"{'Model':<12} {'MSE':>12} {'R²':>12}")
    summary_lines.append("-" * 36)
    for model_name, metrics in sorted(
        results.items(),
        key=lambda item: item[1]["R2_test"],
        reverse=True,
    ):
        summary_lines.append(
            f"{model_name:<12} {metrics['MSE_test']:>12.6f} {metrics['R2_test']:>12.5f}"
        )

    summary_lines.append("\n3. BEST MODEL")
    summary_lines.append("-" * 50)
    best_model = max(results.items(), key=lambda item: item[1]["R2_test"])
    summary_lines.append(f"   {best_model[0]} with R² = {best_model[1]['R2_test']:.5f}")
    summary_lines.append(f"   and MSE = {best_model[1]['MSE_test']:.6f}")

    summary_lines.append("\n4. INCREMENTAL FEATURE ANALYSIS")
    summary_lines.append("-" * 50)
    summary_lines.append(_paper_incremental_results_view(incremental_results).to_string(index=False))

    return "\n".join(summary_lines)


def _ranking_to_serializable(
    ranking: Sequence[Tuple[str, float]],
) -> List[Dict[str, Any]]:
    return [
        {
            "rank": index + 1,
            "feature": feature,
            "score": float(score),
        }
        for index, (feature, score) in enumerate(ranking)
    ]


def _select_best_model(
    results: Dict[str, Dict[str, float]],
) -> Tuple[str, Dict[str, float]]:
    best_model_name, best_metrics = min(
        results.items(),
        key=lambda item: (-item[1]["R2_test"], item[1]["MSE_test"]),
    )
    return best_model_name, best_metrics


def _build_model_comparison_table(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, metrics in results.items():
        rows.append(
            {
                "model": model_name,
                "r2_score": float(metrics["R2_test"]),
                "rmse": float(metrics["RMSE_test"]),
                "mse": float(metrics["MSE_test"]),
                "mae": float(metrics["MAE_test"]),
            }
        )
    return pd.DataFrame(rows)


def _save_artifacts(
    best_model_name: str,
    best_model: Any,
    best_metrics: Dict[str, float],
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    results: Dict[str, Dict[str, Any]],
    gra: GreyRelationalAnalysis,
    gra_ranking: Sequence[Tuple[str, float]],
    incremental_results: pd.DataFrame,
    shap_values: np.ndarray,
    shap_input: np.ndarray,
    shap_importance: pd.DataFrame,
    X_norm: np.ndarray,
    y_norm: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    cleaned_df: pd.DataFrame,
) -> Dict[str, str]:
    _ensure_output_directories()

    model_info_text = (
        f"Model thắng cuộc: {best_model_name} - "
        f"R²={best_metrics['R2_test']:.6f} - "
        f"MSE={best_metrics['MSE_test']:.6f}"
    )

    saved_paths: Dict[str, str] = {}
    for model_dir in (MODEL_DIR, LEGACY_MODEL_DIR):
        joblib.dump(best_model, model_dir / BEST_MODEL_FILENAME)
        joblib.dump(scaler_X, model_dir / BEST_SCALER_X_FILENAME)
        joblib.dump(scaler_y, model_dir / BEST_SCALER_Y_FILENAME)
        joblib.dump(scaler_X, model_dir / SHARED_SCALER_FILENAME)
        (model_dir / BEST_MODEL_INFO_FILENAME).write_text(
            model_info_text,
            encoding="utf-8",
        )

    mirrored_anywhere = False
    for public_model_dir in LARAVEL_PUBLIC_MODEL_DIRS:
        try:
            public_model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(MODEL_DIR / BEST_MODEL_FILENAME, public_model_dir / BEST_MODEL_FILENAME)
            shutil.copy2(
                MODEL_DIR / BEST_SCALER_X_FILENAME,
                public_model_dir / BEST_SCALER_X_FILENAME,
            )
            shutil.copy2(
                MODEL_DIR / BEST_SCALER_Y_FILENAME,
                public_model_dir / BEST_SCALER_Y_FILENAME,
            )
            shutil.copy2(
                MODEL_DIR / SHARED_SCALER_FILENAME,
                public_model_dir / SHARED_SCALER_FILENAME,
            )
            shutil.copy2(
                MODEL_DIR / BEST_MODEL_INFO_FILENAME,
                public_model_dir / BEST_MODEL_INFO_FILENAME,
            )
            mirrored_anywhere = True
        except Exception as copy_error:
            print(f"Warning: failed to mirror artifacts to {public_model_dir}: {copy_error}")

    if not mirrored_anywhere:
        print("Warning: best-model artifacts were not mirrored to any Laravel public/models path.")

    gra_ranking_path = ARTIFACT_DIR / GRA_RANKING_FILENAME
    incremental_results_path = ARTIFACT_DIR / INCREMENTAL_RESULTS_FILENAME
    model_comparison_path = ARTIFACT_DIR / MODEL_COMPARISON_TABLE_FILENAME
    shap_values_path = ARTIFACT_DIR / SHAP_VALUES_FILENAME
    shap_importance_path = ARTIFACT_DIR / SHAP_IMPORTANCE_FILENAME
    summary_path = ARTIFACT_DIR / TRAINING_SUMMARY_FILENAME
    fig3a_path = ARTIFACT_DIR / FIG3A_GRA_RANKING_FILENAME
    fig3b_path = ARTIFACT_DIR / FIG3B_SHAP_ANALYSIS_FILENAME
    fig3_feature_path = ARTIFACT_DIR / FIG3_FEATURE_ANALYSIS_FILENAME
    fig4_path = ARTIFACT_DIR / FIG4_UNIVARIATE_ANALYSIS_FILENAME
    fig5_path = ARTIFACT_DIR / FIG5_MODEL_COMPARISON_FILENAME
    fig6ab_path = ARTIFACT_DIR / FIG6AB_MSE_R2_FEATURES_FILENAME
    fig6c_path = ARTIFACT_DIR / FIG6C_PREDICTION_TIME_FILENAME
    results_summary_path = ARTIFACT_DIR / RESULTS_SUMMARY_FILENAME

    with gra_ranking_path.open("w", encoding="utf-8") as handle:
        json.dump(_ranking_to_serializable(gra_ranking), handle, indent=2)
    incremental_results.to_csv(incremental_results_path, index=False)
    _build_model_comparison_table(results).to_csv(model_comparison_path, index=False)
    np.save(shap_values_path, shap_values)
    shap_importance.to_csv(shap_importance_path, index=False)

    gra.plot_ranking(fig3a_path)
    plot_shap_analysis(shap_values, shap_input, feature_names, fig3b_path)
    plot_combined_feature_analysis(
        gra_ranking=gra_ranking,
        shap_values=shap_values,
        shap_input=shap_input,
        feature_names=feature_names,
        save_path=fig3_feature_path,
    )
    plot_univariate_analysis(
        cleaned_df,
        TARGET_COLUMN,
        feature_names,
        save_path=fig4_path,
    )
    plot_model_comparison(results, y_test, save_path=fig5_path)
    plot_mse_r2_vs_features(incremental_results, save_path=fig6ab_path)
    y_pred_all = np.asarray(best_model.predict(X_norm))
    plot_prediction_over_time(y_norm, y_pred_all, save_path=fig6c_path)

    scatter_paths = _save_model_scatter_plots(results, y_test, ARTIFACT_DIR)

    results_summary_text = create_results_summary(
        results=results,
        gra_ranking=gra_ranking,
        incremental_results=incremental_results,
    )
    results_summary_path.write_text(results_summary_text, encoding="utf-8")

    saved_paths["best_model_path"] = str(MODEL_DIR / BEST_MODEL_FILENAME)
    saved_paths["best_scaler_x_path"] = str(MODEL_DIR / BEST_SCALER_X_FILENAME)
    saved_paths["best_scaler_y_path"] = str(MODEL_DIR / BEST_SCALER_Y_FILENAME)
    saved_paths["shared_scaler_path"] = str(MODEL_DIR / SHARED_SCALER_FILENAME)
    saved_paths["best_model_info_path"] = str(MODEL_DIR / BEST_MODEL_INFO_FILENAME)
    saved_paths["gra_ranking_path"] = str(gra_ranking_path)
    saved_paths["incremental_results_path"] = str(incremental_results_path)
    saved_paths["model_comparison_table_path"] = str(model_comparison_path)
    saved_paths["shap_values_path"] = str(shap_values_path)
    saved_paths["shap_importance_path"] = str(shap_importance_path)
    saved_paths["fig3a_gra_ranking_path"] = str(fig3a_path)
    saved_paths["fig3b_shap_analysis_path"] = str(fig3b_path)
    saved_paths["fig3_feature_analysis_path"] = str(fig3_feature_path)
    saved_paths["fig4_univariate_analysis_path"] = str(fig4_path)
    saved_paths["fig5_model_comparison_path"] = str(fig5_path)
    saved_paths["fig6ab_mse_r2_features_path"] = str(fig6ab_path)
    saved_paths["fig6c_prediction_time_path"] = str(fig6c_path)
    saved_paths["results_summary_text_path"] = str(results_summary_path)
    saved_paths["summary_path"] = str(summary_path)
    saved_paths.update(scatter_paths)

    return saved_paths


def train_and_save_model(
    file_input: Any,
    sheet_name: str | None = None,
    progress_callback: ProgressCallback | None = None,
    training_run_id: str | None = None,
    return_runtime: bool = False,
) -> Dict[str, Any]:
    """
    Train the complete paper workflow and save the best model.

    Args:
        file_input:
            A file path, Flask FileStorage object, or file-like object.

    Returns:
        Dict with best model metadata and saved artifact paths.
    """
    _ensure_runtime_dependencies()

    print("=" * 70)
    print("HYDROGEN PRODUCTION TRAINING PIPELINE")
    print("=" * 70)
    print("Methodology aligned with the original Hydrogen-Prediction repo.")
    print("=" * 70)

    _notify_progress(progress_callback, 5, "Loading dataset and validating columns...")
    X, y, feature_names, df = load_and_preprocess_data(
        file_input,
        sheet_name=sheet_name,
    )
    resolved_sheet, _sheet_names, _detected_suffix = _resolve_sheet_name(
        file_input,
        sheet_name,
    )

    print("=" * 60)
    print("STEP 1: DATA NORMALIZATION (MinMaxScaler)")
    print("=" * 60)
    _notify_progress(progress_callback, 15, "Applying MinMax normalization...")
    X_norm, y_norm, scaler_X, scaler_y = normalize_data(X, y)
    print(f"Data normalized to [0, 1]. X shape: {X_norm.shape}, y shape: {y_norm.shape}")

    print("=" * 60)
    print("STEP 2: GREY RELATIONAL ANALYSIS")
    print("=" * 60)
    _notify_progress(progress_callback, 28, "Running Grey Relational Analysis (GRA)...")
    gra = GreyRelationalAnalysis(xi=0.5)
    gra.fit(X_norm, y_norm, feature_names)
    gra_ranking = gra.get_ranking()

    ranking_line = " -> ".join(
        [f"{feature} {score:.4f}" for feature, score in gra_ranking]
    )
    print(f"GRA ranking: {ranking_line}")

    print("=" * 60)
    print("STEP 3: TRAIN/TEST SPLIT (80/20)")
    print("=" * 60)
    _notify_progress(progress_callback, 40, "Splitting train/test data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm,
        y_norm,
        test_size=0.2,
        random_state=42,
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    print("=" * 60)
    print("STEP 4: MODEL COMPARISON")
    print("=" * 60)
    _notify_progress(progress_callback, 55, "Training 5 paper models and comparing performance...")
    results, trained_models = train_evaluate_models(X_train, X_test, y_train, y_test)

    print("=" * 60)
    print("STEP 5: SEQUENTIAL FEATURE ADDITION")
    print("=" * 60)
    _notify_progress(progress_callback, 72, "Running sequential feature addition...")
    incremental_results = incremental_feature_analysis(
        X_norm,
        y_norm,
        feature_names,
        gra_ranking,
    )

    best_model_name, best_metrics = _select_best_model(results)
    best_model = trained_models[best_model_name]
    top_features = [feature for feature, _score in gra_ranking[:3]]

    print("=" * 60)
    print("STEP 6: BEST MODEL SELECTION")
    print("=" * 60)
    _notify_progress(progress_callback, 84, "Selecting the best model from the paper benchmark...")
    print(
        f"Best model: {best_model_name} "
        f"(R2={best_metrics['R2_test']:.6f}, MSE={best_metrics['MSE_test']:.6f})"
    )

    print("=" * 60)
    print("STEP 7: SHAP ANALYSIS")
    print("=" * 60)
    _notify_progress(progress_callback, 92, "Computing SHAP values for the winning model...")
    shap_values, shap_importance, shap_input = perform_shap_analysis(
        best_model,
        X_train,
        feature_names,
    )

    _notify_progress(progress_callback, 97, "Saving trained model, scalers, and analysis artifacts...")
    artifact_paths = _save_artifacts(
        best_model_name=best_model_name,
        best_model=best_model,
        best_metrics=best_metrics,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        results=results,
        gra=gra,
        gra_ranking=gra_ranking,
        incremental_results=incremental_results,
        shap_values=shap_values,
        shap_input=shap_input,
        shap_importance=shap_importance,
        X_norm=X_norm,
        y_norm=y_norm,
        y_test=y_test,
        feature_names=feature_names,
        cleaned_df=df,
    )

    resolved_training_run_id = str(training_run_id or "").strip() or (
        f"artifacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    created_at = datetime.now().isoformat()

    summary = {
        "status": "success",
        "training_run_id": resolved_training_run_id,
        "created_at": created_at,
        "best_model": best_model_name,
        "r2": float(best_metrics["R2_test"]),
        "mse": float(best_metrics["MSE_test"]),
        "metrics": {
            "r2_score": float(best_metrics["R2_test"]),
            "mse": float(best_metrics["MSE_test"]),
            "mae": float(best_metrics["MAE_test"]),
            "rmse": float(best_metrics["RMSE_test"]),
        },
        "top_features": top_features,
        "message": "Training completed. Best model saved.",
        "gra_ranking": _ranking_to_serializable(gra_ranking),
        "artifacts": artifact_paths,
        "best_model_type": MODEL_NAME_TO_API_TYPE[best_model_name],
        "selected_sheet": resolved_sheet,
        "rows_after_preprocessing": int(len(df)),
    }

    with Path(artifact_paths["summary_path"]).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    build_chart_explanation_payload(
        ARTIFACT_DIR,
        training_run_id=resolved_training_run_id,
        created_at=created_at,
    )
    chart_payload_path = ARTIFACT_DIR / CHART_EXPLANATION_PAYLOAD_FILENAME
    artifact_paths["chart_explanation_payload_path"] = str(chart_payload_path)
    summary["artifacts"]["chart_explanation_payload_path"] = str(chart_payload_path)

    with Path(artifact_paths["summary_path"]).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    if return_runtime:
        summary["_runtime"] = {
            "best_model_object": best_model,
            "best_model_name": best_model_name,
            "best_model_type": MODEL_NAME_TO_API_TYPE[best_model_name],
            "model_results": results,
            "trained_models": trained_models,
            "feature_names": list(feature_names),
            "X_train_df": pd.DataFrame(X_train, columns=feature_names),
            "X_test_df": pd.DataFrame(X_test, columns=feature_names),
            "X_norm_df": pd.DataFrame(X_norm, columns=feature_names),
            "y_train_series": pd.Series(y_train, name=TARGET_COLUMN),
            "y_test_series": pd.Series(y_test, name=TARGET_COLUMN),
            "y_norm_series": pd.Series(y_norm, name=TARGET_COLUMN),
            "cleaned_df": df.copy(),
            "incremental_results_df": incremental_results.copy(),
            "shap_importance_df": shap_importance.copy(),
            "top_features": list(top_features),
            "gra_ranking": _ranking_to_serializable(gra_ranking),
            "selected_sheet": resolved_sheet,
            "rows_after_preprocessing": int(len(df)),
        }

    _notify_progress(progress_callback, 100, "Training completed. Best model saved.")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned hydrogen training pipeline."
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to a CSV/XLS/XLSX dataset.",
    )
    parser.add_argument(
        "--sheet-name",
        required=False,
        help="Specific Excel sheet to use when the workbook contains multiple sheets.",
    )
    args = parser.parse_args()

    result = train_and_save_model(args.file, sheet_name=args.sheet_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
