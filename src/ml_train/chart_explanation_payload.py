from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


CHART_EXPLANATION_PAYLOAD_FILENAME = "chart_explanation_payload.json"
TARGET_VARIABLE = "HPR"

MODEL_KEY_TO_DISPLAY = {
    "svm": "SVM",
    "dt": "DT",
    "rf": "RF",
    "knn": "KNN",
    "xgboost": "XGBoost",
}

MODEL_NAME_ALIASES = {
    "svm": "svm",
    "support_vector_machine": "svm",
    "dt": "dt",
    "decision_tree": "dt",
    "random_forest": "rf",
    "rf": "rf",
    "knn": "knn",
    "xgboost": "xgboost",
}

KNOWN_CHART_FILENAMES = [
    "fig_model_comparison_bars.png",
    "fig5_model_comparison.png",
    "fig_predicted_vs_actual.png",
    "fig_residuals.png",
    "fig_feature_importance.png",
    "fig_gra_ranking.png",
    "fig3a_gra_ranking.png",
    "fig3b_shap_analysis.png",
    "fig_correlation_heatmap.png",
    "fig6ab_mse_r2_features.png",
    "fig3_feature_analysis.png",
    "fig4_univariate_analysis.png",
    "fig6c_prediction_time.png",
    "fig_feature_distributions.png",
    "fig_feature_vs_target.png",
    "fig_boxplots.png",
    "fig_time_series.png",
]

DEFAULT_COVERAGE_POLICY = {
    "must_use_all_required_claims": True,
    "must_use_all_allowed_claims": False,
    "max_sentences": 5,
}


def build_chart_explanation_payload(
    output_dir: str | Path,
    training_run_id: str | None = None,
    created_at: str | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sources = _load_sources(output_path)
    resolved_training_run_id = (
        str(training_run_id or "").strip()
        or str(sources["bundle_manifest"].get("report_id") or "").strip()
        or str(sources["summary"].get("model_name") or "").strip()
        or str(sources["best_summary"].get("training_run_id") or "").strip()
        or output_path.name
    )
    resolved_created_at = (
        str(created_at or "").strip()
        or str(sources["summary"].get("created_at") or "").strip()
        or str(sources["bundle_manifest"].get("created_at") or "").strip()
        or datetime.now().isoformat()
    )

    payload = {
        "training_run_id": resolved_training_run_id,
        "created_at": resolved_created_at,
        "global_claims": _build_global_claims(sources),
        "charts": {},
    }

    chart_files = _discover_chart_files(output_path)
    for chart_file in chart_files:
        chart_id = _chart_id_from_filename(chart_file)
        payload["charts"][chart_id] = _build_chart_entry(chart_id, chart_file, sources)

    payload_path = output_path / CHART_EXPLANATION_PAYLOAD_FILENAME
    with payload_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    total_claims = sum(
        len(chart_entry.get("allowed_claims", {}))
        for chart_entry in payload["charts"].values()
    )
    print(
        "Generated "
        f"{CHART_EXPLANATION_PAYLOAD_FILENAME} with "
        f"{len(payload['charts'])} charts and {total_claims} allowed claims "
        f"at {payload_path}"
    )
    return payload


def validate_chart_explanation(
    chart_payload: Mapping[str, Any],
    llm_output: Mapping[str, Any],
) -> Dict[str, Any]:
    allowed_claims = set((chart_payload.get("allowed_claims") or {}).keys())
    explanation = llm_output.get("explanation") or {}
    sentences = explanation.get("sentences") or []
    errors: List[Dict[str, str]] = []

    for sentence in sentences:
        sentence_text = str(sentence.get("text") or "")
        used_claims = sentence.get("used_claims") or []
        if not isinstance(used_claims, list):
            used_claims = [used_claims]

        for claim in used_claims:
            claim_name = str(claim or "").strip()
            if claim_name and claim_name not in allowed_claims:
                errors.append(
                    {
                        "type": "claim_not_allowed_for_this_chart",
                        "claim": claim_name,
                        "sentence": sentence_text,
                    }
                )

    unsupported_claims = llm_output.get("unsupported_claims") or []
    needs_review = len(unsupported_claims) > 0
    return {
        "valid": not errors and not needs_review,
        "needs_review": needs_review,
        "errors": errors,
        "unsupported_claims": unsupported_claims,
    }


def _load_sources(output_path: Path) -> Dict[str, Any]:
    summary = _read_json(output_path / "summary.json") or {}
    best_summary = _read_json(output_path / "best_model_summary.json") or {}
    bundle_manifest = _read_json(output_path / "bundle_manifest.json") or {}

    model_comparison_rows = _read_csv_rows(output_path / "table_model_comparison.csv")
    if not model_comparison_rows and isinstance(summary.get("benchmark_models"), list):
        model_comparison_rows = list(summary["benchmark_models"])
    if not model_comparison_rows and best_summary.get("metrics"):
        metrics = best_summary.get("metrics") or {}
        best_model_name = best_summary.get("best_model") or summary.get("model_label") or "BestModel"
        model_comparison_rows = [
            {
                "model": best_model_name,
                "r2_score": metrics.get("r2_score"),
                "mse": metrics.get("mse"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
            }
        ]

    return {
        "summary": summary,
        "best_summary": best_summary,
        "bundle_manifest": bundle_manifest,
        "model_comparison": model_comparison_rows,
        "feature_importance": _read_csv_rows(output_path / "table_feature_importance.csv"),
        "shap_importance": _read_csv_rows(output_path / "best_model_shap_importance.csv"),
        "gra_ranking": _read_json(output_path / "gra_ranking.json")
        or best_summary.get("gra_ranking")
        or [],
        "correlation_matrix": _read_correlation_matrix(output_path / "table_correlation_matrix.csv"),
        "incremental_results": _read_csv_rows(output_path / "table1_incremental_results.csv"),
        "output_path": output_path,
    }


def _discover_chart_files(output_path: Path) -> List[str]:
    files = {path.name for path in output_path.glob("*.png")}
    ordered: List[str] = []

    for filename in KNOWN_CHART_FILENAMES:
        if filename in files:
            ordered.append(filename)

    for filename in sorted(files):
        if filename.startswith("model_") and filename.endswith("_scatter.png"):
            ordered.append(filename)

    for filename in sorted(files):
        if filename not in ordered:
            ordered.append(filename)

    return ordered


def _build_global_claims(sources: Mapping[str, Any]) -> Dict[str, str]:
    summary = sources["summary"]
    best_summary = sources["best_summary"]
    bundle_manifest = sources["bundle_manifest"]

    return _prune_empty_claims(
        {
            "dataset_selected_sheet": _stringify_claim(
                bundle_manifest.get("selected_sheet")
                or best_summary.get("selected_sheet")
                or summary.get("selected_sheet")
            ),
            "rows_after_preprocessing": _stringify_claim(
                bundle_manifest.get("rows_after_preprocessing")
                or best_summary.get("rows_after_preprocessing")
                or summary.get("rows_after_preprocessing")
            ),
            "target_variable": TARGET_VARIABLE,
            "best_model": _stringify_claim(
                bundle_manifest.get("best_model")
                or best_summary.get("best_model")
                or summary.get("model_label")
            ),
            "best_model_type": _stringify_claim(
                bundle_manifest.get("best_model_type")
                or best_summary.get("best_model_type")
                or summary.get("model_type")
            ),
        }
    )


def _build_chart_entry(chart_id: str, chart_file: str, sources: Mapping[str, Any]) -> Dict[str, Any]:
    chart_title = _chart_title_from_filename(chart_file)
    chart_type = _chart_type_from_filename(chart_file)
    allowed_claims = _build_allowed_claims(chart_file, sources)
    required_claim_keys = _build_required_claim_keys(chart_file, allowed_claims)
    return {
        "chart_id": chart_id,
        "chart_type": chart_type,
        "chart_title": chart_title,
        "file": chart_file,
        "allowed_claims": allowed_claims,
        "required_claim_keys": required_claim_keys,
        "coverage_policy": dict(DEFAULT_COVERAGE_POLICY),
    }


def _build_allowed_claims(chart_file: str, sources: Mapping[str, Any]) -> Dict[str, str]:
    filename = chart_file.lower()
    model_rows = _prepare_model_rows(sources["model_comparison"])
    feature_importance_rows = _prepare_ranked_rows(
        sources["feature_importance"],
        value_field="importance",
        sort_desc=True,
    )
    shap_rows = _prepare_ranked_rows(
        sources["shap_importance"],
        value_field="mean_abs_shap",
        sort_desc=True,
    )
    gra_rows = _prepare_gra_rows(sources["gra_ranking"])
    correlation_matrix = sources["correlation_matrix"]
    incremental_rows = _prepare_incremental_rows(sources["incremental_results"])

    if filename == "fig_model_comparison_bars.png":
        return _build_model_comparison_claims(model_rows)

    if filename == "fig5_model_comparison.png":
        return _build_paper_model_comparison_claims(model_rows)

    if filename == "fig_predicted_vs_actual.png":
        return _build_report_predicted_vs_actual_claims(model_rows)

    if filename == "fig_residuals.png":
        return _build_report_residual_claims(model_rows)

    if filename == "fig_feature_importance.png":
        return _build_feature_importance_claims(feature_importance_rows)

    if filename in {"fig_gra_ranking.png", "fig3a_gra_ranking.png"}:
        return _build_gra_claims(gra_rows)

    if filename == "fig3b_shap_analysis.png":
        return _build_shap_claims(shap_rows)

    if filename == "fig_correlation_heatmap.png":
        return _build_correlation_claims(correlation_matrix)

    if filename == "fig6ab_mse_r2_features.png":
        return _build_incremental_claims(incremental_rows)

    if filename == "fig3_feature_analysis.png":
        return _prune_empty_claims(
            {
                "combined_feature_analysis_definition": (
                    "This chart combines Grey Relational Analysis ranking and SHAP importance "
                    "for the selected training run."
                ),
                **_build_gra_claims(gra_rows),
                **_build_shap_claims(shap_rows),
            }
        )

    if filename in {"fig4_univariate_analysis.png", "fig_feature_vs_target.png"}:
        return _build_univariate_analysis_claims(sources)

    if filename == "fig6c_prediction_time.png":
        return _build_prediction_time_claims(sources)

    if filename == "fig_feature_distributions.png":
        return {
            "distribution_definition": "Histograms show how observed values are distributed across bins.",
            "distribution_caution": (
                "Do not claim normality, skewness, or multimodality unless those statistics are computed explicitly."
            ),
        }

    if filename == "fig_boxplots.png":
        return {
            "boxplot_definition": "Boxplots summarize median, quartiles, and spread for each variable.",
            "boxplot_caution": (
                "Do not claim outliers or distribution shape beyond what is directly supported by the chart."
            ),
        }

    if filename == "fig_time_series.png":
        return {
            "time_series_definition": "Time-series charts show how values change across sample order or time.",
            "time_series_caution": (
                "Do not infer seasonality, regime change, or causality unless the evidence is computed explicitly."
            ),
        }

    specific_scatter_match = re.fullmatch(r"model_([a-z0-9_]+)_scatter\.png", filename)
    if specific_scatter_match:
        return _build_individual_model_scatter_claims(
            model_rows,
            specific_model_key=_canonical_model_key(specific_scatter_match.group(1)),
        )

    specific_residual_match = re.fullmatch(r"model_([a-z0-9_]+)_residuals?\.png", filename)
    if specific_residual_match:
        return _build_individual_model_residual_claims(
            model_rows,
            specific_model_key=_canonical_model_key(specific_residual_match.group(1)),
        )

    return {
        "generic_chart_definition": f"{_chart_title_from_filename(chart_file)} is part of the training report.",
        "generic_chart_caution": (
            "Use only directly supported labels, annotations, and values when explaining this chart."
        ),
    }


def _build_model_comparison_claims(model_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {}
    valid_rows = [row for row in model_rows if row.get("r2_score") is not None]

    for row in valid_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))
        _set_claim(claims, f"{model_key}_mse", _format_decimal(row.get("mse"), 6))
        _set_claim(claims, f"{model_key}_rmse", _format_decimal(row.get("rmse"), 6))
        _set_claim(claims, f"{model_key}_mae", _format_decimal(row.get("mae"), 6))

    if valid_rows:
        ranked_rows = sorted(
            valid_rows,
            key=lambda row: (
                -(row.get("r2_score") if row.get("r2_score") is not None else float("-inf")),
                row.get("mse") if row.get("mse") is not None else float("inf"),
            ),
        )
        mse_ranked_rows = sorted(
            [row for row in valid_rows if row.get("mse") is not None],
            key=lambda row: row["mse"],
        )
        worst_rows = sorted(
            valid_rows,
            key=lambda row: row.get("r2_score") if row.get("r2_score") is not None else float("inf"),
        )

        _set_claim(claims, "best_model_by_r2", ranked_rows[0]["model_display"])
        _set_claim(claims, "best_r2_value", _format_decimal(ranked_rows[0].get("r2_score"), 4))
        _set_claim(claims, "worst_model_by_r2", worst_rows[0]["model_display"])
        _set_claim(claims, "worst_r2_value", _format_decimal(worst_rows[0].get("r2_score"), 4))
        _set_claim(
            claims,
            "model_r2_ranking",
            " > ".join(row["model_display"] for row in ranked_rows),
        )

        if mse_ranked_rows:
            _set_claim(claims, "best_model_by_mse", mse_ranked_rows[0]["model_display"])
            _set_claim(claims, "best_mse_value", _format_decimal(mse_ranked_rows[0].get("mse"), 6))

    claims["metric_context"] = (
        "Regression model comparison uses higher R2 and lower MSE/RMSE/MAE as better performance."
    )
    return _prune_empty_claims(claims)


def _build_predicted_vs_actual_claims(
    model_rows: List[Dict[str, Any]],
    specific_model_key: str | None = None,
) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "r2_definition": "R2 measures how much variation in the target is explained by the model; higher is better.",
        "mse_definition": "MSE is the average squared prediction error; lower is better.",
        "scatter_reference_line_meaning": (
            "Points closer to the diagonal reference line indicate predictions closer to actual values."
        ),
    }

    target_rows = model_rows
    if specific_model_key:
        target_rows = [row for row in model_rows if row["model_key"] == specific_model_key]
        if target_rows:
            claims["model_name"] = target_rows[0]["model_display"]
    elif model_rows:
        claims["predicted_vs_actual_models_included"] = " > ".join(
            row["model_display"]
            for row in sorted(
                model_rows,
                key=lambda row: (
                    -(row.get("r2_score") if row.get("r2_score") is not None else float("-inf")),
                    row.get("mse") if row.get("mse") is not None else float("inf"),
                ),
            )
        )

    for row in target_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))
        _set_claim(claims, f"{model_key}_mse", _format_decimal(row.get("mse"), 6))
        _set_claim(claims, f"{model_key}_rmse", _format_decimal(row.get("rmse"), 6))
        _set_claim(claims, f"{model_key}_mae", _format_decimal(row.get("mae"), 6))

    return _prune_empty_claims(claims)


def _build_paper_model_comparison_claims(model_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "scatter_point_meaning": (
            "Each scatter point represents one actual-versus-predicted test sample for the model shown in that panel."
        ),
        "best_fit_line_meaning": (
            "The red line in each scatter panel is the fitted line through the plotted actual-versus-predicted samples."
        ),
        "r2_bar_chart_context": "The bar panel compares test-set R2 values across models; higher R2 is better.",
    }

    valid_rows = [row for row in model_rows if row.get("r2_score") is not None]
    for row in valid_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))

    if valid_rows:
        ranked_rows = sorted(valid_rows, key=lambda row: row["r2_score"], reverse=True)
        worst_rows = sorted(valid_rows, key=lambda row: row["r2_score"])

        _set_claim(claims, "best_model_by_r2", ranked_rows[0]["model_display"])
        _set_claim(claims, "best_r2_value", _format_decimal(ranked_rows[0].get("r2_score"), 4))
        _set_claim(claims, "worst_model_by_r2", worst_rows[0]["model_display"])
        _set_claim(claims, "worst_r2_value", _format_decimal(worst_rows[0].get("r2_score"), 4))
        _set_claim(
            claims,
            "model_r2_ranking",
            " > ".join(row["model_display"] for row in ranked_rows),
        )

    return _prune_empty_claims(claims)


def _build_report_predicted_vs_actual_claims(model_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "scatter_point_meaning": "Each point represents one actual-versus-predicted sample.",
        "diagonal_reference_line_meaning": (
            "The dashed diagonal reference line marks perfect agreement between actual and predicted values."
        ),
        "best_fit_line_meaning": (
            "The fitted line summarizes the trend of the plotted actual-versus-predicted samples in each panel."
        ),
        "actual_axis_label": "Actual HPR (scaled)",
        "predicted_axis_label": "Predicted HPR (scaled)",
        "r2_definition": "R2 measures how much variation in the target is explained by the model; higher is better.",
    }

    if model_rows:
        claims["predicted_vs_actual_models_included"] = " > ".join(
            row["model_display"]
            for row in sorted(
                [row for row in model_rows if row.get("r2_score") is not None],
                key=lambda row: row["r2_score"],
                reverse=True,
            )
        )

    for row in model_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))

    return _prune_empty_claims(claims)


def _build_individual_model_scatter_claims(
    model_rows: List[Dict[str, Any]],
    specific_model_key: str | None = None,
) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "scatter_point_meaning": "Each point represents one actual-versus-predicted sample for this model.",
        "best_fit_line_meaning": (
            "The red line is the fitted line through the plotted actual-versus-predicted samples."
        ),
        "actual_axis_label": "Measured",
        "predicted_axis_label": "Predicted",
        "r2_definition": "R2 measures how much variation in the target is explained by the model; higher is better.",
    }

    target_rows = model_rows
    if specific_model_key:
        target_rows = [row for row in model_rows if row["model_key"] == specific_model_key]
        if target_rows:
            claims["model_name"] = target_rows[0]["model_display"]

    for row in target_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))

    return _prune_empty_claims(claims)


def _build_residual_claims(
    model_rows: List[Dict[str, Any]],
    specific_model_key: str | None = None,
) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "residual_definition": "Residuals are prediction errors computed as actual value minus predicted value.",
        "residual_zero_line_meaning": "Residuals closer to zero indicate smaller prediction errors.",
        "unsupported_pattern_warning": (
            "Do not claim bias, randomness, normality, or heteroscedasticity unless residual statistics "
            "are computed explicitly."
        ),
    }

    target_rows = model_rows
    if specific_model_key:
        target_rows = [row for row in model_rows if row["model_key"] == specific_model_key]
        if target_rows:
            claims["model_name"] = target_rows[0]["model_display"]
    elif model_rows:
        claims["residual_models_included"] = " > ".join(
            row["model_display"]
            for row in sorted(
                model_rows,
                key=lambda row: (
                    -(row.get("r2_score") if row.get("r2_score") is not None else float("-inf")),
                    row.get("mse") if row.get("mse") is not None else float("inf"),
                ),
            )
        )

    for row in target_rows:
        model_key = row["model_key"]
        _set_claim(claims, f"{model_key}_r2", _format_decimal(row.get("r2_score"), 4))
        _set_claim(claims, f"{model_key}_mse", _format_decimal(row.get("mse"), 6))

    return _prune_empty_claims(claims)


def _build_report_residual_claims(model_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "residual_definition": "Residuals are prediction errors computed as actual value minus predicted value.",
        "residual_zero_line_meaning": "The horizontal zero line marks predictions with no residual error.",
        "residual_scatter_point_meaning": "Each point shows one sample's residual at its predicted value.",
        "predicted_axis_label": "Predicted HPR (scaled)",
        "residual_axis_label": "Residual (actual - predicted)",
        "unsupported_pattern_warning": (
            "Do not claim bias, randomness, normality, or heteroscedasticity unless residual statistics "
            "are computed explicitly."
        ),
    }

    ranked_rows = [
        row for row in sorted(
            model_rows,
            key=lambda row: row.get("r2_score") if row.get("r2_score") is not None else float("-inf"),
            reverse=True,
        )
        if row.get("r2_score") is not None
    ]
    if ranked_rows:
        claims["residual_models_included"] = " > ".join(row["model_display"] for row in ranked_rows)

    return _prune_empty_claims(claims)


def _build_individual_model_residual_claims(
    model_rows: List[Dict[str, Any]],
    specific_model_key: str | None = None,
) -> Dict[str, str]:
    claims = _build_report_residual_claims([])
    if specific_model_key:
        target_rows = [row for row in model_rows if row["model_key"] == specific_model_key]
        if target_rows:
            claims["model_name"] = target_rows[0]["model_display"]
    return _prune_empty_claims(claims)


def _build_feature_importance_claims(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "feature_importance_caution": "Feature importance shows model reliance or contribution, not causal effect."
    }
    if not rows:
        return claims

    for row in rows:
        feature_key = _snake_case(str(row["feature"]))
        _set_claim(claims, f"{feature_key}_feature_importance", _format_decimal(row.get("value"), 4))

    _set_rank_summary_claims(claims, rows, "feature_importance", "feature")
    return _prune_empty_claims(claims)


def _build_gra_claims(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "gra_definition": (
            "Grey Relational Analysis ranks features by relational grade; higher score means stronger "
            "association in this analysis."
        )
    }
    if not rows:
        return claims

    for row in rows:
        feature_key = _snake_case(str(row["feature"]))
        _set_claim(claims, f"{feature_key}_gra_rank", _stringify_claim(row.get("rank")))
        _set_claim(claims, f"{feature_key}_gra_score", _format_decimal(row.get("value"), 4))

    _set_gra_rank_summary_claims(claims, rows)
    return _prune_empty_claims(claims)


def _build_shap_claims(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "shap_definition": "Mean absolute SHAP summarizes average feature impact magnitude on model output.",
        "shap_caution": "SHAP importance does not prove that a feature causes the target.",
    }
    if not rows:
        return claims

    for row in rows:
        feature_key = _snake_case(str(row["feature"]))
        _set_claim(claims, f"{feature_key}_mean_abs_shap", _format_decimal(row.get("value"), 4))

    _set_rank_summary_claims(claims, rows, "shap", "feature")
    return _prune_empty_claims(claims)


def _build_correlation_claims(correlation_matrix: Mapping[str, Mapping[str, float | None]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "correlation_definition": "Correlation ranges from -1 to 1 and measures linear association between two variables.",
        "correlation_caution": "Correlation does not prove causation.",
    }
    if not correlation_matrix:
        return claims

    target_correlations = _extract_target_correlations(correlation_matrix, TARGET_VARIABLE)
    if not target_correlations:
        return claims

    for feature_name, value in target_correlations.items():
        _set_claim(
            claims,
            f"{_snake_case(feature_name)}_correlation_with_{_snake_case(TARGET_VARIABLE)}",
            _format_decimal(value, 4),
        )

    positive_pairs = [(feature, value) for feature, value in target_correlations.items() if value is not None and value > 0]
    negative_pairs = [(feature, value) for feature, value in target_correlations.items() if value is not None and value < 0]

    if positive_pairs:
        feature_name, value = max(positive_pairs, key=lambda item: item[1])
        _set_claim(
            claims,
            f"strongest_positive_correlation_with_{_snake_case(TARGET_VARIABLE)}_feature",
            feature_name,
        )
        _set_claim(
            claims,
            f"strongest_positive_correlation_with_{_snake_case(TARGET_VARIABLE)}_value",
            _format_decimal(value, 4),
        )

    if negative_pairs:
        feature_name, value = min(negative_pairs, key=lambda item: item[1])
        _set_claim(
            claims,
            f"strongest_negative_correlation_with_{_snake_case(TARGET_VARIABLE)}_feature",
            feature_name,
        )
        _set_claim(
            claims,
            f"strongest_negative_correlation_with_{_snake_case(TARGET_VARIABLE)}_value",
            _format_decimal(value, 4),
        )

    return _prune_empty_claims(claims)


def _build_incremental_claims(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    claims: Dict[str, str] = {
        "incremental_analysis_definition": (
            "Incremental feature analysis evaluates performance after adding ranked features step by step."
        )
    }
    if not rows:
        return claims

    best_candidate: Dict[str, Any] | None = None
    steps_by_model: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        n_features = row.get("n_features")
        feature_subset = row.get("feature_subset")
        for model_key, metrics in (row.get("models") or {}).items():
            r2_value = metrics.get("r2")
            mse_value = metrics.get("mse")
            _set_claim(
                claims,
                f"{model_key}_r2_at_{n_features}_features",
                _format_decimal(r2_value, 4),
            )
            _set_claim(
                claims,
                f"{model_key}_mse_at_{n_features}_features",
                _format_decimal(mse_value, 6),
            )

            steps_by_model.setdefault(model_key, []).append(
                {
                    "n_features": n_features,
                    "r2": r2_value,
                    "mse": mse_value,
                }
            )

            candidate = {
                "model_key": model_key,
                "model_display": MODEL_KEY_TO_DISPLAY.get(model_key, model_key.upper()),
                "n_features": n_features,
                "r2": r2_value,
                "mse": mse_value,
                "feature_subset": feature_subset,
            }
            if _is_better_incremental_candidate(candidate, best_candidate):
                best_candidate = candidate

    for model_key, steps in steps_by_model.items():
        ordered_steps = sorted(steps, key=lambda item: item["n_features"])
        for previous, current in zip(ordered_steps, ordered_steps[1:]):
            if previous.get("r2") is None or current.get("r2") is None:
                continue
            if current["r2"] > previous["r2"]:
                claims[
                    f"{model_key}_r2_increases_from_{previous['n_features']}_to_{current['n_features']}_features"
                ] = "true"
            elif current["r2"] < previous["r2"]:
                claims[
                    f"{model_key}_r2_decreases_from_{previous['n_features']}_to_{current['n_features']}_features"
                ] = "true"

    if best_candidate:
        _set_claim(claims, "best_incremental_model", best_candidate["model_display"])
        _set_claim(claims, "best_incremental_n_features", _stringify_claim(best_candidate["n_features"]))
        _set_claim(claims, "best_incremental_r2", _format_decimal(best_candidate.get("r2"), 4))
        _set_claim(claims, "best_incremental_mse", _format_decimal(best_candidate.get("mse"), 6))
        _set_claim(
            claims,
            "best_incremental_feature_subset",
            _stringify_claim(best_candidate.get("feature_subset")),
        )

    return _prune_empty_claims(claims)


def _build_prediction_time_claims(sources: Mapping[str, Any]) -> Dict[str, str]:
    best_summary = sources["best_summary"]
    best_model = best_summary.get("best_model")
    claims: Dict[str, str] = {
        "prediction_time_definition": "This chart compares experimental target values and model predictions across sample order.",
        "experimental_series_label": "Experimental",
        "predicted_series_label": "Predicted",
        "x_axis_label": "Time (day)",
        "y_axis_label": "Hydrogen Production Rate (L/h/L)",
        "prediction_time_caution": (
            "Do not claim temporal causality or forecast stability unless the evaluation protocol supports it."
        ),
    }

    if best_model:
        _set_claim(claims, "model_name", best_model)
    return _prune_empty_claims(claims)


def _build_univariate_analysis_claims(sources: Mapping[str, Any]) -> Dict[str, str]:
    summary = sources["summary"]
    claims: Dict[str, str] = {
        "univariate_analysis_definition": (
            "This chart shows feature-versus-target relationships for individual variables."
        ),
        "scatter_point_meaning": (
            "Each point represents one observed sample for the feature shown in that subplot."
        ),
        "trend_line_meaning": (
            "The dashed line in each subplot is a simple fitted trend line for the plotted samples."
        ),
        "trend_line_caution": (
            "Visual trend lines suggest association in the plotted sample and do not prove causation."
        ),
        "target_axis_label": "HPR (L/h/L)",
        "subplot_title_definition": (
            "Each subplot letter labels a different feature-versus-target scatter plot."
        ),
    }

    feature_names = _extract_feature_names(summary)
    plotted_features = feature_names[:8]
    _set_claim(claims, "plotted_feature_count", _stringify_claim(len(plotted_features)))

    for index, feature_name in enumerate(plotted_features, start=1):
        _set_claim(claims, f"plotted_feature_{index}", feature_name)

    if plotted_features:
        _set_claim(claims, "first_plotted_feature", plotted_features[0])
        _set_claim(claims, "last_plotted_feature", plotted_features[-1])

    return _prune_empty_claims(claims)


def _build_required_claim_keys(
    chart_file: str,
    allowed_claims: Mapping[str, str],
) -> List[str]:
    filename = chart_file.lower()
    allowed_keys = list(allowed_claims.keys())

    if filename == "fig_model_comparison_bars.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "best_model_by_r2",
                "best_r2_value",
                "best_model_by_mse",
                "best_mse_value",
                "worst_model_by_r2",
                "worst_r2_value",
                "model_r2_ranking",
                "metric_context",
            ],
        )

    if filename == "fig5_model_comparison.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "best_model_by_r2",
                "best_r2_value",
                "worst_model_by_r2",
                "worst_r2_value",
                "model_r2_ranking",
                "scatter_point_meaning",
                "best_fit_line_meaning",
                "r2_bar_chart_context",
            ],
        )

    if filename == "fig_predicted_vs_actual.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "predicted_vs_actual_models_included",
                "scatter_point_meaning",
                "diagonal_reference_line_meaning",
                "best_fit_line_meaning",
                "actual_axis_label",
                "predicted_axis_label",
                "r2_definition",
            ],
        )

    if filename == "fig_residuals.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "residual_models_included",
                "residual_definition",
                "residual_zero_line_meaning",
                "residual_scatter_point_meaning",
                "predicted_axis_label",
                "residual_axis_label",
                "unsupported_pattern_warning",
            ],
        )

    if filename == "fig_feature_importance.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "top_feature_importance_feature",
                "top_feature_importance_value",
                "second_feature_importance_feature",
                "second_feature_importance_value",
                "lowest_feature_importance_feature",
                "lowest_feature_importance_value",
                "feature_importance_caution",
            ],
        )

    if filename in {"fig_gra_ranking.png", "fig3a_gra_ranking.png"}:
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "top_gra_feature",
                "top_gra_score",
                "second_gra_feature",
                "second_gra_score",
                "lowest_gra_feature",
                "lowest_gra_score",
                "gra_definition",
            ],
        )

    if filename == "fig3b_shap_analysis.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "top_shap_feature",
                "top_shap_value",
                "second_shap_feature",
                "second_shap_value",
                "lowest_shap_feature",
                "lowest_shap_value",
                "shap_definition",
                "shap_caution",
            ],
        )

    if filename == "fig_correlation_heatmap.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                f"strongest_positive_correlation_with_{_snake_case(TARGET_VARIABLE)}_feature",
                f"strongest_positive_correlation_with_{_snake_case(TARGET_VARIABLE)}_value",
                f"strongest_negative_correlation_with_{_snake_case(TARGET_VARIABLE)}_feature",
                f"strongest_negative_correlation_with_{_snake_case(TARGET_VARIABLE)}_value",
                "correlation_definition",
                "correlation_caution",
            ],
        )

    if filename == "fig6ab_mse_r2_features.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "best_incremental_model",
                "best_incremental_n_features",
                "best_incremental_r2",
                "best_incremental_mse",
                "best_incremental_feature_subset",
                "incremental_analysis_definition",
            ],
        )

    if filename == "fig3_feature_analysis.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "combined_feature_analysis_definition",
                "top_gra_feature",
                "top_gra_score",
                "top_shap_feature",
                "top_shap_value",
                "shap_caution",
            ],
        )

    if filename in {"fig4_univariate_analysis.png", "fig_feature_vs_target.png"}:
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "univariate_analysis_definition",
                "plotted_feature_count",
                "plotted_feature_1",
                "target_axis_label",
                "trend_line_caution",
            ],
        )

    if filename == "fig6c_prediction_time.png":
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "model_name",
                "prediction_time_definition",
                "experimental_series_label",
                "predicted_series_label",
                "x_axis_label",
                "y_axis_label",
                "prediction_time_caution",
            ],
        )

    if filename == "fig_feature_distributions.png":
        return _filter_required_claim_keys(
            allowed_claims,
            ["distribution_definition", "distribution_caution"],
        )

    if filename == "fig_boxplots.png":
        return _filter_required_claim_keys(
            allowed_claims,
            ["boxplot_definition", "boxplot_caution"],
        )

    if filename == "fig_time_series.png":
        return _filter_required_claim_keys(
            allowed_claims,
            ["time_series_definition", "time_series_caution"],
        )

    specific_scatter_match = re.fullmatch(r"model_([a-z0-9_]+)_scatter\.png", filename)
    if specific_scatter_match:
        model_key = _canonical_model_key(specific_scatter_match.group(1))
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "model_name",
                f"{model_key}_r2",
                "scatter_point_meaning",
                "best_fit_line_meaning",
                "actual_axis_label",
                "predicted_axis_label",
                "r2_definition",
            ],
        )

    specific_residual_match = re.fullmatch(r"model_([a-z0-9_]+)_residuals?\.png", filename)
    if specific_residual_match:
        model_key = _canonical_model_key(specific_residual_match.group(1))
        return _filter_required_claim_keys(
            allowed_claims,
            [
                "model_name",
                "residual_definition",
                "residual_zero_line_meaning",
                "residual_scatter_point_meaning",
                "predicted_axis_label",
                "residual_axis_label",
                "unsupported_pattern_warning",
            ],
        )

    return allowed_keys[: min(3, len(allowed_keys))]


def _prepare_model_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for raw_row in rows:
        model_key = _canonical_model_key(raw_row.get("model"))
        prepared_rows.append(
            {
                "model_key": model_key,
                "model_display": MODEL_KEY_TO_DISPLAY.get(model_key, str(raw_row.get("model") or model_key).upper()),
                "r2_score": _to_float(raw_row.get("r2_score")),
                "mse": _to_float(raw_row.get("mse")),
                "rmse": _to_float(raw_row.get("rmse")),
                "mae": _to_float(raw_row.get("mae")),
            }
        )
    return prepared_rows


def _prepare_ranked_rows(
    rows: Iterable[Mapping[str, Any]],
    value_field: str,
    sort_desc: bool,
) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(rows, start=1):
        value = _to_float(raw_row.get(value_field))
        if value is None:
            continue
        prepared_rows.append(
            {
                "rank": index,
                "feature": str(raw_row.get("feature") or "").strip(),
                "value": value,
            }
        )

    prepared_rows.sort(key=lambda item: item["value"], reverse=sort_desc)
    for index, row in enumerate(prepared_rows, start=1):
        row["rank"] = index
    return prepared_rows


def _prepare_gra_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(rows, start=1):
        value = _to_float(raw_row.get("score"))
        feature = str(raw_row.get("feature") or "").strip()
        if not feature or value is None:
            continue
        prepared_rows.append(
            {
                "rank": _to_int(raw_row.get("rank")) or index,
                "feature": feature,
                "value": value,
            }
        )

    prepared_rows.sort(key=lambda item: item["rank"])
    return prepared_rows


def _prepare_incremental_rows(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    prepared_rows: List[Dict[str, Any]] = []
    for raw_row in rows:
        n_features = _to_int(raw_row.get("n_features"))
        if n_features is None:
            continue
        row = {
            "n_features": n_features,
            "feature_subset": str(raw_row.get("feature_subset") or "").strip(),
            "models": {},
        }
        for column, value in raw_row.items():
            match = re.fullmatch(r"([A-Za-z0-9_]+)_(R2|MSE)", str(column))
            if not match:
                continue
            model_key = _canonical_model_key(match.group(1))
            metric_key = "r2" if match.group(2) == "R2" else "mse"
            row["models"].setdefault(model_key, {})[metric_key] = _to_float(value)
        prepared_rows.append(row)

    prepared_rows.sort(key=lambda item: item["n_features"])
    return prepared_rows


def _extract_target_correlations(
    correlation_matrix: Mapping[str, Mapping[str, float | None]],
    target_variable: str,
) -> Dict[str, float]:
    if target_variable in correlation_matrix:
        row = correlation_matrix[target_variable]
        return {
            feature: value
            for feature, value in row.items()
            if feature != target_variable and value is not None
        }

    extracted: Dict[str, float] = {}
    for feature_name, feature_row in correlation_matrix.items():
        if feature_name == target_variable:
            continue
        value = feature_row.get(target_variable)
        if value is not None:
            extracted[feature_name] = value
    return extracted


def _extract_feature_names(summary: Mapping[str, Any]) -> List[str]:
    feature_names = summary.get("feature_names") or summary.get("selected_features") or []
    if not isinstance(feature_names, list):
        return []

    normalized: List[str] = []
    for item in feature_names:
        feature_name = str(item or "").strip()
        if feature_name:
            normalized.append(feature_name)
    return normalized


def _set_rank_summary_claims(
    claims: Dict[str, str],
    rows: List[Dict[str, Any]],
    prefix: str,
    label_field: str,
) -> None:
    if not rows:
        return

    def _set_named_rank(rank_name: str, index: int) -> None:
        if index >= len(rows):
            return
        label = rows[index][label_field]
        value = rows[index]["value"]
        _set_claim(claims, f"{rank_name}_{prefix}_{label_field}", label)
        _set_claim(claims, f"{rank_name}_{prefix}_value", _format_decimal(value, 4))

    _set_named_rank("top", 0)
    _set_named_rank("second", 1)
    _set_named_rank("third", 2)
    _set_named_rank("lowest", len(rows) - 1)


def _set_gra_rank_summary_claims(claims: Dict[str, str], rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    def _set_named_rank(rank_name: str, index: int) -> None:
        if index >= len(rows):
            return
        feature_name = rows[index]["feature"]
        value = rows[index]["value"]
        _set_claim(claims, f"{rank_name}_gra_feature", feature_name)
        _set_claim(claims, f"{rank_name}_gra_score", _format_decimal(value, 4))

    _set_named_rank("top", 0)
    _set_named_rank("second", 1)
    _set_named_rank("third", 2)
    _set_named_rank("lowest", len(rows) - 1)


def _is_better_incremental_candidate(
    candidate: Mapping[str, Any],
    current_best: Mapping[str, Any] | None,
) -> bool:
    if candidate.get("r2") is None:
        return False
    if current_best is None or current_best.get("r2") is None:
        return True
    if candidate["r2"] > current_best["r2"]:
        return True
    if candidate["r2"] < current_best["r2"]:
        return False

    candidate_mse = candidate.get("mse")
    current_best_mse = current_best.get("mse")
    if candidate_mse is None:
        return False
    if current_best_mse is None:
        return True
    return candidate_mse < current_best_mse


def _chart_id_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    if stem.startswith("fig_"):
        stem = stem[4:]
    return f"chart_{_snake_case(stem)}"


def _chart_title_from_filename(filename: str) -> str:
    title_overrides = {
        "fig_model_comparison_bars.png": "Model Comparison Bars",
        "fig5_model_comparison.png": "Model Comparison",
        "fig_predicted_vs_actual.png": "Predicted vs Actual",
        "fig_residuals.png": "Residual Analysis",
        "fig_feature_importance.png": "Feature Importance",
        "fig_gra_ranking.png": "GRA Feature Ranking",
        "fig3a_gra_ranking.png": "GRA Feature Ranking",
        "fig3b_shap_analysis.png": "SHAP Analysis",
        "fig_correlation_heatmap.png": "Correlation Heatmap",
        "fig6ab_mse_r2_features.png": "Incremental Feature Analysis",
        "fig3_feature_analysis.png": "Combined Feature Analysis",
        "fig4_univariate_analysis.png": "Univariate Feature Analysis",
        "fig6c_prediction_time.png": "Prediction Over Samples",
        "fig_feature_distributions.png": "Feature Distributions",
        "fig_feature_vs_target.png": "Feature vs Target Relationships",
        "fig_boxplots.png": "Feature Boxplots",
        "fig_time_series.png": "Time Series Overview",
    }
    if filename in title_overrides:
        return title_overrides[filename]
    stem = Path(filename).stem
    if stem.startswith("model_") and stem.endswith("_scatter"):
        model_key = _canonical_model_key(stem[len("model_") : -len("_scatter")])
        return f"{MODEL_KEY_TO_DISPLAY.get(model_key, model_key.upper())} Predicted vs Actual"
    return " ".join(part.capitalize() for part in _snake_case(stem).split("_"))


def _chart_type_from_filename(filename: str) -> str:
    stem = Path(filename).stem.lower()
    if "heatmap" in stem:
        return "heatmap"
    if "scatter" in stem or "predicted_vs_actual" in stem:
        return "scatter_plot"
    if "residual" in stem:
        return "residual_plot"
    if "boxplot" in stem:
        return "boxplot"
    if "distribution" in stem:
        return "histogram_grid"
    if "time_series" in stem or "prediction_time" in stem or "mse_r2_features" in stem:
        return "line_chart"
    if "comparison" in stem or "ranking" in stem or "importance" in stem:
        return "bar_chart"
    return "chart"


def _canonical_model_key(value: Any) -> str:
    normalized = _snake_case(value or "")
    return MODEL_NAME_ALIASES.get(normalized, normalized or "model")


def _find_focus_metric_keys(allowed_claims: Mapping[str, str]) -> List[str]:
    model_label = _extract_focus_model_label(allowed_claims)
    if model_label:
        model_key = _canonical_model_key(model_label)
        preferred_keys = [
            f"{model_key}_r2",
            f"{model_key}_mse",
            f"{model_key}_rmse",
            f"{model_key}_mae",
        ]
        selected = [key for key in preferred_keys if key in allowed_claims]
        if selected:
            return selected

    metric_keys = [
        key
        for key in allowed_claims
        if key.endswith(("_r2", "_mse", "_rmse", "_mae"))
    ]
    return metric_keys[:4]


def _extract_focus_model_label(allowed_claims: Mapping[str, str]) -> str | None:
    if allowed_claims.get("model_name"):
        return str(allowed_claims["model_name"])

    for ranking_key in ("predicted_vs_actual_models_included", "residual_models_included", "model_r2_ranking"):
        ranking_value = allowed_claims.get(ranking_key)
        if not ranking_value:
            continue
        first_model = str(ranking_value).split(" > ")[0].strip()
        if first_model:
            return first_model

    return None


def _filter_required_claim_keys(
    allowed_claims: Mapping[str, str],
    preferred_keys: Iterable[str],
) -> List[str]:
    selected_keys: List[str] = []
    seen_keys = set()

    for key in preferred_keys:
        if key in allowed_claims and key not in seen_keys:
            selected_keys.append(key)
            seen_keys.add(key)

    if selected_keys:
        return selected_keys

    fallback_keys = list(allowed_claims.keys())[: min(3, len(allowed_claims))]
    return fallback_keys


def _snake_case(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "")).strip("_")
    return text.lower() or "value"


def _to_float(value: Any) -> float | None:
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _format_decimal(value: float | None, precision: int) -> str | None:
    if value is None:
        return None
    return f"{value:.{precision}f}"


def _stringify_claim(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip()
    return text or None


def _set_claim(claims: Dict[str, str], key: str, value: str | None) -> None:
    if value is None:
        return
    claims[key] = value


def _prune_empty_claims(claims: Mapping[str, Any]) -> Dict[str, str]:
    return {
        str(key): str(value)
        for key, value in claims.items()
        if value not in (None, "")
    }


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _read_correlation_matrix(path: Path) -> Dict[str, Dict[str, float | None]]:
    if not path.exists():
        return {}
    matrix: Dict[str, Dict[str, float | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if not rows:
        return matrix

    headers = rows[0][1:]
    for row in rows[1:]:
        if not row:
            continue
        row_name = str(row[0]).strip()
        if not row_name:
            continue
        matrix[row_name] = {}
        for header, value in zip(headers, row[1:]):
            matrix[row_name][str(header).strip()] = _to_float(value)
    return matrix
