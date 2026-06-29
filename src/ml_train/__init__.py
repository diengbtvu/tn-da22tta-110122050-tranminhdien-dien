"""Training utilities aligned with the original Hydrogen-Prediction repo."""


def train_and_save_model(*args, **kwargs):
    from .train_pipeline import train_and_save_model as _train_and_save_model

    return _train_and_save_model(*args, **kwargs)


def build_chart_explanation_payload(*args, **kwargs):
    from .chart_explanation_payload import (
        build_chart_explanation_payload as _build_chart_explanation_payload,
    )

    return _build_chart_explanation_payload(*args, **kwargs)


def validate_chart_explanation(*args, **kwargs):
    from .chart_explanation_payload import (
        validate_chart_explanation as _validate_chart_explanation,
    )

    return _validate_chart_explanation(*args, **kwargs)


__all__ = [
    "train_and_save_model",
    "build_chart_explanation_payload",
    "validate_chart_explanation",
]
