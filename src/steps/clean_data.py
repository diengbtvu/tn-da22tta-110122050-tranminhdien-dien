import logging
from typing import Tuple

import pandas as pd
from typing_extensions import Annotated
from zenml import step

from src.data_cleaning import (
    DataCleaning,
    DataDivideStrategy,
    DataPreprocessStrategy,
)


@step
def clean_df(
    data: pd.DataFrame,
) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Preprocess and split the Hydrogen dataset.

    Steps:
    1. DataPreprocessStrategy: rename columns, convert types, interpolate, drop NaN.
    2. DataDivideStrategy: MinMax normalise + 80/20 train-test split.
    """
    try:
        # Step 1 — Preprocess
        preprocess_strategy = DataPreprocessStrategy()
        cleaner = DataCleaning(data, preprocess_strategy)
        preprocessed = cleaner.handle_data()

        # Step 2 — Split
        divide_strategy = DataDivideStrategy()
        cleaner2 = DataCleaning(preprocessed, divide_strategy)
        X_train, X_test, y_train, y_test = cleaner2.handle_data()

        logging.info("clean_df step completed successfully.")
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in clean_df step: {e}")
        raise
