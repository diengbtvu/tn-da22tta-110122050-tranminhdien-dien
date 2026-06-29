import logging
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# -----------------------------------------------------------------------
# Column mapping from raw Excel file → standardised Python names
# -----------------------------------------------------------------------
EXCEL_COLUMN_RENAME = {
    'Unnamed: 0': 'DateTime',
    'Unnamed: 1': 'Day',
    'Unnamed: 2': 'HRT',
    'TS degradation': 'Sucrose_Degradation',
    'EtOH_Avg': 'Ethanol',
    'HAc_Avg': 'Acetate',
    'HPr_AVG': 'Propionate',
    'HBu_Total': 'Butyrate',
    'VFA_avg': 'VFA',
    'ORP_中': 'ORP_Mid',
    'ORP_下': 'ORP_Low',
}

# 11 input features (in preferred order)
FEATURE_COLUMNS = [
    'pH',
    'VSS',
    'Ethanol',
    'Acetate',
    'Propionate',
    'Butyrate',
    'Sucrose_Degradation',
    'ORP_Mid',
    'ORP_Low',
    'VFA',
    'COD-O',
]

TARGET_COLUMN = 'HPR'          # Hydrogen Production Rate (L/h/L)
EXCEL_SHEET_NAME = 'HRT 12 -4 h'


# -----------------------------------------------------------------------
# Abstract strategy
# -----------------------------------------------------------------------
class DataStrategy(ABC):
    """Abstract class defining strategy for handling data."""

    @abstractmethod
    def handle_data(
        self, data: pd.DataFrame
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        pass


# -----------------------------------------------------------------------
# Strategy 1: Preprocessing
# -----------------------------------------------------------------------
class DataPreprocessStrategy(DataStrategy):
    """
    Preprocess Hydrogen Production dataset:
    1. Rename columns from raw Excel names to Python-friendly names.
    2. Convert all columns to numeric.
    3. Fill missing values via linear interpolation (as per the paper).
    4. Drop remaining NaN rows.
    5. Keep only known feature columns + target.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            # 1. Rename columns if raw Excel names are present
            data = data.rename(columns=EXCEL_COLUMN_RENAME)

            # 2. Select only relevant columns that exist
            available_features = [c for c in FEATURE_COLUMNS if c in data.columns]
            cols_to_use = available_features + ([TARGET_COLUMN] if TARGET_COLUMN in data.columns else [])
            data = data[cols_to_use].copy()

            # 3. Convert to numeric
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

            # 4. Linear interpolation for missing values (paper Section 2.1)
            data = data.interpolate(method='linear', limit_direction='both')

            # 5. Drop any remaining NaN rows
            data = data.dropna()

            logging.info(
                "Data preprocessing completed. "
                f"Shape: {data.shape}, Features: {list(data.columns)}"
            )
            return data

        except Exception as e:
            logging.error(f"Error in DataPreprocessStrategy: {e}")
            raise e


# -----------------------------------------------------------------------
# Strategy 2: Train / test split
# -----------------------------------------------------------------------
class DataDivideStrategy(DataStrategy):
    """
    Split preprocessed DataFrame into X_train, X_test, y_train, y_test.
    Uses MinMaxScaler as described in Equation (1) of Wang et al. (2024).
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def handle_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        try:
            available_features = [c for c in FEATURE_COLUMNS if c in data.columns]

            X = data[available_features]
            y = data[TARGET_COLUMN]

            # Raw split first (scaler fitted on train only)
            X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # MinMax normalisation — fit on train, transform both
            X_train = pd.DataFrame(
                self.scaler_X.fit_transform(X_train_raw),
                columns=available_features,
                index=X_train_raw.index,
            )
            X_test = pd.DataFrame(
                self.scaler_X.transform(X_test_raw),
                columns=available_features,
                index=X_test_raw.index,
            )

            y_train = pd.Series(
                self.scaler_y.fit_transform(y_train_raw.values.reshape(-1, 1)).ravel(),
                index=y_train_raw.index,
                name=TARGET_COLUMN,
            )
            y_test = pd.Series(
                self.scaler_y.transform(y_test_raw.values.reshape(-1, 1)).ravel(),
                index=y_test_raw.index,
                name=TARGET_COLUMN,
            )

            logging.info(
                f"Data split complete. Train: {X_train.shape[0]} rows, "
                f"Test: {X_test.shape[0]} rows."
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in DataDivideStrategy: {e}")
            raise e


# -----------------------------------------------------------------------
# Context class
# -----------------------------------------------------------------------
class DataCleaning:
    """Context class — delegates work to the chosen DataStrategy."""

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(
        self,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error in DataCleaning: {e}")
            raise e