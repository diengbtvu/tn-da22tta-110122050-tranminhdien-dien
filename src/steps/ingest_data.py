import logging

import pandas as pd
from zenml import step

from src.data_cleaning import EXCEL_SHEET_NAME


class IngestData:
    """Load dataset from CSV or XLSX file into a Pandas DataFrame."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self) -> pd.DataFrame:
        logging.info(f"Ingesting data from: {self.data_path}")
        lower_path = self.data_path.lower()
        if lower_path.endswith('.xlsx'):
            df = pd.read_excel(
                self.data_path,
                sheet_name=EXCEL_SHEET_NAME,
                engine='openpyxl',
            )
        elif lower_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(
                "Unsupported dataset format. Only .csv and .xlsx are supported."
            )
        logging.info(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
        return df


@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    ZenML step: Load dataset from CSV or XLSX file.

    Supports:
    - .csv  — plain CSV
    - .xlsx — modern Excel (requires openpyxl)
    """
    try:
        ingestor = IngestData(data_path)
        return ingestor.get_data()
    except Exception as e:
        logging.error(f"Error in ingest_df: {e}")
        raise
