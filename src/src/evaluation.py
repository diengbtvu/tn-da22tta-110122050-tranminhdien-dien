import logging

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class R2Score:
    """R-squared (coefficient of determination)."""

    def calculate_score(self, y_true, y_pred) -> float:
        try:
            score = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {score:.6f}")
            return float(score)
        except Exception as e:
            logging.error(f"R2Score error: {e}")
            raise


class RMSE:
    """Root Mean Squared Error."""

    def calculate_score(self, y_true, y_pred) -> float:
        try:
            mse = mean_squared_error(y_true, y_pred)
            rmse = float(np.sqrt(mse))
            logging.info(f"RMSE: {rmse:.6f}")
            return rmse
        except Exception as e:
            logging.error(f"RMSE error: {e}")
            raise


class MSE:
    """Mean Squared Error (Equation 3 in Wang et al. 2024)."""

    def calculate_score(self, y_true, y_pred) -> float:
        try:
            score = float(mean_squared_error(y_true, y_pred))
            logging.info(f"MSE: {score:.6f}")
            return score
        except Exception as e:
            logging.error(f"MSE error: {e}")
            raise


class MAE:
    """Mean Absolute Error."""

    def calculate_score(self, y_true, y_pred) -> float:
        try:
            score = float(mean_absolute_error(y_true, y_pred))
            logging.info(f"MAE: {score:.6f}")
            return score
        except Exception as e:
            logging.error(f"MAE error: {e}")
            raise