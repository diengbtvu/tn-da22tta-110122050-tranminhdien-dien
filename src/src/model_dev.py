import logging
from abc import ABC, abstractmethod

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# -----------------------------------------------------------------------
# Abstract base
# -----------------------------------------------------------------------
class Model(ABC):
    """Abstract base class for all regression models."""

    @abstractmethod
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        pass


class RandomForestModel(Model):
    """
    Random Forest Regressor — best performer in Wang et al. (2024),
    R² ≈ 0.875.
    """
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        try:
            defaults = {
                'n_estimators': 100,
                'random_state': 42,
            }
            defaults.update(kwargs)
            rf = RandomForestRegressor(**defaults)
            rf.fit(X_train, y_train)
            logging.info("Random Forest Regressor training completed.")
            return rf
        except Exception as e:
            logging.error(f"RandomForest training error: {e}")
            raise


class SVMModel(Model):
    """
    Support Vector Machine Regressor with RBF kernel.
    Default: C=1.0, gamma='scale' (as in paper).
    """
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        try:
            defaults = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
            defaults.update(kwargs)
            svm = SVR(**defaults)
            svm.fit(X_train, y_train)
            logging.info("SVM Regressor training completed.")
            return svm
        except Exception as e:
            logging.error(f"SVM training error: {e}")
            raise


class KNNModel(Model):
    """
    K-Nearest Neighbours Regressor.
    Default: k=5 (as in paper).
    """
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        try:
            defaults = {'n_neighbors': 5}
            defaults.update(kwargs)
            knn = KNeighborsRegressor(**defaults)
            knn.fit(X_train, y_train)
            logging.info(f"KNN Regressor (k={defaults['n_neighbors']}) training completed.")
            return knn
        except Exception as e:
            logging.error(f"KNN training error: {e}")
            raise


class DecisionTreeModel(Model):
    """
    Decision Tree Regressor.
    """
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        try:
            defaults = {'random_state': 42}
            defaults.update(kwargs)
            dt = DecisionTreeRegressor(**defaults)
            dt.fit(X_train, y_train)
            logging.info("Decision Tree Regressor training completed.")
            return dt
        except Exception as e:
            logging.error(f"DecisionTree training error: {e}")
            raise


class XGBoostModel(Model):
    """
    XGBoost Regressor.
    """
    def train(self, X_train, y_train, **kwargs) -> RegressorMixin:
        if not XGBOOST_AVAILABLE:
            raise ImportError("xgboost is not installed. Run: pip install xgboost")
        try:
            defaults = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'random_state': 42,
                'verbosity': 0,
            }
            defaults.update(kwargs)
            xgb = XGBRegressor(**defaults)
            xgb.fit(X_train, y_train)
            logging.info("XGBoost Regressor training completed.")
            return xgb
        except Exception as e:
            logging.error(f"XGBoost training error: {e}")
            raise
