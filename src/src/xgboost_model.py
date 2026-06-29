"""
XGBoost Model Training Module
Extreme Gradient Boosting implementation
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """
    XGBoost Model for regression
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        learning_rate=0.1,
        random_state=42,
        **kwargs
    ):
        """
        Initialize XGBoost model
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate (eta)
            random_state (int): Random seed for reproducibility
            **kwargs: Additional XGBoost parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth is not None else 6
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Build model parameters
        self.params = {
            'objective': 'reg:squarederror',
            'max_depth': self.max_depth,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'random_state': random_state,
            'n_jobs': -1,  # Use all CPU cores
            'tree_method': 'hist',  # Faster histogram-based algorithm
            'verbosity': 1,
        }
        
        # Add any additional parameters
        self.params.update(kwargs)
        
        self.model = None
        logger.info(f"Initialized XGBoost with n_estimators={n_estimators}, max_depth={self.max_depth}, lr={learning_rate}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            early_stopping_rounds (int): Early stopping rounds
            
        Returns:
            model: Trained XGBoost model
        """
        logger.info("Training XGBoost model...")
        
        # Create DMatrix for efficient training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Prepare evaluation list
        eval_list = [(dtrain, 'train')]
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_list.append((dval, 'validation'))
        
        # Train model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_list,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose_eval=10
        )
        
        logger.info("Training completed")
        return self.model
    
    def train_sklearn_api(self, X_train, y_train, eval_set=None, early_stopping_rounds=10):
        """
        Train using sklearn-style API
        
        Args:
            X_train: Training features
            y_train: Training target
            eval_set: List of (X, y) tuples for evaluation
            early_stopping_rounds (int): Early stopping rounds (not used, kept for API compatibility)
            
        Returns:
            model: Trained XGBoost model
        """
        logger.info("Training XGBoost model (sklearn API)...")
        
        self.model = xgb.XGBRegressor(**self.params)
        
        # Simple training without early stopping to avoid version compatibility issues
        fit_params = {'verbose': 10}
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            
        self.model.fit(X_train, y_train, **fit_params)
        
        logger.info("Training completed")
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            predictions: Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Handle both native XGBoost and sklearn API
        if isinstance(self.model, xgb.Booster):
            dtest = xgb.DMatrix(X)
            predictions = self.model.predict(dtest)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        metrics = {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae),
            'mse': float(mse)
        }
        
        logger.info(f"Evaluation metrics: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
        return metrics
    
    def save(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Handle both native XGBoost and sklearn API
        if isinstance(self.model, xgb.Booster):
            self.model.save_model(filepath)
        else:
            self.model.save_model(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to load the model from
        """
        # Try to load as Booster first
        try:
            self.model = xgb.Booster()
            self.model.load_model(filepath)
        except:
            # Fallback to sklearn API
            self.model = xgb.XGBRegressor()
            self.model.load_model(filepath)
        
        logger.info(f"Model loaded from {filepath}")
        return self.model
    
    def get_feature_importance(self, importance_type='gain'):
        """
        Get feature importance
        
        Args:
            importance_type (str): Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            dict: Feature importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if isinstance(self.model, xgb.Booster):
            importance = self.model.get_score(importance_type=importance_type)
        else:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
        
        return importance
