"""
ANN Model Training Module
Artificial Neural Network implementation using TensorFlow/Keras
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ANNModel:
    """
    Artificial Neural Network Model for regression
    """
    
    def __init__(
        self,
        hidden_layers='64,32,16',
        activation='relu',
        dropout_rate=0.2,
        learning_rate=0.001,
        random_state=42
    ):
        """
        Initialize ANN model
        
        Args:
            hidden_layers (str): Comma-separated layer sizes, e.g., '64,32,16'
            activation (str): Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate (float): Dropout rate for regularization
            learning_rate (float): Learning rate for optimizer
            random_state (int): Random seed for reproducibility
        """
        self.hidden_layers = [int(x.strip()) for x in hidden_layers.split(',')]
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        logger.info(f"Initialized ANN with layers: {self.hidden_layers}")
    
    def build_model(self, input_dim):
        """
        Build the neural network architecture
        
        Args:
            input_dim (int): Number of input features
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(input_dim,)))
        
        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_regularizer=regularizers.l2(0.001),
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer (regression - single neuron, linear activation)
        model.add(layers.Dense(1, activation='linear', name='output'))
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        logger.info(f"Model architecture built successfully")
        logger.info(f"Total parameters: {model.count_params()}")
        
        return model
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1):
        """
        Train the ANN model
        
        Args:
            X_train: Training features
            y_train: Training target
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Validation data split ratio
            verbose (int): Verbosity mode
            
        Returns:
            history: Training history
        """
        if self.model is None:
            self.build_model(X_train.shape[1])
        
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        # Train the model
        logger.info(f"Training ANN for {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=verbose
        )
        
        logger.info("Training completed")
        return history
    
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
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
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
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load a trained model
        
        Args:
            filepath (str): Path to load the model from
        """
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """
        Get model architecture summary
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not built yet"
        
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        return '\n'.join(summary_list)
