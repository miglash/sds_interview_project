from typing import Dict
import xgboost as xgb
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator

class BaselineModel(BaseEstimator):
    """ Doc string"""
    def __init__(self, method="last"):
        super().__init__()
        self.method = method
        self.n_targets = None
        self.n_inputs = None

    def fit(self, X, Y):
        self.n_targets = Y.shape[-1]
        self.n_inputs = X.shape[-1]
        return self

    def predict(self, X):
        n_samples = X.shape[0]

        if self.method == "last":
            value = X.reshape(-1, 1)
            predictions = np.broadcast_to(value, (n_samples, self.n_targets)) 
            return predictions
        
        elif self.method == "mean":
            value = X.mean(axis=-1, keepdims=True)
            predictions = np.broadcast_to(value, (n_samples, self.n_targets)) 
            return predictions
        else:
            return
    

def train_model(X_train, y_train, config: Dict):
    """Train the model specified in config."""
    model_type = config.get("model_type", "xgboost")
    model_params = config.get("model_params", {})

    if model_type == "xgboost":
        model = xgb.XGBRegressor(
            **model_params,
            random_state=42,
        )
    elif model_type == "baseline":
        model = BaselineModel(**model_params)
    else:
        model = LinearRegression(**model_params)

    model.fit(X_train, y_train)
    return model

def cross_validate_model(X, Y, config: Dict):
    ts_cv = TimeSeriesSplit(n_splits=3)

    results = []
    for i, (train_index, test_index) in enumerate(ts_cv.split(X)):
        model = train_model(X[train_index], Y[train_index], config)
        Y_pred_val = forecast(model, X[test_index])
        results.append((Y[test_index], Y_pred_val))
    return results
    
def forecast(model, X_val):
    """Generate forecasts."""
    return model.predict(X_val)

def save_model(model, path="./models/forecast_model_latest.pkl"):
    """Save trained model."""
    joblib.dump(model, path)
    return
