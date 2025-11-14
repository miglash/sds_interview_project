from pathlib import Path
import joblib

import logging
from typing import Dict, List, Tuple
import numpy as np

import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.model import BaselineModel


def train_model(X: np.ndarray, Y: np.ndarray, config: Dict):
    """
    Train the model specified in config and return the fitted estimator.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix for training (n_samples, n_features).
    Y : np.ndarray
        Target array (n_samples, n_targets).
    config : Dict
        Configuration dict with optional keys:
          - "model_type" (str): 'xgboost' (default), 'baseline', or 'linear' regression.
          - "model_params" (dict): keyword arguments forwarded to the chosen model constructor.

    Returns
    -------
    object
        A fitted model instance (scikit-learn-like estimator).

    Notes
    -----
    - No explicit input validation is performed; X and Y must be shape-compatible for model.fit.
    """
    VALID_TYPES = ("baseline", "linear", "xgboost")

    model_type = config.get("model_type", None)
    model_params = config.get("model_params", {})

    if model_type is None or model_type.lower() not in VALID_TYPES:
        logging.warning(
            f"{model_type} model type not implemented. Using default type 'xgboost'"
        )
        model_type = "xgboost"

    if model_type == "baseline":
        model = BaselineModel(**model_params)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(**model_params)
    elif model_type == "linear":
        model = LinearRegression(**model_params)
    else:
        err_msg = f"{model_type} model type couldn't be trained."
        logging.error(err_msg)
        raise ValueError(err_msg)

    model.fit(X, Y)
    return model


def cross_validate_model(
    X: np.ndarray, Y: np.ndarray, config: Dict, n_cv_split: int = 3
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Cross-validate a time-series forecasting model using TimeSeriesSplit.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features). Rows must be ordered in time.
    Y : np.ndarray
        Target array of shape (n_samples, n_targets), aligned with X.
    config : Dict
        Configuration dict forwarded to train_model (e.g. contains "model_type" and "model_params").
    n_cv_split : int, optional (default=3)
        Number of splits to use for TimeSeriesSplit.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (Y_true, Y_pred) tuples, one per validation fold.
        Y_true and Y_pred have shape: (n_val_samples, n_targets).
    """
    ts_cv = TimeSeriesSplit(n_splits=n_cv_split)

    results = []
    for train_index, test_index in ts_cv.split(X):
        model = train_model(X[train_index], Y[train_index], config)
        Y_pred_val = forecast(model, X[test_index])
        results.append((Y[test_index], Y_pred_val))
    return results


def forecast(model, X: np.ndarray, config: dict) -> np.ndarray:
    """Generate forecasts using a fitted model.

    Placeholder: if different prediction methods need to be handled.

    Parameters
    ----------
    model : object
        Trained estimator exposing a predict(X) method.
    X : np.ndarray
        Input features for which to generate forecasts.
    config : dict
        Contains information on the model

    Returns
    -------
    np.ndarray
        Predictions produced by model.predict(X).
    """
    return model.predict(X)


def save_model(model, path: str = "./models/forecast_model_latest.pkl") -> str:
    """Save a trained model to disk and return the absolute file path.

    Parameters
    ----------
    model : object
        Trained model object to serialize (required).
    path: str
        Destination file path (default "./models/forecast_model_latest.pkl").
        Unknown extensions are replaced with ".pkl".

    Returns
    -------
    str
        Absolute path to the saved model file.
    """
    if model is None:
        raise ValueError("No model provided to save.")

    p = Path(path).expanduser()
    if p.is_dir():
        raise IsADirectoryError(
            f"Save path '{p}' is a directory; expected a file path."
        )

    # Ensure reasonable suffix
    if p.suffix not in (".pkl", ".joblib"):
        logging.warning(
            f"Unrecognized file extension '{p.suffix}'. Using '.pkl' suffix."
        )
        p = p.with_suffix(".pkl")

    p.parent.mkdir(parents=True, exist_ok=True)

    try:
        joblib.dump(model, p)
    except Exception as exc:
        logging.error(f"Failed to save model to '{p}': {exc}")
        raise

    return str(p.resolve())


def load_model(path: str = "./models/forecast_model_latest.pkl") -> object:
    """Load a serialized model from disk.

    Parameters
    ----------
    path : str, optional
        File path to the serialized model.
        Defaults to "./models/forecast_model_latest.pkl".

    Returns
    -------
    object
        The deserialized model instance.
    """

    p = Path(path).expanduser()

    if p.is_dir():
        raise IsADirectoryError(
            f"Load path '{p}' is a directory; expected a file path."
        )

    if not p.exists():
        raise FileNotFoundError(f"Model file not found: '{p}'")

    if p.suffix not in (".pkl", ".joblib"):
        logging.warning(
            f"Unrecognized file extension '{p.suffix}'. Attempting to load anyway."
        )

    try:
        model = joblib.load(p)
    except Exception as exc:
        logging.error(f"Failed to load model from '{p}': {exc}")
        raise

    return model
