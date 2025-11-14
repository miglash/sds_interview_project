from typing import Dict, List, Tuple
import xgboost as xgb
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator
import logging
from pathlib import Path


class BaselineModel(BaseEstimator):
    """
    An intentionally naive baseline estimator for time-series
    predictions. Supports two naive strategies for producing
    predictions: "mean" and "last" (value).
    The class follows a minimal scikit-learn-like API.

    Parameters
    ----------
    method : str, optional (default="last")
        Strategy used to generate predictions. Supported options are:
        - "last": take a single input feature (indexed by last_ts_value) and
          repeat it for every target.
        - "mean": compute the mean across input features from the start up to
          the index given by last_ts_value, repeat value for every target.
    last_ts_value : int, optional (default=-1)
        Index used to determine the last feature of the timeseries. In case
        other feature types are included. (default = -1) assumes all values
        in X are a timeseries sequence.

    Methods
    -------
    fit(X, Y)
        Record input/output dimensionalities. Expects:
        - X: array-like of shape (n_samples, n_inputs)
        - Y: array-like of shape (n_samples, n_targets)
        The method sets n_targets and n_inputs from Y and X respectively.

    predict(X)
        Produce predictions using the configured strategy. Expects X of shape
        (n_samples, n_inputs) and returns a numpy.ndarray of shape
        (n_samples, n_targets) where n_targets was set during fit().
    """

    def __init__(self, method: str = "last", last_ts_value: int = -1):
        super().__init__()
        self.METHODS = ("last", "mean")
        self.method = method.lower()
        self._validate_method()

        self.n_targets = None
        self.n_inputs = None
        self.last_ts_value = last_ts_value

    def fit(self, X, Y):
        self.n_targets = Y.shape[-1]
        self.n_inputs = X.shape[-1]
        if self.n_inputs < abs(self.last_ts_value):
            err_msg = f"Less input features {self.n_inputs} than expected."
            err_msg += f"Time series of min length {abs(self.last_ts_value)} required.\n"
            err_msg += "Adjust last_ts_value or check dimensions of inputs X"
            logging.error(err_msg)
            raise ValueError(err_msg)
        return self

    def predict(self, X):
        if self.n_targets is None:
            err_msg = (
                "Unknown number of targets to predict. Run .fit() method first."
            )
            logging.error(err_msg)
            raise Exception(err_msg)

        n_samples = X.shape[0]
        if self.method == "last":
            value = X[:, self.last_ts_value].reshape(-1, 1)
            predictions = np.broadcast_to(value, (n_samples, self.n_targets))
            return predictions
        elif self.method == "mean":
            value = X[:, : (self.last_ts_value + 1)].mean(
                axis=-1, keepdims=True
            )
            predictions = np.broadcast_to(value, (n_samples, self.n_targets))
            return predictions
        else:
            return

    def _validate_method(self):
        if self.method in self.METHODS:
            return
        else:
            logging.warning(
                f"Method {self.method} unavailable. Setting to default = 'last'"
            )
            self.method = "last"


# ---- Model functionality ---- #
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
          - "model_type" (str): 'xgboost' (default), 'baseline', or other (falls back to LinearRegression).
          - "model_params" (dict): keyword arguments forwarded to the chosen model constructor.

    Returns
    -------
    object
        A fitted model instance (scikit-learn-like estimator).

    Notes
    -----
    - No explicit input validation is performed; X and Y must be shape-compatible for model.fit.
    """
    model_type = config.get("model_type", "xgboost").lower()
    model_params = config.get("model_params", {})

    if model_type == "baseline":
        model = BaselineModel(**model_params)
    if model_type == "xgboost":
        model = xgb.XGBRegressor(**model_params)
    else:
        model = LinearRegression(**model_params)

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
