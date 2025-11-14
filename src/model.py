import logging

import numpy as np
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


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

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.n_targets = Y.shape[-1]
        self.n_inputs = X.shape[-1]
        if self.n_inputs < abs(self.last_ts_value):
            err_msg = f"Less input features than expected: {self.n_inputs}."
            err_msg += f"Time series of min length {abs(self.last_ts_value)} required.\n"
            err_msg += "Adjust last_ts_value or check dimensions of inputs X"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.n_targets is None:
            err_msg = (
                "Unknown number of targets to predict. Run .fit() method first."
            )
            logger.error(err_msg)
            raise Exception(err_msg)

        n_samples = X.shape[0]
        if self.method == "last":
            value = X[:, self.last_ts_value].reshape(-1, 1)
            predictions = np.broadcast_to(value, (n_samples, self.n_targets))
            return predictions
        elif self.method == "mean":
            # Include the last timeseries value in the average
            if self.last_ts_value == -1:
                n_max = self.n_inputs
            else:
                n_max = self.last_ts_value + 1
            value = X[:, :n_max].mean(axis=-1, keepdims=True)
            predictions = np.broadcast_to(value, (n_samples, self.n_targets))
            return predictions
        else:
            return

    def _validate_method(self):
        if self.method in self.METHODS:
            return
        else:
            logger.warning(
                f"Method {self.method} unavailable. Setting to default = 'last'"
            )
            self.method = "last"
