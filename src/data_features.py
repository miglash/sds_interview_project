import logging

from typing import Tuple, Dict

import polars as pl
import numpy as np
from datetime import date

logger = logging.getLogger(__name__)


def build_features(
    df: pl.DataFrame, config: Dict
) -> Tuple[pl.DataFrame, pl.Series]:
    """From dataframe build features for training a timeseries model"""
    # Unpack config
    target_col = config["target_column"]
    n_input = config["time_input_length"]
    n_forecast = config["forecast_horizon"]

    df = handle_holidays(df, target=target_col, handle="impute")
    # TODO: validate no missing dates

    targets = df.get_column(target_col).to_numpy()
    X, Y = to_samples(targets, n_input, n_forecast)

    X_all = [X]
    X_all.append(add_stats_features(X))
    # X = add_feature_two(X)

    # Flatten features
    X_all = np.concat(X_all, axis=-1)
    return X_all, Y


def to_samples(
    ts: np.ndarray, n_in: int = 90, n_out: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Cuts a time-series into samples: X, Y for prediction

    Returns:
    ------
    X(n_samples, n_in)
    Y(n_samples, n_out)
    """
    Y = ts[n_in:]
    X = ts[:-n_out]

    # Avoid partial sequences - cuts of start of dataset
    n_drop = len(Y) % n_out
    Y = Y[n_drop:]
    X = X[n_drop:]

    # Segment X and Y into samples
    n_segments = len(Y) // n_out
    Xs = []
    Ys = []
    for i in range(n_segments):
        Xs.append(X[i * n_out : (i * n_out + n_in)])
        Ys.append(Y[i * n_out : (i + 1) * n_out])

    Xs = np.stack(Xs)
    Ys = np.stack(Ys)
    return Xs, Ys


# ---- FEATURES ---- #
def add_weekday_feature(
    df: pl.DataFrame, date_col: str = "DATE"
) -> pl.DataFrame:
    """Add a WEEKDAY feature to the DataFrame."""

    df = df.with_columns(
        pl.col(date_col)
        .map_elements(to_week_date, return_dtype=pl.Int64)
        .alias("WEEKDAY")
    )
    return df


def add_stats_features(X: np.ndarray) -> np.ndarray:
    """Adds basic stats: mean, std. per sample

    expects X (n_samples, n_ts_features)"""
    X_feats = [X.mean(axis=-1, keepdims=True), X.std(axis=-1, keepdims=True)]

    X_feats = np.concat(X_feats, axis=-1)
    return X_feats


# ---- CLEANUP FUNCTIONS ---- #
def remove_weekends(df: pl.DataFrame, target: str = "WEEKDAY") -> pl.DataFrame:
    """Remove weekend data. INTS 0-6 represent Mon-Sun."""

    if "WEEKDAY" not in df.columns:
        df = add_weekday_feature(df)
    WEEKEND_INT = 5
    mask_weekday = df[target] < WEEKEND_INT
    return df.filter(mask_weekday)


def handle_holidays(
    df: pl.DataFrame,
    target: str = "TOTAL_SOLD",
    threshold: int = -2100,
    handle: str = "remove",
) -> pl.DataFrame:
    """Heuristic to handle holidays based on sales threshold. Ideally would use a holiday calendar.

    Available methods:
    - remove: removes holiday rows from DataFrame
    - impute: imputes holiday rows with median value of non-holiday rows"""

    if "WEEKDAY" not in df.columns:
        df = add_weekday_feature(df)

    # Mask weekdays - otherwise would treat all weekends as outliers too
    mask_weekday = df["WEEKDAY"] < 5
    mask_outlier = df[target] > threshold
    mask = ~(mask_outlier & mask_weekday)

    if handle == "remove":
        return df.filter(mask)

    elif handle == "impute" and mask_outlier.any():
        # Impute median based on weekday
        WEEKDAY_INT = 5
        for i in range(WEEKDAY_INT):
            # TODO: improve logic
            mask_weekday_i = df["WEEKDAY"] == i
            mask_i = (~mask_outlier) & mask_weekday_i
            median_val = (
                df.filter(mask_i).select(pl.col(target).median()).item()
            )

            mask_i = mask_outlier & mask_weekday_i
            df = df.with_columns(
                pl.when(mask_i)
                .then(median_val)
                .otherwise(pl.col(target))
                .alias(target)
            )
        return df
    else:
        logger.warning(
            f"Handle method {handle} not recognized. handle set to default='remove'."
        )
        return df.filter(mask)


# ---- HELPER FUNCTIONS ---- #
def to_week_date(value: date) -> int:
    """Convert date to weekday integer using internal date method."""
    return value.weekday()  # Monday is 0 and Sunday is 6
