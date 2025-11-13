import polars as pl 
from datetime import date
import logging

def add_weekday_feature(df: pl.DataFrame, date_col: str = "DATE") -> pl.DataFrame:
    """Add a WEEKDAY feature to the DataFrame."""

    df = df.with_columns(pl.col(date_col).map_elements(to_week_date, return_dtype=pl.Int64).alias("WEEKDAY"))
    return df 

### CLEANUP FUNCTIONS  ###
def remove_weekends(df: pl.DataFrame, target: str = "WEEKDAY") -> pl.DataFrame:
    """Remove weekend data. INTS 0-6 represent Mon-Sun."""

    if "WEEKDAY" not in df.columns:
        df = add_weekday_feature(df)
    WEEKEND_INT = 5
    mask_weekday = df[target] < WEEKEND_INT
    return df.filter(mask_weekday)

def handle_holidays(df: pl.DataFrame, target: str = "TOTAL_SALES", threshold: int =-2100,
                    handle: str = "remove") -> pl.DataFrame:
    """Heuristic to handle holidays based on sales threshold. Ideally would use a holiday calendar.
    
    Available methods:
    - remove: removes holiday rows from DataFrame
    - impute: imputes holiday rows with median value of non-holiday rows"""

    if "WEEKDAY" not in df.columns:
        df = add_weekday_feature(df)

    #Mask weekdays - otherwise would treat all weekends as outliers too
    mask_weekday = df["WEEKDAY"] < 5  
    mask_outlier = df[target] > threshold
    mask = ~(mask_outlier & mask_weekday)

    if handle == "remove":
        return df.filter(mask)
    
    elif handle == "impute" and mask_outlier.any():
        ## Impute median based on weekday
        WEEKDAY_INT = 5
        for i in range(WEEKDAY_INT):
            # TODO: improve logic
            mask_weekday_i = df["WEEKDAY"] == i
            mask_i = (~mask_outlier) & mask_weekday_i
            median_val = df.filter(mask_i).select(pl.col(target).median()).item()

            mask_i = mask_outlier & mask_weekday_i
            df = df.with_columns(
                pl.when(mask_i).then(median_val).otherwise(pl.col(target)).alias(target)
            )
        return df
    else:
        #logging.warning(f"Handle method {handle} not recognized. handle set to default='remove'.")
        return df.filter(mask)

### HELPER FUNCTIONS ###
def to_week_date(value: date) -> int:
    """Convert date to weekday integer using internal date method."""
    return value.weekday()  # Monday is 0 and Sunday is 6