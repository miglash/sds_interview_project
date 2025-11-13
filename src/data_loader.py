import polars as pl
import logging

def load_data(path: str, date_col: str = "CREATEDATE", **kwargs) -> pl.DataFrame:
    """Load data from a CSV file into a Polars DataFrame."""
    try:
        df = pl.read_csv(path, **kwargs)
        return df
    
    except FileNotFoundError as e:
        logging.error(f"Dataset file not found. Check path {path}: {e}")
        raise    


def aggregate_all(df: pl.DataFrame, date_col: str = "CREATEDATE",
                    group_col: str = "POS_EVENT", agg_col: str = "INVENTORYDELTA"
                    ) -> pl.DataFrame:
    """Aggregate data by date, summing over the agg column per group_col type."""

    # Convert date column to DATE only (remove time component)
    new_date_col = "DATE"
    df = df.with_columns(pl.col(date_col).str.to_date("%Y-%m-%d %H:%M:%S%.f").alias(new_date_col), exact=False)

    agg_df = df.group_by(new_date_col, group_col).agg(pl.col(agg_col).sum())
    
    dfs = []
    event_types = agg_df[group_col].unique().to_list()
    for event_ in event_types:
        mask = agg_df[group_col] == event_
        col_name = f"TOTAL_{event_.upper()}"
        dfs.append(agg_df.filter(mask)[[new_date_col, agg_col]].rename({agg_col: col_name}))

    dfs = pl.concat(dfs, how="align").sort(new_date_col)
    
    logging.info(f"Aggregated data per date. Contains event types: {event_types}")
    return dfs