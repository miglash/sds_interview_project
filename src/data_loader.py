import polars as pl
import logging

logger = logging.getLogger(__name__)


def load_data(path: str, **kwargs) -> pl.DataFrame:
    """Load data from a CSV file into a Polars DataFrame."""
    try:
        df = pl.read_csv(path, **kwargs)
        return df
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found. Check path {path}: {e}")
        raise


def load_as_sales_data(
    df: pl.DataFrame, date_col: str = "CREATEDATE", **kwargs
) -> pl.DataFrame:
    """Converts loaded dataframe to aggregated sales data for training"""
    df = load_data(df, **kwargs)
    df = convert_date(df, date_col)
    df = aggregate_inventory(df)
    return df


def convert_date(
    df: pl.DataFrame, date_col: str = "CREATEDATE"
) -> pl.DataFrame:
    """Convert date column to DATE only (remove time component)"""
    new_date_col = "DATE"
    df = df.with_columns(
        pl.col(date_col)
        .str.to_date("%Y-%m-%d %H:%M:%S%.f")
        .alias(new_date_col),
        exact=False,
    )
    return df


def aggregate_inventory(
    df: pl.DataFrame,
    date_col: str = "DATE",
    agg_col: str = "INVENTORYDELTA",
    mask_col: str = "POS_EVENT",
    mask_type: str = "sold",
) -> pl.DataFrame:
    """Aggregate sales data by date, summing over the target column."""

    mask = df[mask_col] == mask_type
    agg_inv = df.filter(mask).group_by(date_col).agg(pl.col(agg_col).sum())

    col_name = f"TOTAL_{mask_type.upper()}"
    agg_inv = agg_inv.rename({agg_col: col_name})

    # Sort by date before returning
    return agg_inv.sort(date_col)


# Only used for EDA - remove in production
def aggregate_all(
    df: pl.DataFrame,
    date_col: str = "CREATEDATE",
    group_col: str = "POS_EVENT",
    agg_col: str = "INVENTORYDELTA",
) -> pl.DataFrame:
    """Aggregate data by date, summing over the agg column per group_col type."""
    # TODO: could rewrite using aggregate_inventory

    # Convert date column to DATE only (remove time component)
    new_date_col = "DATE"
    df = convert_date(df, date_col)

    # Aggregate agg_col values
    agg_df = df.group_by(new_date_col, group_col).agg(pl.col(agg_col).sum())
    event_types = agg_df[group_col].unique().to_list()
    dfs = []
    for event_ in event_types:
        mask = agg_df[group_col] == event_
        col_name = f"TOTAL_{event_.upper()}"
        dfs.append(
            agg_df.filter(mask)[[new_date_col, agg_col]].rename(
                {agg_col: col_name}
            )
        )

    dfs = pl.concat(dfs, how="align").sort(new_date_col)

    logger.info(
        f"Aggregated data per date. Contains event types: {event_types}"
    )
    return dfs
