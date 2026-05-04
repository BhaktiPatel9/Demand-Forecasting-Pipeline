"""
Feature Engineering Module — Demand Forecasting Pipeline
=========================================================
Transforms raw demand data into ML-ready feature matrix.

Features engineered:
- Lag features (7d, 14d, 30d)
- Rolling statistics (mean, std, min, max over multiple windows)
- Calendar features (day_of_week, month, quarter, is_weekend)
- Seasonal indicators (holiday proximity, peak season flags)
- Price signals (price_change_pct, days_since_promo)
- Trend features (YoY growth rate)

Performs time-based train/test split (80/20) to prevent data leakage.

Author: Bhakti Patel
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ─── US Holidays (simplified) ────────────────────────────────────────────────

US_HOLIDAYS = [
    "2021-01-01", "2021-01-18", "2021-02-15", "2021-05-31",
    "2021-07-04", "2021-09-06", "2021-11-25", "2021-12-25",
    "2022-01-01", "2022-01-17", "2022-02-21", "2022-05-30",
    "2022-07-04", "2022-09-05", "2022-11-24", "2022-12-25",
    "2023-01-01", "2023-01-16", "2023-02-20", "2023-05-29",
    "2023-07-04", "2023-09-04", "2023-11-23", "2023-12-25",
]


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    Load raw demand data from CSV.

    If file doesn't exist, runs data_ingestion to generate it.

    Parameters
    ----------
    data_path : str, optional
        Path to raw_demand.csv

    Returns
    -------
    pd.DataFrame
        Raw demand data.
    """
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "raw_demand.csv"

    data_path = Path(data_path)

    if not data_path.exists():
        logger.info("Raw data not found. Running data ingestion first...")
        from data_ingestion import run_ingestion
        run_ingestion()

    logger.info(f"Loading raw data from {data_path}...")
    df = pd.read_csv(data_path, parse_dates=["date"])
    logger.info(f"  Loaded {len(df):,} records")
    return df


def create_lag_features(group: pd.DataFrame, lags: list = None) -> pd.DataFrame:
    """
    Create lag features for a single product's time series.

    Parameters
    ----------
    group : pd.DataFrame
        Single product time series sorted by date.
    lags : list, optional
        List of lag periods. Defaults to [7, 14, 30, 60, 90].

    Returns
    -------
    pd.DataFrame
        DataFrame with lag columns added.
    """
    if lags is None:
        lags = [7, 14, 30, 60, 90]

    for lag in lags:
        group[f"demand_lag_{lag}d"] = group["units_sold"].shift(lag)

    return group


def create_rolling_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling window statistics.

    Windows: 7d, 14d, 30d
    Statistics: mean, std, min, max

    Parameters
    ----------
    group : pd.DataFrame
        Single product time series sorted by date.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling feature columns.
    """
    windows = [7, 14, 30]

    for window in windows:
        rolling = group["units_sold"].rolling(window=window, min_periods=1)
        group[f"rolling_mean_{window}d"] = rolling.mean()
        group[f"rolling_std_{window}d"] = rolling.std()
        group[f"rolling_min_{window}d"] = rolling.min()
        group[f"rolling_max_{window}d"] = rolling.max()

    # Demand volatility (coefficient of variation over 30d)
    group["demand_cv_30d"] = (
        group["rolling_std_30d"] / group["rolling_mean_30d"].replace(0, np.nan)
    )

    return group


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract calendar-based features from date column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'date' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with calendar features added.
    """
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Holiday proximity (days to nearest holiday)
    holidays = pd.to_datetime(US_HOLIDAYS)
    df["days_to_holiday"] = df["date"].apply(
        lambda x: min(abs((x - h).days) for h in holidays)
    )
    df["is_holiday_week"] = (df["days_to_holiday"] <= 3).astype(int)

    return df


def create_price_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create price-based features for demand elasticity signals.

    Parameters
    ----------
    group : pd.DataFrame
        Single product time series with 'price' column.

    Returns
    -------
    pd.DataFrame
        DataFrame with price features added.
    """
    # Price change percentage (day-over-day)
    group["price_change_pct"] = group["price"].pct_change().fillna(0)

    # Price relative to 30-day average (discount indicator)
    group["price_vs_avg_30d"] = (
        group["price"] / group["price"].rolling(30, min_periods=1).mean()
    )

    # Is on promotion (price < 90% of rolling average)
    group["is_promo"] = (group["price_vs_avg_30d"] < 0.90).astype(int)

    # Days since last promotion
    promo_mask = group["is_promo"] == 1
    group["days_since_promo"] = (~promo_mask).groupby(
        promo_mask.cumsum()
    ).cumcount()

    # Price elasticity proxy (rolling correlation of price change & demand change)
    group["demand_change"] = group["units_sold"].pct_change()
    group["price_demand_corr_30d"] = (
        group["price_change_pct"]
        .rolling(30, min_periods=7)
        .corr(group["demand_change"])
    )

    return group


def create_trend_features(group: pd.DataFrame) -> pd.DataFrame:
    """
    Create trend and growth features.

    Parameters
    ----------
    group : pd.DataFrame
        Single product time series.

    Returns
    -------
    pd.DataFrame
        DataFrame with trend features.
    """
    # Year-over-year growth (compare to same day last year)
    group["demand_yoy"] = group["units_sold"] / group["units_sold"].shift(365).replace(0, np.nan)

    # 30-day trend (slope of demand over last 30 days)
    group["trend_30d"] = (
        group["rolling_mean_7d"] - group["rolling_mean_30d"]
    ) / group["rolling_mean_30d"].replace(0, np.nan)

    return group


def run_feature_engineering(
    data_path: str = None,
    output_dir: str = None,
    sample_products: int = None
) -> tuple:
    """
    Main feature engineering pipeline.

    Parameters
    ----------
    data_path : str, optional
        Path to raw demand CSV.
    output_dir : str, optional
        Output directory for feature files.
    sample_products : int, optional
        Number of products to process (for faster testing).

    Returns
    -------
    tuple
        (train_df, test_df) DataFrames
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING PIPELINE — Feature Engineering")
    logger.info("=" * 60)

    # Load data
    df = load_raw_data(data_path)

    # Optional sampling for development
    if sample_products:
        products = df["product_id"].unique()[:sample_products]
        df = df[df["product_id"].isin(products)].copy()
        logger.info(f"  Sampled to {sample_products} products for development")

    # Sort by product and date
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # ─── Calendar Features (vectorized, apply to full df) ─────────────────
    logger.info("Creating calendar features...")
    df = create_calendar_features(df)

    # ─── Per-product features (lag, rolling, price, trend) ────────────────
    logger.info("Creating per-product features (lags, rolling stats, price signals)...")
    processed_groups = []
    products = df["product_id"].unique()

    for i, product_id in enumerate(products):
        if (i + 1) % 50 == 0:
            logger.info(f"  Processing product {i + 1}/{len(products)}...")

        group = df[df["product_id"] == product_id].copy()
        group = create_lag_features(group)
        group = create_rolling_features(group)
        group = create_price_features(group)
        group = create_trend_features(group)
        processed_groups.append(group)

    df = pd.concat(processed_groups, ignore_index=True)

    # ─── Category encoding ────────────────────────────────────────────────
    logger.info("Encoding categorical features...")
    df = pd.get_dummies(df, columns=["category", "warehouse", "region"], prefix=["cat", "wh", "reg"])

    # ─── Target variable ──────────────────────────────────────────────────
    df["target"] = df["units_sold"]

    # ─── Drop rows with NaN from lag features (first 90 days per product) ─
    initial_rows = len(df)
    df = df.dropna(subset=["demand_lag_90d"])
    logger.info(f"  Dropped {initial_rows - len(df):,} rows (insufficient history)")
    logger.info(f"  Remaining: {len(df):,} rows")

    # ─── Time-based train/test split (80/20) ──────────────────────────────
    logger.info("Performing time-based train/test split...")
    split_date = df["date"].quantile(0.8)
    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()

    logger.info(f"  Train set: {len(train_df):,} rows ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    logger.info(f"  Test set:  {len(test_df):,} rows ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # ─── Feature summary ──────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in [
        "product_id", "date", "units_sold", "revenue", "target",
        "year", "week", "demand_change"
    ]]

    logger.info(f"\n{'=' * 60}")
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Total features: {len(feature_cols)}")
    logger.info(f"  Feature groups:")
    logger.info(f"    - Lag features:      {sum(1 for c in feature_cols if 'lag' in c)}")
    logger.info(f"    - Rolling features:  {sum(1 for c in feature_cols if 'rolling' in c)}")
    logger.info(f"    - Calendar features: {sum(1 for c in feature_cols if c in ['day_of_week', 'month', 'quarter', 'week_of_year', 'is_weekend', 'is_month_start', 'is_month_end'])}")
    logger.info(f"    - Price features:    {sum(1 for c in feature_cols if 'price' in c or 'promo' in c)}")
    logger.info(f"    - Trend features:    {sum(1 for c in feature_cols if 'trend' in c or 'yoy' in c)}")
    logger.info(f"    - Category dummies:  {sum(1 for c in feature_cols if c.startswith('cat_') or c.startswith('wh_') or c.startswith('reg_'))}")

    # ─── Feature importance preview (correlation with target) ─────────────
    logger.info("\nTop 15 features by correlation with target:")
    numeric_features = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32]]
    correlations = df[numeric_features].corrwith(df["target"]).abs().sort_values(ascending=False)

    for i, (feat, corr) in enumerate(correlations.head(15).items()):
        logger.info(f"  {i+1:2d}. {feat:<30s} r={corr:.4f}")

    # ─── Save processed features ─────────────────────────────────────────
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train_features.csv"
    test_path = output_dir / "test_features.csv"

    logger.info(f"\nSaving train features to {train_path}...")
    train_df.to_csv(train_path, index=False)

    logger.info(f"Saving test features to {test_path}...")
    test_df.to_csv(test_path, index=False)

    logger.info("Feature engineering complete.")
    return train_df, test_df


if __name__ == "__main__":
    # Use sample for faster local testing; remove for full pipeline
    sample = 20 if "--sample" in sys.argv else None
    train_df, test_df = run_feature_engineering(sample_products=sample)
    print(f"\n✓ Feature engineering complete:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")
