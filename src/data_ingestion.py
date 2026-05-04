"""
Data Ingestion Module — Demand Forecasting Pipeline
====================================================
Simulates loading 2M+ demand records from AWS S3.
Generates synthetic product demand data across 200+ product lines,
3 years of daily granularity, multiple warehouses and regions.

In production, this module connects to S3 via boto3 and queries
Athena for historical demand. For local development and testing,
synthetic data is generated with realistic demand patterns.

Author: Bhakti Patel
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

NUM_PRODUCTS = 210
NUM_DAYS = 1095  # 3 years
START_DATE = "2021-01-01"

CATEGORIES = [
    "Electronics", "Apparel", "Home & Garden", "Health & Beauty",
    "Grocery", "Toys & Games", "Sports & Outdoors", "Automotive",
    "Office Supplies", "Pet Supplies"
]

WAREHOUSES = ["WH-East", "WH-West", "WH-Central", "WH-South", "WH-Northeast"]

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]

SEASONAL_PEAKS = {
    "Electronics": [11, 12],        # Holiday season
    "Apparel": [3, 4, 9, 10],      # Spring & Fall
    "Home & Garden": [4, 5, 6],    # Spring/Summer
    "Health & Beauty": [1, 2, 12], # New Year & Holiday
    "Grocery": [11, 12],           # Thanksgiving/Holiday
    "Toys & Games": [11, 12],      # Holiday
    "Sports & Outdoors": [5, 6, 7, 8],  # Summer
    "Automotive": [3, 4, 5],       # Spring maintenance
    "Office Supplies": [8, 9, 1],  # Back to school & New Year
    "Pet Supplies": [5, 6, 7],     # Summer
}


def generate_product_catalog(num_products: int) -> pd.DataFrame:
    """
    Generate a product catalog with realistic attributes.

    Parameters
    ----------
    num_products : int
        Number of unique product lines to generate.

    Returns
    -------
    pd.DataFrame
        Product catalog with product_id, category, base_price, base_demand.
    """
    np.random.seed(42)

    products = []
    for i in range(num_products):
        category = CATEGORIES[i % len(CATEGORIES)]
        base_price = np.random.uniform(5.99, 299.99)
        base_demand = np.random.randint(10, 500)

        products.append({
            "product_id": f"PROD-{i+1:04d}",
            "category": category,
            "base_price": round(base_price, 2),
            "base_demand": base_demand,
            "warehouse": WAREHOUSES[i % len(WAREHOUSES)],
            "region": REGIONS[i % len(REGIONS)],
        })

    return pd.DataFrame(products)


def generate_demand_signal(
    base_demand: int,
    category: str,
    dates: pd.DatetimeIndex,
    product_seed: int
) -> np.ndarray:
    """
    Generate realistic demand time series for a single product.

    Incorporates:
    - Weekly seasonality (lower on weekends for B2B, higher for B2C)
    - Monthly/seasonal patterns based on category
    - Year-over-year growth trend
    - Random noise and occasional demand spikes (promotions)
    - Holiday effects

    Parameters
    ----------
    base_demand : int
        Average daily demand units.
    category : str
        Product category for seasonal pattern selection.
    dates : pd.DatetimeIndex
        Date range for the time series.
    product_seed : int
        Random seed for reproducibility per product.

    Returns
    -------
    np.ndarray
        Daily demand values (integer units sold).
    """
    rng = np.random.RandomState(product_seed)
    n_days = len(dates)

    # Base demand with slight upward trend (3-8% annual growth)
    annual_growth = 1 + rng.uniform(0.03, 0.08)
    trend = np.array([
        base_demand * (annual_growth ** (i / 365)) for i in range(n_days)
    ])

    # Weekly pattern: day-of-week effects
    dow = dates.dayofweek.values
    weekly_multiplier = np.ones(n_days)
    if category in ["Office Supplies", "Automotive"]:
        # B2B categories: lower weekends
        weekly_multiplier[dow >= 5] = 0.4
    else:
        # B2C categories: slightly higher weekends
        weekly_multiplier[dow >= 5] = 1.15

    # Seasonal pattern
    months = dates.month.values
    seasonal_multiplier = np.ones(n_days)
    peak_months = SEASONAL_PEAKS.get(category, [])
    for m in peak_months:
        seasonal_multiplier[months == m] = rng.uniform(1.3, 1.8)

    # Holiday spikes (Black Friday, Christmas, Prime Day)
    for year in range(2021, 2024):
        # Black Friday (4th Thursday of November + Friday)
        bf_date = pd.Timestamp(f"{year}-11-25")
        mask = (dates >= bf_date) & (dates <= bf_date + timedelta(days=3))
        seasonal_multiplier[mask] = rng.uniform(2.5, 4.0)

        # Christmas week
        xmas = pd.Timestamp(f"{year}-12-20")
        mask = (dates >= xmas) & (dates <= pd.Timestamp(f"{year}-12-26"))
        seasonal_multiplier[mask] *= 1.5

    # Random promotional spikes (2-5 per year)
    for year in range(2021, 2024):
        n_promos = rng.randint(2, 6)
        promo_days = rng.choice(365, size=n_promos, replace=False)
        for day_offset in promo_days:
            start_idx = (year - 2021) * 365 + day_offset
            end_idx = min(start_idx + rng.randint(1, 5), n_days)
            if start_idx < n_days:
                seasonal_multiplier[start_idx:end_idx] *= rng.uniform(1.5, 2.5)

    # Combine components
    demand = trend * weekly_multiplier * seasonal_multiplier

    # Add noise (CV ~ 15-25%)
    noise_cv = rng.uniform(0.15, 0.25)
    noise = rng.normal(1.0, noise_cv, n_days)
    demand = demand * noise

    # Floor at 0 and convert to integer
    demand = np.maximum(demand, 0).astype(int)

    return demand


def generate_price_series(
    base_price: float,
    n_days: int,
    product_seed: int
) -> np.ndarray:
    """
    Generate realistic price time series with occasional promotions.

    Parameters
    ----------
    base_price : float
        Standard retail price.
    n_days : int
        Length of time series.
    product_seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Daily price values.
    """
    rng = np.random.RandomState(product_seed + 1000)

    prices = np.full(n_days, base_price)

    # Annual price increases (2-5%)
    for year_start in range(365, n_days, 365):
        increase = rng.uniform(1.02, 1.05)
        prices[year_start:] *= increase

    # Promotional discounts (10-30% off, 3-7 days each)
    n_promos = rng.randint(8, 20)
    for _ in range(n_promos):
        start = rng.randint(0, n_days - 7)
        duration = rng.randint(3, 8)
        discount = rng.uniform(0.70, 0.90)
        prices[start:start + duration] *= discount

    return np.round(prices, 2)


def run_ingestion(output_dir: str = None) -> pd.DataFrame:
    """
    Main ingestion pipeline. Generates synthetic demand data
    simulating a production S3 data pull.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save output CSV. Defaults to data/ in project root.

    Returns
    -------
    pd.DataFrame
        Complete demand dataset.
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING PIPELINE — Data Ingestion")
    logger.info("=" * 60)

    # Simulate S3 connection
    logger.info("Connecting to data source (S3 simulation)...")
    logger.info(f"  Bucket: s3://demand-forecast-prod/raw/")
    logger.info(f"  Date range: {START_DATE} to 2023-12-31")
    logger.info(f"  Product lines: {NUM_PRODUCTS}")

    # Generate date range
    dates = pd.date_range(start=START_DATE, periods=NUM_DAYS, freq="D")
    logger.info(f"  Date range: {dates[0].date()} to {dates[-1].date()}")

    # Generate product catalog
    catalog = generate_product_catalog(NUM_PRODUCTS)
    logger.info(f"  Generated product catalog: {len(catalog)} products")

    # Generate demand data for all products
    logger.info("Generating demand time series for all product lines...")
    all_records = []

    for idx, row in catalog.iterrows():
        if (idx + 1) % 50 == 0:
            logger.info(f"  Processing product {idx + 1}/{NUM_PRODUCTS}...")

        demand = generate_demand_signal(
            base_demand=row["base_demand"],
            category=row["category"],
            dates=dates,
            product_seed=idx
        )

        prices = generate_price_series(
            base_price=row["base_price"],
            n_days=NUM_DAYS,
            product_seed=idx
        )

        product_df = pd.DataFrame({
            "product_id": row["product_id"],
            "date": dates,
            "units_sold": demand,
            "price": prices,
            "category": row["category"],
            "warehouse": row["warehouse"],
            "region": row["region"],
        })

        all_records.append(product_df)

    # Combine all product data
    df = pd.concat(all_records, ignore_index=True)
    logger.info(f"Total records generated: {len(df):,}")

    # Calculate revenue
    df["revenue"] = df["units_sold"] * df["price"]

    # Add metadata columns (simulating Athena partition keys)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("DATA SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total records:      {len(df):>12,}")
    logger.info(f"  Unique products:    {df['product_id'].nunique():>12,}")
    logger.info(f"  Date range:         {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Categories:         {df['category'].nunique():>12}")
    logger.info(f"  Warehouses:         {df['warehouse'].nunique():>12}")
    logger.info(f"  Regions:            {df['region'].nunique():>12}")
    logger.info(f"  Avg daily demand:   {df['units_sold'].mean():>12.1f} units")
    logger.info(f"  Total revenue:      ${df['revenue'].sum():>14,.0f}")
    logger.info(f"  Memory usage:       {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Save to CSV
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "data"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "raw_demand.csv"

    logger.info(f"\nSaving to {output_path}...")
    df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / 1e6
    logger.info(f"  File size: {file_size_mb:.1f} MB")
    logger.info("Data ingestion complete.")

    return df


if __name__ == "__main__":
    df = run_ingestion()
    print(f"\n✓ Ingestion complete: {len(df):,} records saved to data/raw_demand.csv")
