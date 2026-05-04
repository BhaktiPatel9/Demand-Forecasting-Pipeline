"""
ARIMA Model Module — Demand Forecasting Pipeline
==================================================
Fits Seasonal ARIMA models on individual product time series.
Captures trend, seasonality, and autocorrelation patterns.

The ARIMA component excels at:
- Capturing weekly/monthly seasonality
- Modeling trend shifts
- Short-term forecast accuracy (1-14 days)

Limitations addressed by ensemble:
- Cannot incorporate external features (price, promotions)
- Struggles with sudden demand shocks
- Computationally expensive for 200+ product lines

Author: Bhakti Patel
"""

import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    float
        MAPE as percentage.
    """
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Parameters
    ----------
    actual : np.ndarray
        Actual values.
    predicted : np.ndarray
        Predicted values.

    Returns
    -------
    float
        RMSE value.
    """
    return np.sqrt(np.mean((actual - predicted) ** 2))


def fit_arima_product(
    series: pd.Series,
    order: tuple = (2, 1, 2),
    seasonal_order: tuple = (1, 1, 1, 7),
    forecast_horizon: int = 30
) -> dict:
    """
    Fit ARIMA model on a single product's demand series.

    Uses statsmodels SARIMAX if available, otherwise falls back
    to a simplified autoregressive approach for demonstration.

    Parameters
    ----------
    series : pd.Series
        Daily demand time series (indexed by date).
    order : tuple
        ARIMA (p, d, q) order.
    seasonal_order : tuple
        Seasonal (P, D, Q, s) order.
    forecast_horizon : int
        Number of days to forecast.

    Returns
    -------
    dict
        Dictionary with forecast, actuals, and metrics.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        # Split into train/test
        train_size = len(series) - forecast_horizon
        train = series[:train_size]
        test = series[train_size:]

        # Fit SARIMAX model
        model = SARIMAX(
            train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        fitted = model.fit(disp=False, maxiter=200)

        # Generate forecast
        forecast = fitted.forecast(steps=forecast_horizon)
        forecast = np.maximum(forecast.values, 0)  # No negative demand

        return {
            "forecast": forecast,
            "actual": test.values[:forecast_horizon],
            "train_size": train_size,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "method": "SARIMAX"
        }

    except ImportError:
        logger.warning("statsmodels not available. Using simplified AR approach.")
        return _fit_simple_ar(series, forecast_horizon)


def _fit_simple_ar(series: pd.Series, forecast_horizon: int = 30) -> dict:
    """
    Simplified autoregressive forecast as fallback.
    Uses weighted combination of recent history and seasonal pattern.

    Parameters
    ----------
    series : pd.Series
        Demand time series.
    forecast_horizon : int
        Days to forecast.

    Returns
    -------
    dict
        Forecast results dictionary.
    """
    train_size = len(series) - forecast_horizon
    train = series.values[:train_size]
    test = series.values[train_size:train_size + forecast_horizon]

    # Weighted average of:
    # 1. Last 7 days pattern (repeated)
    # 2. Same period last year
    # 3. 30-day rolling mean

    last_7d = train[-7:]
    last_30d_mean = np.mean(train[-30:])

    # Same week last year (if enough data)
    if train_size > 365:
        same_period_ly = train[train_size - 365:train_size - 365 + forecast_horizon]
        yoy_growth = np.mean(train[-30:]) / np.mean(train[-395:-365]) if np.mean(train[-395:-365]) > 0 else 1.0
        seasonal_component = same_period_ly * yoy_growth
    else:
        seasonal_component = np.full(forecast_horizon, last_30d_mean)

    # Combine components
    recent_pattern = np.tile(last_7d, (forecast_horizon // 7) + 1)[:forecast_horizon]

    forecast = (
        0.4 * recent_pattern +
        0.35 * seasonal_component[:forecast_horizon] +
        0.25 * last_30d_mean
    )

    # Add slight noise for realism
    np.random.seed(42)
    noise = np.random.normal(1.0, 0.05, forecast_horizon)
    forecast = np.maximum(forecast * noise, 0)

    return {
        "forecast": forecast,
        "actual": test,
        "train_size": train_size,
        "aic": None,
        "bic": None,
        "method": "SimpleAR"
    }


def run_arima_forecasting(data_path: str = None, n_products: int = 5) -> dict:
    """
    Run ARIMA forecasting on sample product lines.

    Parameters
    ----------
    data_path : str, optional
        Path to raw demand CSV.
    n_products : int
        Number of products to forecast (for demonstration).

    Returns
    -------
    dict
        Aggregated results across all products.
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING PIPELINE — ARIMA Model")
    logger.info("=" * 60)

    # Load data
    if data_path is None:
        data_path = Path(__file__).parent.parent / "data" / "raw_demand.csv"

    if not Path(data_path).exists():
        logger.info("Raw data not found. Generating synthetic data...")
        from data_ingestion import run_ingestion
        run_ingestion()

    df = pd.read_csv(data_path, parse_dates=["date"])
    logger.info(f"Loaded {len(df):,} records")

    # Select top products by volume for ARIMA fitting
    product_volumes = df.groupby("product_id")["units_sold"].sum().sort_values(ascending=False)
    top_products = product_volumes.head(n_products).index.tolist()

    logger.info(f"\nFitting ARIMA on top {n_products} products by volume...")
    logger.info(f"  Model order: (2, 1, 2)")
    logger.info(f"  Seasonal order: (1, 1, 1, 7)")
    logger.info(f"  Forecast horizon: 30 days")

    results = []
    all_forecasts = []
    all_actuals = []

    for i, product_id in enumerate(top_products):
        product_data = df[df["product_id"] == product_id].sort_values("date")
        series = product_data.set_index("date")["units_sold"]

        logger.info(f"\n  [{i+1}/{n_products}] Product: {product_id}")
        logger.info(f"    Series length: {len(series)} days")
        logger.info(f"    Mean demand: {series.mean():.1f} units/day")

        # Fit model
        result = fit_arima_product(series, forecast_horizon=30)

        # Calculate metrics
        forecast = result["forecast"]
        actual = result["actual"]

        # Ensure same length
        min_len = min(len(forecast), len(actual))
        forecast = forecast[:min_len]
        actual = actual[:min_len]

        mape = calculate_mape(actual, forecast)
        rmse = calculate_rmse(actual, forecast)

        logger.info(f"    Method: {result['method']}")
        logger.info(f"    MAPE: {mape:.2f}%")
        logger.info(f"    RMSE: {rmse:.1f} units")
        if result["aic"]:
            logger.info(f"    AIC: {result['aic']:.1f}")

        results.append({
            "product_id": product_id,
            "mape": mape,
            "rmse": rmse,
            "method": result["method"],
            "mean_demand": series.mean()
        })

        all_forecasts.extend(forecast)
        all_actuals.extend(actual)

    # ─── Aggregate Results ────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    overall_mape = calculate_mape(np.array(all_actuals), np.array(all_forecasts))
    overall_rmse = calculate_rmse(np.array(all_actuals), np.array(all_forecasts))

    logger.info(f"\n{'=' * 60}")
    logger.info("ARIMA MODEL RESULTS")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Products evaluated:  {n_products}")
    logger.info(f"  Forecast horizon:    30 days")
    logger.info(f"  Overall MAPE:        {overall_mape:.2f}%")
    logger.info(f"  Overall RMSE:        {overall_rmse:.1f} units")
    logger.info(f"  Avg product MAPE:    {results_df['mape'].mean():.2f}%")
    logger.info(f"  Best product MAPE:   {results_df['mape'].min():.2f}%")
    logger.info(f"  Worst product MAPE:  {results_df['mape'].max():.2f}%")

    # Save ARIMA forecasts for ensemble
    output_dir = Path(__file__).parent.parent / "data"
    forecast_df = pd.DataFrame({
        "actual": all_actuals,
        "arima_forecast": all_forecasts
    })
    forecast_path = output_dir / "arima_forecasts.csv"
    forecast_df.to_csv(forecast_path, index=False)
    logger.info(f"\n  Forecasts saved to {forecast_path}")

    return {
        "overall_mape": overall_mape,
        "overall_rmse": overall_rmse,
        "product_results": results_df,
        "forecasts": all_forecasts,
        "actuals": all_actuals
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    results = run_arima_forecasting(n_products=n)
    print(f"\n✓ ARIMA forecasting complete")
    print(f"  Overall MAPE: {results['overall_mape']:.2f}%")
    print(f"  Overall RMSE: {results['overall_rmse']:.1f} units")
