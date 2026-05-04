"""
Random Forest Model Module — Demand Forecasting Pipeline
==========================================================
Trains a Random Forest regressor on engineered features to predict
daily demand across 200+ product lines.

The Random Forest component excels at:
- Capturing non-linear feature interactions
- Handling promotional effects and price sensitivity
- Robust to outliers and missing values
- Feature importance for business interpretability

Model configuration:
- 500 estimators with max_depth=20
- Min samples split: 10, min samples leaf: 5
- Out-of-bag scoring enabled for validation

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


# ─── Model Parameters ─────────────────────────────────────────────────────────

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 20,
    "min_samples_split": 10,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "oob_score": True,
    "n_jobs": -1,
    "random_state": 42,
    "verbose": 0
}

# Features to exclude from model input
EXCLUDE_COLS = [
    "product_id", "date", "units_sold", "revenue", "target",
    "year", "week", "demand_change"
]


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    mask = actual != 0
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """
    Calculate comprehensive regression metrics.

    Parameters
    ----------
    actual : np.ndarray
        True values.
    predicted : np.ndarray
        Model predictions.

    Returns
    -------
    dict
        Dictionary of metric name -> value.
    """
    mape = calculate_mape(actual, predicted)
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    # R-squared
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Median Absolute Percentage Error (more robust)
    mask = actual != 0
    mdape = np.median(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    return {
        "mape": mape,
        "mdape": mdape,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target from processed DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Processed DataFrame with features and target.

    Returns
    -------
    tuple
        (X, y, feature_names) where X is feature matrix, y is target.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Keep only numeric columns
    numeric_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64, np.float32, np.int32, np.uint8]]

    X = df[numeric_cols].values
    y = df["target"].values if "target" in df.columns else df["units_sold"].values

    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, numeric_cols


def generate_synthetic_features(n_samples: int = 50000) -> tuple:
    """
    Generate synthetic feature data for demonstration when
    feature engineering output is not available.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate.

    Returns
    -------
    tuple
        (X, y, feature_names)
    """
    np.random.seed(42)

    feature_names = [
        "demand_lag_7d", "demand_lag_14d", "demand_lag_30d",
        "demand_lag_60d", "demand_lag_90d",
        "rolling_mean_7d", "rolling_mean_14d", "rolling_mean_30d",
        "rolling_std_7d", "rolling_std_14d", "rolling_std_30d",
        "rolling_min_7d", "rolling_max_7d",
        "rolling_min_30d", "rolling_max_30d",
        "demand_cv_30d",
        "day_of_week", "day_of_month", "month", "quarter",
        "week_of_year", "is_weekend", "is_month_start", "is_month_end",
        "days_to_holiday", "is_holiday_week",
        "price", "price_change_pct", "price_vs_avg_30d",
        "is_promo", "days_since_promo",
        "price_demand_corr_30d",
        "demand_yoy", "trend_30d",
        "cat_Electronics", "cat_Apparel", "cat_Home_Garden",
        "cat_Grocery", "cat_Health_Beauty",
        "wh_East", "wh_West", "wh_Central",
        "reg_Northeast", "reg_Southeast", "reg_Midwest"
    ]

    n_features = len(feature_names)
    X = np.random.randn(n_samples, n_features)

    # Make lag features dominant predictors (realistic)
    base_demand = np.random.uniform(20, 400, n_samples)
    X[:, 0] = base_demand + np.random.normal(0, 20, n_samples)  # lag_7d
    X[:, 1] = base_demand + np.random.normal(0, 30, n_samples)  # lag_14d
    X[:, 2] = base_demand + np.random.normal(0, 40, n_samples)  # lag_30d
    X[:, 5] = base_demand + np.random.normal(0, 15, n_samples)  # rolling_mean_7d
    X[:, 6] = base_demand + np.random.normal(0, 20, n_samples)  # rolling_mean_14d
    X[:, 7] = base_demand + np.random.normal(0, 25, n_samples)  # rolling_mean_30d

    # Calendar features
    X[:, 16] = np.random.randint(0, 7, n_samples)   # day_of_week
    X[:, 17] = np.random.randint(1, 32, n_samples)  # day_of_month
    X[:, 18] = np.random.randint(1, 13, n_samples)  # month
    X[:, 19] = np.random.randint(1, 5, n_samples)   # quarter
    X[:, 21] = np.random.randint(0, 2, n_samples)   # is_weekend
    X[:, 29] = np.random.randint(0, 2, n_samples)   # is_promo

    # Target: primarily driven by lag features + some non-linear effects
    y = (
        0.35 * X[:, 0] +           # lag_7d
        0.20 * X[:, 5] +           # rolling_mean_7d
        0.15 * X[:, 2] +           # lag_30d
        0.10 * X[:, 7] +           # rolling_mean_30d
        5.0 * X[:, 29] * 20 +      # promo effect
        -3.0 * X[:, 21] * 10 +     # weekend effect (category-dependent)
        np.random.normal(0, 15, n_samples)  # noise
    )
    y = np.maximum(y, 0)

    return X, y, feature_names


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> dict:
    """
    Train Random Forest model and evaluate.

    Parameters
    ----------
    X_train, y_train : np.ndarray
        Training data.
    X_test, y_test : np.ndarray
        Test data.
    feature_names : list
        Feature column names.

    Returns
    -------
    dict
        Model results including metrics and feature importances.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        logger.info("Training Random Forest model...")
        logger.info(f"  Parameters: {RF_PARAMS}")
        logger.info(f"  Training samples: {X_train.shape[0]:,}")
        logger.info(f"  Features: {X_train.shape[1]}")

        # Train model
        model = RandomForestRegressor(**RF_PARAMS)
        model.fit(X_train, y_train)

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Metrics
        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)

        # Feature importance
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        # OOB score
        oob_score = model.oob_score_ if RF_PARAMS.get("oob_score") else None

        return {
            "model": model,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "feature_importance": importance_df,
            "oob_score": oob_score,
            "predictions": test_pred,
            "method": "RandomForestRegressor"
        }

    except ImportError:
        logger.error("scikit-learn not installed. Cannot train Random Forest.")
        sys.exit(1)


def run_random_forest(data_dir: str = None) -> dict:
    """
    Main Random Forest training pipeline.

    Parameters
    ----------
    data_dir : str, optional
        Directory containing train/test feature CSVs.

    Returns
    -------
    dict
        Complete model results.
    """
    logger.info("=" * 60)
    logger.info("DEMAND FORECASTING PIPELINE — Random Forest Model")
    logger.info("=" * 60)

    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    train_path = Path(data_dir) / "train_features.csv"
    test_path = Path(data_dir) / "test_features.csv"

    # Load engineered features or generate synthetic
    if train_path.exists() and test_path.exists():
        logger.info("Loading engineered features...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        X_train, y_train, feature_names = prepare_features(train_df)
        X_test, y_test, _ = prepare_features(test_df)
    else:
        logger.info("Engineered features not found. Using synthetic data for demonstration...")
        X, y, feature_names = generate_synthetic_features(n_samples=80000)

        # 80/20 split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info(f"  Train shape: {X_train.shape}")
    logger.info(f"  Test shape:  {X_test.shape}")

    # Train model
    results = train_random_forest(X_train, y_train, X_test, y_test, feature_names)

    # ─── Print Results ────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 60}")
    logger.info("RANDOM FOREST RESULTS")
    logger.info(f"{'=' * 60}")

    test_metrics = results["test_metrics"]
    train_metrics = results["train_metrics"]

    logger.info(f"\n  Training Metrics:")
    logger.info(f"    R²:    {train_metrics['r2']:.4f}")
    logger.info(f"    MAPE:  {train_metrics['mape']:.2f}%")
    logger.info(f"    RMSE:  {train_metrics['rmse']:.2f}")

    logger.info(f"\n  Test Metrics:")
    logger.info(f"    R²:    {test_metrics['r2']:.4f}")
    logger.info(f"    MAPE:  {test_metrics['mape']:.2f}%")
    logger.info(f"    MdAPE: {test_metrics['mdape']:.2f}%")
    logger.info(f"    RMSE:  {test_metrics['rmse']:.2f}")
    logger.info(f"    MAE:   {test_metrics['mae']:.2f}")

    if results["oob_score"]:
        logger.info(f"\n  OOB Score (R²): {results['oob_score']:.4f}")

    # Feature importance
    logger.info(f"\n  Top 10 Features by Importance:")
    logger.info(f"  {'─' * 50}")
    importance_df = results["feature_importance"]
    for i, row in importance_df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 100)
        logger.info(f"    {row['feature']:<30s} {row['importance']:.4f} {bar}")

    # Save predictions for ensemble
    output_dir = Path(data_dir)
    pred_df = pd.DataFrame({
        "actual": y_test,
        "rf_forecast": results["predictions"]
    })
    pred_path = output_dir / "rf_forecasts.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"\n  Predictions saved to {pred_path}")

    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    logger.info(f"  Feature importance saved to {importance_path}")

    return results


if __name__ == "__main__":
    results = run_random_forest()
    test_metrics = results["test_metrics"]
    print(f"\n✓ Random Forest training complete")
    print(f"  Test R²:   {test_metrics['r2']:.4f}")
    print(f"  Test MAPE: {test_metrics['mape']:.2f}%")
    print(f"  Test RMSE: {test_metrics['rmse']:.2f}")
