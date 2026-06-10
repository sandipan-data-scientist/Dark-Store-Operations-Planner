"""
Dark Store Forecast — Model Training Script
Run this locally before deploying to HuggingFace Spaces.
Output: models/ directory with LightGBM .txt files + Prophet forecast .pkl files.

Usage:
    python scripts/train_and_pickle.py            # All 32 products
    python scripts/train_and_pickle.py --fast     # Top 5 products only (quick demo)
"""

import sys, os, logging, argparse, warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from app.config import (
    DATA_PATH, MODELS_DIR, ALL_PRODUCTS, FRUITS, VEGETABLES,
    PRETTY_NAMES, PROPHET_PARAMS, LGBM_PARAMS, FORECAST_HORIZON_DAYS,
    LAG_FEATURES, ROLLING_WINDOWS,
)
from app.utils.data_loader import load_enriched_data, create_features, compute_seasonal_indices
from app.utils.model_registry import (
    save_lgbm, save_prophet_forecast, save_artifact,
)


def train_lgbm(df: pd.DataFrame, product: str) -> tuple:
    """Train LightGBM model; return (model, feature_cols, metrics_dict)."""
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df_feat = create_features(df, product)
    feature_cols = [c for c in df_feat.columns if c not in ["Date", "target"]]
    df_clean = df_feat.dropna(subset=feature_cols)

    # Walk-forward split: last 180 days as test
    n = len(df_clean)
    split = max(365, n - 180)
    train = df_clean.iloc[:split]
    test = df_clean.iloc[split:]

    X_train = train[feature_cols].values
    y_train = train["target"].values
    X_test = test[feature_cols].values
    y_test = test["target"].values

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict(X_test).clip(0)
    nonzero = y_test > 1e-3
    mape = float(np.mean(np.abs((y_test[nonzero] - preds[nonzero]) / y_test[nonzero])) * 100) if nonzero.any() else None
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))

    metrics = {"MAPE": round(mape, 2) if mape else None, "MAE": round(mae, 2), "RMSE": round(rmse, 2)}
    return model, feature_cols, metrics


def train_prophet_and_forecast(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Train Prophet and return forecast DataFrame for next FORECAST_HORIZON_DAYS days."""
    from prophet import Prophet

    df_p = df[["Date", product]].rename(columns={"Date": "ds", product: "y"})
    df_p["y"] = df_p["y"].replace(0, np.nan)

    m = Prophet(**PROPHET_PARAMS)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=FORECAST_HORIZON_DAYS, freq="D")
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def compute_wfv_lgbm(df: pd.DataFrame, product: str, feature_cols: list) -> dict:
    """Quick walk-forward validation metrics on last 90 days."""
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    df_feat = create_features(df, product).dropna(subset=feature_cols)
    n = len(df_feat)
    if n < 500:
        return {}
    train = df_feat.iloc[:n - 90]
    test = df_feat.iloc[n - 90:]
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(train[feature_cols].values, train["target"].values)
    preds = model.predict(test[feature_cols].values).clip(0)
    actual = test["target"].values
    nonzero = actual > 1e-3
    mape = float(np.mean(np.abs((actual[nonzero] - preds[nonzero]) / actual[nonzero])) * 100) if nonzero.any() else None
    return {
        "MAPE": round(mape, 2) if mape else None,
        "MAE": round(float(mean_absolute_error(actual, preds)), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(actual, preds))), 2),
        "horizon": "90-day WFV",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Train only top 5 products")
    args = parser.parse_args()

    logger.info(f"=== Training Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info(f"Models dir: {MODELS_DIR}")

    if not DATA_PATH.exists():
        logger.error(f"Data file not found at {DATA_PATH}. Aborting.")
        sys.exit(1)

    df = load_enriched_data()
    logger.info(f"Loaded data: {df.shape[0]} rows, {df['Date'].min().date()} to {df['Date'].max().date()}")

    products_to_train = (
        ["potato_kg", "onion_kg", "tomato_kg", "banana_kg", "apple_kg"]
        if args.fast else ALL_PRODUCTS
    )
    logger.info(f"Products to train: {len(products_to_train)}")

    # --- Seasonal indices ---
    logger.info("Computing seasonal indices...")
    si = compute_seasonal_indices(df)
    save_artifact("seasonal_data", {
        "seasonal_index": si,
        "monthly_means": df.groupby("month")[ALL_PRODUCTS].mean(),
        "yearly_means": df[df["year"] < 2025].groupby("year")[ALL_PRODUCTS].mean(),
        "dow_means": df.groupby("day_of_week")[ALL_PRODUCTS].mean(),
    })
    logger.info("Seasonal data saved.")

    # --- LightGBM models ---
    feature_cols_map = {}
    wfv_metrics = {}

    logger.info("=== Training LightGBM models ===")
    for i, product in enumerate(products_to_train, 1):
        logger.info(f"[{i:02d}/{len(products_to_train)}] LightGBM: {PRETTY_NAMES[product]}")
        try:
            model, feature_cols, metrics = train_lgbm(df, product)
            save_lgbm(product, model)
            feature_cols_map[product] = feature_cols
            wfv_metrics[product] = {"lgbm": metrics}
            logger.info(f"    MAPE={metrics.get('MAPE')}%  MAE={metrics['MAE']}  RMSE={metrics['RMSE']}")
        except Exception as e:
            logger.error(f"    FAILED: {e}")

    save_artifact("feature_cols", feature_cols_map)
    logger.info("Feature columns map saved.")

    # --- Prophet forecasts ---
    logger.info("=== Training Prophet models and saving forecasts ===")
    for i, product in enumerate(products_to_train, 1):
        logger.info(f"[{i:02d}/{len(products_to_train)}] Prophet: {PRETTY_NAMES[product]}")
        try:
            fc = train_prophet_and_forecast(df, product)
            save_prophet_forecast(product, fc)
            logger.info(f"    Forecast saved: {len(fc)} rows through {fc['ds'].max().date()}")
        except Exception as e:
            logger.error(f"    FAILED: {e}")

    # --- Walk-forward validation summary ---
    logger.info("=== Computing WFV metrics ===")
    for product in products_to_train:
        if product in feature_cols_map:
            try:
                wfv = compute_wfv_lgbm(df, product, feature_cols_map[product])
                if product not in wfv_metrics:
                    wfv_metrics[product] = {}
                wfv_metrics[product]["wfv_90d"] = wfv
            except Exception as e:
                logger.warning(f"WFV failed for {PRETTY_NAMES[product]}: {e}")

    save_artifact("wfv_metrics", wfv_metrics)
    logger.info("WFV metrics saved.")

    # --- Save zero-analysis ---
    zero_df = pd.DataFrame({
        "zero_days": (df[ALL_PRODUCTS] == 0).sum(),
        "zero_pct": ((df[ALL_PRODUCTS] == 0).sum() / len(df) * 100).round(2),
        "category": ["Fruit" if p in FRUITS else "Vegetable" for p in ALL_PRODUCTS],
        "mean_nonzero": [df.loc[df[p] > 0, p].mean() for p in ALL_PRODUCTS],
    })
    zero_df.index = [PRETTY_NAMES[p] for p in ALL_PRODUCTS]
    save_artifact("zero_analysis", zero_df)

    # --- Descriptive stats ---
    desc = df[ALL_PRODUCTS].describe().T
    desc.index = [PRETTY_NAMES[p] for p in desc.index]
    desc["cv"] = (desc["std"] / desc["mean"] * 100).round(1)
    desc["category"] = ["Fruit" if p in FRUITS else "Vegetable" for p in ALL_PRODUCTS]
    save_artifact("descriptive_stats", desc)

    logger.info(f"=== Training Complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logger.info(f"All artifacts saved to: {MODELS_DIR}")
    logger.info(f"Products trained: {len(products_to_train)}")
    logger.info("Next: commit the models/ directory to your HuggingFace repo.")


if __name__ == "__main__":
    main()