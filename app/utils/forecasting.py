import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from app.config import ALL_PRODUCTS, PRETTY_NAMES, PROPHET_PARAMS, LGBM_PARAMS
from app.utils.data_loader import create_features
from app.utils.model_registry import load_lgbm, load_prophet_forecast, load_artifact


def predict_daily_lgbm(product: str, horizon_days: int = 30) -> pd.DataFrame:
    """
    Generate daily forecast using pre-trained LightGBM model.
    Uses stored feature column list for alignment.
    Returns DataFrame with columns: ds, yhat, yhat_lower, yhat_upper.
    """
    import lightgbm as lgb
    from app.utils.data_loader import load_enriched_data

    bst = load_lgbm(product)
    feature_cols_map = load_artifact("feature_cols") or {}
    feature_cols = feature_cols_map.get(product, [])

    if bst is None or not feature_cols:
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper"])

    df = load_enriched_data()
    df_feat = create_features(df, product).dropna(subset=feature_cols)

    last_date = df["Date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    # Build future feature rows by extending last known features
    last_row = df_feat[feature_cols].iloc[-1].values
    rows = []
    for fd in future_dates:
        row = _build_future_row(fd, df[product].values, feature_cols)
        rows.append(row)

    X_future = np.array(rows)
    preds = bst.predict(X_future).clip(0)

    # Uncertainty: ±1 RMSE estimated from last-30-day errors (rough heuristic)
    hist_feat = df_feat[feature_cols].values
    hist_pred = bst.predict(hist_feat).clip(0)
    hist_actual = df_feat["target"].values
    std_err = np.std(hist_actual - hist_pred) if len(hist_actual) > 10 else preds.std()

    return pd.DataFrame({
        "ds": future_dates,
        "yhat": preds,
        "yhat_lower": (preds - 1.645 * std_err).clip(0),
        "yhat_upper": preds + 1.645 * std_err,
    })


def _build_future_row(date: pd.Timestamp, historical_values: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """Construct a single feature row for a future date."""
    doy = date.dayofyear
    dow = date.dayofweek
    row = {}
    row["year"] = date.year
    row["month"] = date.month
    row["quarter"] = date.quarter
    row["day_of_week"] = dow
    row["day_of_month"] = date.day
    row["day_of_year"] = doy
    row["week_of_year"] = date.isocalendar().week
    row["is_weekend"] = int(dow >= 5)
    row["is_month_start"] = int(date.is_month_start)
    row["is_month_end"] = int(date.is_month_end)
    for k in range(1, 4):
        row[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        row[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)
    row["sin_dow"] = np.sin(2 * np.pi * dow / 7)
    row["cos_dow"] = np.cos(2 * np.pi * dow / 7)

    # Lags from historical
    for lag in [7, 14, 30, 60, 365]:
        idx = len(historical_values) - lag
        row[f"lag_{lag}"] = historical_values[idx] if idx >= 0 else np.nan

    # Rolling stats
    for w in [7, 14, 30, 90]:
        window = historical_values[-w:] if len(historical_values) >= w else historical_values
        nonzero = window[window > 0]
        row[f"rolling_mean_{w}"] = np.mean(nonzero) if len(nonzero) > 0 else 0.0
        row[f"rolling_std_{w}"] = np.std(nonzero) if len(nonzero) > 1 else 0.0

    return np.array([row.get(fc, 0.0) for fc in feature_cols])


def get_prophet_forecast(product: str, freq: str = "D") -> Optional[pd.DataFrame]:
    """
    Load pre-computed Prophet forecast and optionally resample.
    freq: 'D'=daily, 'W'=weekly, 'ME'=monthly, 'QE'=quarterly, 'YE'=annual
    """
    fc = load_prophet_forecast(product)
    if fc is None:
        return None
    fc["yhat"] = fc["yhat"].clip(lower=0)
    fc["yhat_lower"] = fc["yhat_lower"].clip(lower=0)
    fc["yhat_upper"] = fc["yhat_upper"].clip(lower=0)
    if freq == "D":
        return fc
    agg = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].set_index("ds").resample(freq).sum()
    return agg.reset_index()


def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    a, p = np.array(actual)[mask], np.array(predicted)[mask].clip(0)
    if len(a) == 0:
        return {k: None for k in ["MAPE", "WAPE", "MAE", "RMSE", "R2"]}
    nonzero = a > 1e-3
    mape = float(np.mean(np.abs((a[nonzero] - p[nonzero]) / a[nonzero])) * 100) if nonzero.any() else None
    wape = float(np.sum(np.abs(a - p)) / np.sum(a) * 100) if np.sum(a) > 0 else None
    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    r2 = float(r2_score(a, p)) if len(a) > 1 else None
    return {"MAPE": round(mape, 2) if mape else None,
            "WAPE": round(wape, 2) if wape else None,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3) if r2 else None}