import pandas as pd
import numpy as np
from functools import lru_cache
from app.config import DATA_PATH, ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES


@lru_cache(maxsize=1)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_enriched_data() -> pd.DataFrame:
    df = load_raw_data().copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_year"] = df["Date"].dt.dayofyear
    df["month_name"] = df["Date"].dt.strftime("%b")
    df["total_fruit_kg"] = df[FRUITS].sum(axis=1)
    df["total_veg_kg"] = df[VEGETABLES].sum(axis=1)
    df["total_sales_kg"] = df[ALL_PRODUCTS].sum(axis=1)
    df["rolling_7d"] = df["total_sales_kg"].rolling(7, min_periods=1).mean()
    df["rolling_30d"] = df["total_sales_kg"].rolling(30, min_periods=1).mean()
    return df


def get_date_range_data(start_date, end_date) -> pd.DataFrame:
    df = load_enriched_data()
    mask = (df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))
    return df[mask].copy()


def create_features(df_input: pd.DataFrame, product_col: str) -> pd.DataFrame:
    from app.config import LAG_FEATURES, ROLLING_WINDOWS
    df_feat = df_input[["Date", product_col]].copy()
    df_feat = df_feat.sort_values("Date").reset_index(drop=True)
    df_feat.columns = ["Date", "target"]
    clean = df_feat["target"].replace(0, np.nan)

    df_feat["year"] = df_feat["Date"].dt.year
    df_feat["month"] = df_feat["Date"].dt.month
    df_feat["quarter"] = df_feat["Date"].dt.quarter
    df_feat["day_of_week"] = df_feat["Date"].dt.dayofweek
    df_feat["day_of_month"] = df_feat["Date"].dt.day
    df_feat["day_of_year"] = df_feat["Date"].dt.dayofyear
    df_feat["week_of_year"] = df_feat["Date"].dt.isocalendar().week.astype(int)
    df_feat["is_weekend"] = (df_feat["day_of_week"] >= 5).astype(int)
    df_feat["is_month_start"] = df_feat["Date"].dt.is_month_start.astype(int)
    df_feat["is_month_end"] = df_feat["Date"].dt.is_month_end.astype(int)

    for k in range(1, 4):
        df_feat[f"sin_doy_{k}"] = np.sin(2 * np.pi * k * df_feat["day_of_year"] / 365.25)
        df_feat[f"cos_doy_{k}"] = np.cos(2 * np.pi * k * df_feat["day_of_year"] / 365.25)
    df_feat["sin_dow"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["cos_dow"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)

    for lag in LAG_FEATURES:
        df_feat[f"lag_{lag}"] = clean.shift(lag)

    for w in ROLLING_WINDOWS:
        df_feat[f"rolling_mean_{w}"] = clean.shift(1).rolling(w, min_periods=max(1, w // 2)).mean()
        df_feat[f"rolling_std_{w}"] = clean.shift(1).rolling(w, min_periods=max(1, w // 2)).std()

    df_feat["target"] = df_input[product_col].values
    return df_feat


def compute_seasonal_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Returns seasonal index (monthly mean / annual mean * 100) for all products."""
    month_means = df.groupby("month")[ALL_PRODUCTS].mean()
    annual_mean = df[ALL_PRODUCTS].mean()
    si = (month_means / annual_mean.replace(0, np.nan) * 100).fillna(100)
    si.columns = [PRETTY_NAMES[c] for c in si.columns]
    return si