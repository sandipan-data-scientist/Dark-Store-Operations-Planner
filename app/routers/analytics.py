from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from typing import Optional, List

from app.schemas import AnalyticsRequest
from app.config import ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES
from app.utils.data_loader import load_enriched_data, get_date_range_data, compute_seasonal_indices

router = APIRouter()


@router.post("/summary")
def sales_summary(req: AnalyticsRequest):
    df = get_date_range_data(req.start_date, req.end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data found for the specified date range.")

    products = req.products or ALL_PRODUCTS
    products = [p for p in products if p in ALL_PRODUCTS]

    result = {}
    for p in products:
        series = df[p]
        nonzero = series[series > 0]
        result[PRETTY_NAMES[p]] = {
            "total_kg": round(float(series.sum()), 1),
            "daily_avg_kg": round(float(nonzero.mean()) if len(nonzero) > 0 else 0.0, 1),
            "daily_std_kg": round(float(nonzero.std()) if len(nonzero) > 1 else 0.0, 1),
            "max_kg": round(float(series.max()), 1),
            "min_kg": round(float(series[series > 0].min()) if (series > 0).any() else 0.0, 1),
            "zero_days": int((series == 0).sum()),
            "category": "Fruit" if p in FRUITS else "Vegetable",
        }
    return result


@router.post("/timeseries")
def sales_timeseries(req: AnalyticsRequest):
    df = get_date_range_data(req.start_date, req.end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data in range.")

    products = req.products or ALL_PRODUCTS
    products = [p for p in products if p in ALL_PRODUCTS]

    df_sub = df[["Date"] + products].copy()
    if req.granularity != "D":
        df_sub = df_sub.set_index("Date").resample(req.granularity).sum().reset_index()

    result = {
        "dates": [str(d)[:10] for d in df_sub["Date"]],
        "series": {}
    }
    for p in products:
        result["series"][PRETTY_NAMES[p]] = [round(float(v), 1) for v in df_sub[p].values]
    return result


@router.get("/seasonal-indices")
def seasonal_indices():
    df = load_enriched_data()
    si = compute_seasonal_indices(df)
    return {
        "months": [f"Month {i}" for i in si.index],
        "products": list(si.columns),
        "data": si.round(1).to_dict()
    }


@router.get("/top-products")
def top_products(n: int = 10):
    df = load_enriched_data()
    means = df[ALL_PRODUCTS].mean().sort_values(ascending=False).head(n)
    return [{"product": k, "label": PRETTY_NAMES[k], "avg_daily_kg": round(float(v), 1)}
            for k, v in means.items()]