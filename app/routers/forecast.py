from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd

from app.schemas import ForecastRequest, ForecastResponse, ForecastPoint
from app.config import PRETTY_NAMES, ALL_PRODUCTS
from app.utils.forecasting import predict_daily_lgbm, get_prophet_forecast

router = APIRouter()


@router.post("/", response_model=ForecastResponse)
def get_forecast(req: ForecastRequest):
    if req.product not in ALL_PRODUCTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown product '{req.product}'. Valid: {ALL_PRODUCTS}"
        )
    if req.model_type == "lgbm":
        fc = predict_daily_lgbm(req.product, horizon_days=req.horizon_days)
        if fc.empty:
            raise HTTPException(status_code=503, detail="LightGBM model not loaded. Run train_and_pickle.py.")
        if req.freq != "D":
            fc = fc.set_index("ds").resample(req.freq).sum().reset_index()
    else:
        fc = get_prophet_forecast(req.product, freq=req.freq)
        if fc is None:
            raise HTTPException(status_code=503, detail="Prophet forecast not found. Run train_and_pickle.py.")
        fc = fc[fc["ds"] > pd.Timestamp.today()].head(req.horizon_days)

    points = [
        ForecastPoint(
            ds=str(row["ds"])[:10],
            yhat=round(float(row["yhat"]), 2),
            yhat_lower=round(float(row.get("yhat_lower", row["yhat"] * 0.8)), 2),
            yhat_upper=round(float(row.get("yhat_upper", row["yhat"] * 1.2)), 2),
        )
        for _, row in fc.iterrows()
    ]
    return ForecastResponse(
        product=req.product,
        product_label=PRETTY_NAMES.get(req.product, req.product),
        model_type=req.model_type,
        freq=req.freq,
        forecast=points,
    )


@router.get("/products", response_model=List[str])
def list_products():
    return ALL_PRODUCTS


@router.get("/products/labels")
def list_product_labels():
    return PRETTY_NAMES