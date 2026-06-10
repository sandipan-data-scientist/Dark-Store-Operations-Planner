from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from app.schemas import RestockRequest
from app.config import ALL_PRODUCTS, PRETTY_NAMES
from app.utils.forecasting import predict_daily_lgbm, get_prophet_forecast
from app.utils.supply_chain import (
    compute_total_opex, build_restock_plan, compute_risk_metrics, vendor_comparison
)

router = APIRouter()


@router.post("/restock-plan")
def get_restock_plan(req: RestockRequest):
    all_fc_rows = []
    for product in req.products:
        if product not in ALL_PRODUCTS:
            continue
        fc = predict_daily_lgbm(product, horizon_days=req.horizon_days * 7)
        if fc.empty:
            fc = get_prophet_forecast(product, freq="D")
        if fc is not None and not fc.empty:
            fc = fc.head(req.horizon_days * 7)
            fc["product"] = product
            all_fc_rows.append(fc[["ds", "product", "yhat"]])

    if not all_fc_rows:
        raise HTTPException(status_code=503, detail="No forecast data available. Train models first.")

    forecast_df = pd.concat(all_fc_rows, ignore_index=True)
    restock_df = build_restock_plan(
        forecast_df,
        req.shelf_life_days,
        req.safety_stock_multiplier,
        req.lead_time_days,
    )
    opex = compute_total_opex(
        req.liability_costs.transport,
        req.liability_costs.last_mile,
        req.liability_costs.manpower,
        req.liability_costs.storage_electricity,
        req.liability_costs.rent_licences,
        req.liability_costs.miscellaneous,
    )
    risk = compute_risk_metrics(
        restock_df,
        req.vendor_cost_per_kg,
        req.selling_price_per_kg,
        opex["total_weekly_opex"],
    )
    return {
        "restock_plan": restock_df.to_dict(orient="records"),
        "opex": opex,
        "risk_metrics": risk,
    }