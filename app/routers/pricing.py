from fastapi import APIRouter
import numpy as np
import pandas as pd

from app.schemas import PricingAnalysisRequest
from app.config import PRETTY_NAMES

router = APIRouter()


@router.post("/analysis")
def pricing_analysis(req: PricingAnalysisRequest):
    records = []
    for product in req.products:
        label = PRETTY_NAMES.get(product, product)
        your_price = req.your_price.get(product, 0.0)
        comp_price = req.competitor_prices.get(product, 0.0)
        cost = req.vendor_cost.get(product, 0.0)
        unit = req.unit_type.get(product, "kg")
        unit_weight_g = req.unit_weight_grams.get(product, 1000.0)
        monthly_kg = req.forecast_monthly_kg.get(product, 0.0)

        # Convert unit price to per-kg equivalent for margin calculation
        if unit == "grams":
            price_per_kg = your_price * 1000
            comp_price_per_kg = comp_price * 1000
        elif unit == "units":
            price_per_kg = your_price * (1000 / unit_weight_g) if unit_weight_g > 0 else 0
            comp_price_per_kg = comp_price * (1000 / unit_weight_g) if unit_weight_g > 0 else 0
        else:
            price_per_kg = your_price
            comp_price_per_kg = comp_price

        margin_per_kg = price_per_kg - cost
        margin_pct = (margin_per_kg / price_per_kg * 100) if price_per_kg > 0 else 0
        relative_vs_competitor = ((your_price - comp_price) / comp_price * 100) if comp_price > 0 else 0

        monthly_revenue = monthly_kg * price_per_kg
        monthly_cost = monthly_kg * cost
        monthly_profit = monthly_revenue - monthly_cost

        records.append({
            "product": product,
            "product_label": label,
            "unit_type": unit,
            "your_price": your_price,
            "competitor_price": comp_price,
            "vendor_cost_per_kg": cost,
            "your_price_per_kg_equiv": round(price_per_kg, 2),
            "comp_price_per_kg_equiv": round(comp_price_per_kg, 2),
            "margin_per_kg": round(margin_per_kg, 2),
            "margin_pct": round(margin_pct, 1),
            "vs_competitor_pct": round(relative_vs_competitor, 1),
            "position": "above" if relative_vs_competitor > 0 else "below" if relative_vs_competitor < 0 else "same",
            "monthly_forecast_kg": round(monthly_kg, 1),
            "monthly_revenue_est": round(monthly_revenue, 2),
            "monthly_profit_est": round(monthly_profit, 2),
        })

    total_monthly_revenue = sum(r["monthly_revenue_est"] for r in records)
    total_monthly_profit = sum(r["monthly_profit_est"] for r in records)
    avg_margin = np.mean([r["margin_pct"] for r in records]) if records else 0

    return {
        "products": records,
        "totals": {
            "monthly_revenue": round(total_monthly_revenue, 2),
            "monthly_profit": round(total_monthly_profit, 2),
            "avg_margin_pct": round(float(avg_margin), 1),
        }
    }