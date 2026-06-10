import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from app.config import ALL_PRODUCTS, PRETTY_NAMES, DEFAULT_SHELF_LIFE


def compute_total_opex(
    transport: float,
    last_mile: float,
    manpower: float,
    storage_electricity: float,
    rent_licences: float,
    miscellaneous: float,
) -> Dict:
    items = {
        "Transport": transport,
        "Last Mile Ops": last_mile,
        "Manpower": manpower,
        "Storage & Electricity": storage_electricity,
        "Rent & Licences": rent_licences,
        "Miscellaneous": miscellaneous,
    }
    total = sum(items.values())
    breakdown = {k: {"amount": v, "pct": round(v / total * 100, 1) if total > 0 else 0}
                 for k, v in items.items()}
    return {"total_weekly_opex": total, "breakdown": breakdown}


def build_restock_plan(
    forecast_df: pd.DataFrame,
    shelf_life_days: Dict[str, int],
    safety_stock_multiplier: float = 1.2,
    lead_time_days: int = 1,
) -> pd.DataFrame:
    """
    Given a daily forecast DataFrame with columns [ds, product, yhat],
    produce a weekly restocking plan with order dates, volumes, and risk.
    """
    if forecast_df.empty:
        return pd.DataFrame()

    forecast_df = forecast_df.copy()
    forecast_df["week"] = pd.to_datetime(forecast_df["ds"]).dt.to_period("W")
    weekly = forecast_df.groupby(["week", "product"]).agg(
        forecast_weekly=("yhat", "sum"),
        forecast_daily_avg=("yhat", "mean"),
        forecast_std=("yhat", "std"),
    ).reset_index()

    records = []
    for _, row in weekly.iterrows():
        product = row["product"]
        shelf_life = shelf_life_days.get(product, DEFAULT_SHELF_LIFE.get(product, 7))
        daily_avg = row["forecast_daily_avg"]
        weekly_fc = row["forecast_weekly"]
        std = row["forecast_std"] if not np.isnan(row["forecast_std"]) else daily_avg * 0.1

        # Volume to order: cover demand for min(shelf_life, 7) days with safety buffer
        order_days = min(shelf_life, 7)
        base_order = daily_avg * order_days * safety_stock_multiplier
        degradation_loss_pct = max(0, (order_days / shelf_life - 0.5) * 100) if shelf_life > 0 else 0

        # Deviation risk: probability that actual exceeds forecast (normal approx)
        from scipy.stats import norm
        service_level = 0.95
        z = norm.ppf(service_level)
        safety_stock_kg = z * std * np.sqrt(order_days)
        reorder_qty = base_order + safety_stock_kg

        # Shortfall risk at 1 std
        stockout_risk_pct = round((1 - norm.cdf(reorder_qty, loc=weekly_fc, scale=std * 3)) * 100, 1)

        records.append({
            "week": str(row["week"]),
            "product": product,
            "product_label": PRETTY_NAMES.get(product, product),
            "shelf_life_days": shelf_life,
            "forecast_weekly_kg": round(weekly_fc, 1),
            "recommended_order_kg": round(reorder_qty, 1),
            "safety_stock_kg": round(safety_stock_kg, 1),
            "degradation_loss_pct": round(degradation_loss_pct, 1),
            "stockout_risk_pct": stockout_risk_pct,
            "daily_avg_kg": round(daily_avg, 1),
        })
    return pd.DataFrame(records)


def compute_risk_metrics(
    restock_df: pd.DataFrame,
    vendor_cost_per_kg: Dict[str, float],
    selling_price_per_kg: Dict[str, float],
    weekly_opex: float,
) -> Dict:
    """
    Computes financial risk metrics from restock plan + pricing inputs.
    """
    if restock_df.empty:
        return {}

    total_procurement_cost = sum(
        row["recommended_order_kg"] * vendor_cost_per_kg.get(row["product"], 0)
        for _, row in restock_df.iterrows()
    )
    total_forecasted_revenue = sum(
        row["forecast_weekly_kg"] * selling_price_per_kg.get(row["product"], 0)
        for _, row in restock_df.iterrows()
    )
    degradation_loss_value = sum(
        row["recommended_order_kg"]
        * (row["degradation_loss_pct"] / 100)
        * vendor_cost_per_kg.get(row["product"], 0)
        for _, row in restock_df.iterrows()
    )
    gross_profit = total_forecasted_revenue - total_procurement_cost
    net_profit = gross_profit - weekly_opex - degradation_loss_value
    margin_pct = (net_profit / total_forecasted_revenue * 100) if total_forecasted_revenue > 0 else 0

    # Amount at risk: value of stock that might not sell (stockout risk inverted + degradation)
    amount_at_risk = degradation_loss_value + sum(
        row["recommended_order_kg"]
        * (row["stockout_risk_pct"] / 100)
        * vendor_cost_per_kg.get(row["product"], 0)
        for _, row in restock_df.iterrows()
    )

    high_risk_products = restock_df[restock_df["stockout_risk_pct"] > 15]["product_label"].tolist()

    return {
        "total_procurement_cost": round(total_procurement_cost, 2),
        "total_forecasted_revenue": round(total_forecasted_revenue, 2),
        "degradation_loss_value": round(degradation_loss_value, 2),
        "weekly_opex": round(weekly_opex, 2),
        "gross_profit": round(gross_profit, 2),
        "net_profit": round(net_profit, 2),
        "net_margin_pct": round(margin_pct, 1),
        "amount_at_risk": round(amount_at_risk, 2),
        "high_risk_products": high_risk_products,
    }


def vendor_comparison(
    products: List[str],
    vendor_data: Dict[str, Dict[str, Optional[float]]],
    forecast_weekly_kg: Dict[str, float],
) -> pd.DataFrame:
    """
    Compare vendor pricing across products.
    vendor_data = {vendor_name: {product: price_per_kg or None}}
    """
    records = []
    for product in products:
        label = PRETTY_NAMES.get(product, product)
        row = {"product": product, "product_label": label,
               "weekly_demand_kg": forecast_weekly_kg.get(product, 0)}
        prices = {}
        for vendor, price_map in vendor_data.items():
            price = price_map.get(product)
            row[f"{vendor}_price"] = price
            if price is not None:
                prices[vendor] = price

        if prices:
            best_vendor = min(prices, key=prices.get)
            worst_vendor = max(prices, key=prices.get)
            row["cheapest_vendor"] = best_vendor
            row["cheapest_price"] = prices[best_vendor]
            row["saving_vs_costliest"] = round(prices[worst_vendor] - prices[best_vendor], 2)
        else:
            row["cheapest_vendor"] = None
            row["cheapest_price"] = None
            row["saving_vs_costliest"] = 0.0
        records.append(row)
    return pd.DataFrame(records)