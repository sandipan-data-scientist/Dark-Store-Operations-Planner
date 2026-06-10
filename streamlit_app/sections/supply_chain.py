"""
Section 3: Supply Chain & Inventory Planning Dashboard
Liability costs → Restock timeline → Risk metrics → Vendor comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES, DEFAULT_SHELF_LIFE
from app.utils.data_loader import load_enriched_data
from app.utils.supply_chain import (
    compute_total_opex, build_restock_plan, compute_risk_metrics, vendor_comparison,
)
from app.utils.forecasting import predict_daily_lgbm, get_prophet_forecast
from app.utils.model_registry import models_ready


@st.cache_data(show_spinner=False)
def get_forecast_for_products(products: tuple, horizon_days: int) -> pd.DataFrame:
    """Load or compute daily forecast for a list of products."""
    all_rows = []
    for product in products:
        fc = predict_daily_lgbm(product, horizon_days=horizon_days)
        if fc.empty:
            fc = get_prophet_forecast(product, freq="D")
        if fc is not None and not fc.empty:
            fc = fc.head(horizon_days).copy()
            fc["product"] = product
            all_rows.append(fc[["ds", "product", "yhat"]])
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def render():
    st.title("🏭 Supply Chain & Inventory Planning")
    st.markdown(
        "Configure your cost structure, expiry timelines, and vendor pricing. "
        "The model generates restocking schedules, risk metrics, and procurement recommendations."
    )

    if not models_ready():
        st.warning(
            "⚠️ Models not found. Run `python scripts/train_and_pickle.py` locally first, "
            "then commit the `models/` directory to your HuggingFace repo. "
            "Forecast features will use historical averages as fallback."
        )

    df = load_enriched_data()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: SELECT PRODUCTS
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 1: Select Products to Plan")
    col_cat, col_prod = st.columns([1, 3])
    with col_cat:
        cat_filter = st.radio("Category", ["All", "Fruits Only", "Vegetables Only"])
    with col_prod:
        if cat_filter == "Fruits Only":
            pool = FRUITS
        elif cat_filter == "Vegetables Only":
            pool = VEGETABLES
        else:
            pool = ALL_PRODUCTS
        selected = st.multiselect(
            "Products to include in plan",
            options=pool, default=pool[:8],
            format_func=lambda x: PRETTY_NAMES[x],
        )

    if not selected:
        st.info("Select at least one product above.")
        return

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: LIABILITY COSTS
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 2: Weekly Liability Costs (₹)")
    with st.expander("💰 Enter your weekly fixed & variable costs", expanded=True):
        cols = st.columns(3)
        transport = cols[0].number_input("🚛 Transport", min_value=0.0, value=15000.0, step=500.0)
        last_mile = cols[1].number_input("🛵 Last Mile Ops", min_value=0.0, value=8000.0, step=500.0)
        manpower = cols[2].number_input("👷 Manpower", min_value=0.0, value=20000.0, step=500.0)
        storage_elec = cols[0].number_input("🏭 Storage & Electricity", min_value=0.0, value=6000.0, step=500.0)
        rent_lic = cols[1].number_input("🏢 Rent & Licences", min_value=0.0, value=12000.0, step=500.0)
        misc = cols[2].number_input("📦 Miscellaneous", min_value=0.0, value=3000.0, step=500.0)

    opex = compute_total_opex(transport, last_mile, manpower, storage_elec, rent_lic, misc)
    total_opex = opex["total_weekly_opex"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Weekly OPEX", f"₹{total_opex:,.0f}")
    c2.metric("Daily OPEX", f"₹{total_opex/7:,.0f}")
    c3.metric("Monthly OPEX (est.)", f"₹{total_opex * 4.33:,.0f}")

    # OPEX breakdown chart
    fig_opex = px.pie(
        values=list(opex["breakdown"][k]["amount"] for k in opex["breakdown"]),
        names=list(opex["breakdown"].keys()),
        title="Weekly OPEX Breakdown", hole=0.4,
    )
    fig_opex.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_opex, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: EXPIRY TIMELINES
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 3: Product Expiry Timelines (Shelf Life in Days)")
    st.caption("Enter the shelf life at your storage temperature. Defaults are set conservatively.")

    shelf_cols = st.columns(4)
    shelf_life = {}
    for i, product in enumerate(selected):
        col = shelf_cols[i % 4]
        shelf_life[product] = col.number_input(
            PRETTY_NAMES[product],
            min_value=1, max_value=90,
            value=DEFAULT_SHELF_LIFE.get(product, 7),
            key=f"shelf_{product}",
        )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 4: VENDOR PRICING
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 4: Vendor Pricing Table (₹/kg)")
    st.caption("Enter vendor cost prices per kg. Leave blank if product is not available with that vendor.")

    # Dynamic vendor addition
    if "n_vendors" not in st.session_state:
        st.session_state["n_vendors"] = 3
    col_v1, col_v2 = st.columns([3, 1])
    with col_v2:
        if st.button("➕ Add Vendor"):
            st.session_state["n_vendors"] = min(st.session_state["n_vendors"] + 1, 8)

    vendor_names = []
    for v in range(st.session_state["n_vendors"]):
        vname = st.text_input(f"Vendor {v+1} name", value=f"Vendor {v+1}", key=f"vname_{v}")
        vendor_names.append(vname)

    # Build vendor price table
    vendor_prices: dict = {vn: {} for vn in vendor_names}
    with st.container():
        header_cols = st.columns([2] + [1] * len(vendor_names))
        header_cols[0].markdown("**Product**")
        for i, vn in enumerate(vendor_names):
            header_cols[i + 1].markdown(f"**{vn}**")

        for product in selected:
            row_cols = st.columns([2] + [1] * len(vendor_names))
            row_cols[0].write(PRETTY_NAMES[product])
            for i, vn in enumerate(vendor_names):
                default_cost = float(df[product].mean() * 0.6)
                price = row_cols[i + 1].number_input(
                    "", min_value=0.0, value=round(default_cost, 1),
                    step=0.5, key=f"vp_{product}_{i}",
                    label_visibility="collapsed",
                    help=f"{PRETTY_NAMES[product]} at {vn}"
                )
                vendor_prices[vn][product] = price if price > 0 else None

    # Best vendor per product
    st.subheader("🏆 Vendor Comparison & Recommendation")
    forecast_weekly_kg = {}
    for product in selected:
        hist_avg = float(df[product].mean())
        forecast_weekly_kg[product] = hist_avg * 7  # fallback to historical average

    vendor_comp_df = vendor_comparison(selected, vendor_prices, forecast_weekly_kg)

    display_cols = ["product_label", "weekly_demand_kg", "cheapest_vendor", "cheapest_price",
                    "saving_vs_costliest"]
    for vn in vendor_names:
        display_cols.insert(3, f"{vn}_price")

    st.dataframe(
        vendor_comp_df[display_cols].rename(columns={
            "product_label": "Product",
            "weekly_demand_kg": "Weekly Demand (kg)",
            "cheapest_vendor": "Best Vendor",
            "cheapest_price": "Best Price (₹/kg)",
            "saving_vs_costliest": "Saving vs Costliest (₹/kg)",
        }).style.background_gradient(subset=["Saving vs Costliest (₹/kg)"], cmap="Greens"),
        use_container_width=True,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 5: SELLING PRICE INPUTS
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 5: Your Selling Prices (₹/kg)")
    sp_cols = st.columns(4)
    selling_prices = {}
    vendor_cost_best = {}
    for i, product in enumerate(selected):
        col = sp_cols[i % 4]
        best_price = vendor_comp_df.loc[vendor_comp_df["product"] == product, "cheapest_price"]
        best_price_val = float(best_price.iloc[0]) if not best_price.empty and best_price.iloc[0] is not None else float(df[product].mean() * 0.6)
        vendor_cost_best[product] = best_price_val
        selling_price = col.number_input(
            f"{PRETTY_NAMES[product]} (₹/kg)",
            min_value=0.0, value=round(best_price_val * 1.35, 1),
            step=0.5, key=f"sp_{product}",
        )
        selling_prices[product] = selling_price

    # ─────────────────────────────────────────────────────────────────────
    # STEP 6: GENERATE RESTOCK PLAN
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 6: Weekly Restocking Plan")
    planning_weeks = st.slider("Weeks to plan ahead", min_value=1, max_value=12, value=4)

    with st.spinner("Computing restock plan from forecasts..."):
        fc_df = get_forecast_for_products(tuple(selected), horizon_days=planning_weeks * 7)
        if fc_df.empty:
            # Fallback: use historical mean as pseudo-forecast
            rows = []
            import pandas as pd as _pd
            from datetime import datetime, timedelta
            today = pd.Timestamp.today()
            for product in selected:
                daily_mean = float(df[product].mean())
                for d in range(planning_weeks * 7):
                    rows.append({"ds": today + pd.Timedelta(days=d),
                                 "product": product, "yhat": daily_mean})
            fc_df = pd.DataFrame(rows)

        restock_df = build_restock_plan(fc_df, shelf_life,
                                        safety_stock_multiplier=1.2, lead_time_days=1)
        risk_metrics = compute_risk_metrics(
            restock_df[restock_df["week"] == restock_df["week"].unique()[0]],
            vendor_cost_best, selling_prices, total_opex
        )

    if not restock_df.empty:
        # Summary by week
        st.subheader("📅 Weekly Restock Timeline")
        weekly_summary = restock_df.groupby("week").agg(
            total_order_kg=("recommended_order_kg", "sum"),
            avg_stockout_risk=("stockout_risk_pct", "mean"),
            avg_degradation=("degradation_loss_pct", "mean"),
        ).reset_index()
        weekly_summary["procurement_cost"] = weekly_summary["total_order_kg"] * (
            sum(vendor_cost_best.values()) / max(len(vendor_cost_best), 1)
        )

        fig_wk = make_subplots(rows=2, cols=1, subplot_titles=[
            "Total Weekly Order Volume (kg) by Product",
            "Stockout Risk & Degradation Loss (%)"
        ], vertical_spacing=0.15)

        palette = px.colors.qualitative.Prism
        for i, product in enumerate(selected):
            sub = restock_df[restock_df["product"] == product]
            fig_wk.add_trace(go.Bar(
                x=sub["week"], y=sub["recommended_order_kg"],
                name=PRETTY_NAMES[product],
                marker_color=palette[i % len(palette)],
            ), row=1, col=1)

        fig_wk.add_trace(go.Scatter(
            x=weekly_summary["week"], y=weekly_summary["avg_stockout_risk"],
            mode="lines+markers", name="Avg Stockout Risk (%)",
            line=dict(color="#E74C3C", width=2),
        ), row=2, col=1)
        fig_wk.add_trace(go.Scatter(
            x=weekly_summary["week"], y=weekly_summary["avg_degradation"],
            mode="lines+markers", name="Avg Degradation Loss (%)",
            line=dict(color="#F39C12", width=2),
        ), row=2, col=1)

        fig_wk.update_layout(height=560, barmode="stack", legend=dict(orientation="h", y=-0.18))
        fig_wk.update_xaxes(tickangle=30)
        st.plotly_chart(fig_wk, use_container_width=True)

        # Detailed table
        st.subheader("📋 Detailed Restock Plan (First Week)")
        first_week = restock_df["week"].unique()[0]
        week1 = restock_df[restock_df["week"] == first_week][
            ["product_label", "shelf_life_days", "forecast_weekly_kg",
             "recommended_order_kg", "safety_stock_kg",
             "degradation_loss_pct", "stockout_risk_pct"]
        ].copy()
        week1["procurement_cost"] = week1.apply(
            lambda row: round(
                row["recommended_order_kg"] * vendor_cost_best.get(
                    PRETTY_NAMES.get(row["product_label"], ""), 0
                ), 2
            ), axis=1
        )
        st.dataframe(
            week1.rename(columns={
                "product_label": "Product", "shelf_life_days": "Shelf Life (days)",
                "forecast_weekly_kg": "Forecast (kg)", "recommended_order_kg": "Order Qty (kg)",
                "safety_stock_kg": "Safety Stock (kg)", "degradation_loss_pct": "Degradation Risk (%)",
                "stockout_risk_pct": "Stockout Risk (%)", "procurement_cost": "Procurement Cost (₹)",
            }).style
            .background_gradient(subset=["Stockout Risk (%)"], cmap="YlOrRd")
            .background_gradient(subset=["Degradation Risk (%)"], cmap="Oranges")
            .format({"Forecast (kg)": "{:.1f}", "Order Qty (kg)": "{:.1f}",
                     "Safety Stock (kg)": "{:.1f}", "Degradation Risk (%)": "{:.1f}",
                     "Stockout Risk (%)": "{:.1f}", "Procurement Cost (₹)": "₹{:,.0f}"}),
            use_container_width=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    # RISK SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    st.header("⚠️ Risk & P&L Summary (Week 1)")
    if risk_metrics:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Procurement Cost", f"₹{risk_metrics['total_procurement_cost']:,.0f}")
        r2.metric("Forecasted Revenue", f"₹{risk_metrics['total_forecasted_revenue']:,.0f}")
        r3.metric("Net Profit (est.)", f"₹{risk_metrics['net_profit']:,.0f}",
                  delta=f"{risk_metrics['net_margin_pct']:.1f}% margin")
        r4.metric("Amount at Risk", f"₹{risk_metrics['amount_at_risk']:,.0f}",
                  delta_color="inverse",
                  delta=f"Degradation: ₹{risk_metrics['degradation_loss_value']:,.0f}")

        # P&L waterfall
        labels = ["Revenue", "Procurement", "OPEX", "Degradation Loss", "Net Profit"]
        values = [
            risk_metrics["total_forecasted_revenue"],
            -risk_metrics["total_procurement_cost"],
            -risk_metrics["weekly_opex"],
            -risk_metrics["degradation_loss_value"],
            risk_metrics["net_profit"],
        ]
        fig_pnl = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=labels, y=values,
            connector=dict(line=dict(color="gray", width=0.5, dash="dot")),
            increasing=dict(marker=dict(color="#27AE60")),
            decreasing=dict(marker=dict(color="#E74C3C")),
            totals=dict(marker=dict(color="#3498DB")),
        ))
        fig_pnl.update_layout(title="Weekly P&L Waterfall (₹)", height=350,
                               margin=dict(l=0, r=0, t=40, b=20))
        st.plotly_chart(fig_pnl, use_container_width=True)

        if risk_metrics.get("high_risk_products"):
            st.warning(
                f"⚠️ High stockout risk (>15%) for: **{', '.join(risk_metrics['high_risk_products'])}**. "
                "Consider increasing safety stock for these products."
            )

    # ─────────────────────────────────────────────────────────────────────
    # DAILY DEMAND FORECAST & DEVIATION RISK
    # ─────────────────────────────────────────────────────────────────────
    st.header("📊 Daily Demand Forecast & Deviation Risk")
    forecast_product = st.selectbox(
        "View daily forecast for:",
        options=selected,
        format_func=lambda x: PRETTY_NAMES[x],
    )
    if not fc_df.empty:
        prod_fc = fc_df[fc_df["product"] == forecast_product].copy()
        prod_fc["ds"] = pd.to_datetime(prod_fc["ds"])
        prod_fc["deviation_risk"] = (prod_fc["yhat"].std() / prod_fc["yhat"].mean() * 100
                                     if prod_fc["yhat"].mean() > 0 else 0)

        # Historical actuals for context
        hist = df[["Date", forecast_product]].tail(60)

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=hist["Date"], y=hist[forecast_product],
            mode="lines", name="Historical (last 60 days)",
            line=dict(color="lightgray", width=1.2),
        ))
        fig_daily.add_trace(go.Scatter(
            x=prod_fc["ds"], y=prod_fc["yhat"],
            mode="lines", name="Forecast",
            line=dict(color="#3498DB", width=2.5),
        ))

        # Get prophet forecast for confidence intervals
        prophet_fc = get_prophet_forecast(forecast_product, freq="D")
        if prophet_fc is not None:
            prophet_future = prophet_fc[prophet_fc["ds"] > df["Date"].max()].head(planning_weeks * 7)
            fig_daily.add_trace(go.Scatter(
                x=pd.concat([prophet_future["ds"], prophet_future["ds"].iloc[::-1]]),
                y=pd.concat([prophet_future["yhat_upper"], prophet_future["yhat_lower"].iloc[::-1]]),
                fill="toself", fillcolor="rgba(52,152,219,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="95% CI (Prophet)",
            ))

        fig_daily.add_vline(x=df["Date"].max(), line_dash="dash", line_color="navy", opacity=0.5)
        fig_daily.update_layout(
            title=f"Daily Forecast: {PRETTY_NAMES[forecast_product]} ({planning_weeks} weeks ahead)",
            xaxis_title="Date", yaxis_title="Sales (kg/day)",
            height=380, margin=dict(l=0, r=0, t=40, b=20),
        )
        st.plotly_chart(fig_daily, use_container_width=True)

        # Deviation risk gauge
        hist_std = float(df[forecast_product].std())
        hist_mean = float(df[forecast_product].mean())
        cv = hist_std / hist_mean * 100 if hist_mean > 0 else 0
        weekly_target = hist_mean * 7
        weekly_std = hist_std * np.sqrt(7)

        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Forecast CV (%)", f"{cv:.1f}%",
                      help="Coefficient of Variation — higher = harder to predict")
        col_r2.metric("Weekly Target (kg)", f"{weekly_target:,.1f}")
        col_r3.metric("σ of Weekly Demand (kg)", f"{weekly_std:,.1f}",
                      help="One standard deviation of weekly demand")

        # Risk of missing weekly target
        from scipy.stats import norm
        shortfall_risk = (1 - norm.cdf(weekly_target * 0.95, loc=weekly_target, scale=weekly_std)) * 100
        degradation = shelf_life.get(forecast_product, 7)
        max_hold_pct = min((7 / degradation) * 100, 100) if degradation > 0 else 100

        col_d1, col_d2 = st.columns(2)
        col_d1.metric("Shortfall Risk (<95% target)", f"{shortfall_risk:.1f}%",
                      delta_color="inverse",
                      delta="high" if shortfall_risk > 20 else "moderate" if shortfall_risk > 10 else "low")
        col_d2.metric("Degradation Exposure (%)", f"{max_hold_pct:.0f}%",
                      delta_color="inverse",
                      delta=f"Shelf life: {degradation} days")