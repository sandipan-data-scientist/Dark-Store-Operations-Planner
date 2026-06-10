"""
Section 4: Pricing Dashboard
Competitor pricing, relative pricing, margin analysis, unit-based pricing, monthly P&L forecast.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.config import ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES
from app.utils.data_loader import load_enriched_data
from app.utils.forecasting import get_prophet_forecast
from app.utils.model_registry import load_artifact


@st.cache_data(show_spinner=False)
def get_data():
    return load_enriched_data()


@st.cache_data(show_spinner=False)
def get_monthly_forecast(product: str) -> float:
    """Returns estimated monthly kg demand (Prophet or historical fallback)."""
    df = get_data()
    fc = get_prophet_forecast(product, freq="ME")
    if fc is not None and not fc.empty:
        fc = fc[fc["ds"] > df["Date"].max()]
        if not fc.empty:
            return float(fc.iloc[0]["yhat"])
    return float(df[product].mean() * 30)


def render():
    st.title("💰 Pricing Dashboard")
    st.markdown(
        "Set selling prices per product per unit, compare against competitors, "
        "analyze margins, and forecast monthly profitability."
    )

    df = get_data()

    # ─────────────────────────────────────────────────────────────────────
    # PRODUCT SELECTION
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 1: Select Products & Units")
    cat_filter = st.radio("Category", ["All", "Fruits Only", "Vegetables Only"], horizontal=True)
    if cat_filter == "Fruits Only":
        pool = FRUITS
    elif cat_filter == "Vegetables Only":
        pool = VEGETABLES
    else:
        pool = ALL_PRODUCTS

    selected = st.multiselect(
        "Products", pool, default=pool[:8], format_func=lambda x: PRETTY_NAMES[x]
    )
    if not selected:
        st.info("Select products above.")
        return

    # Unit type per product
    st.header("Step 2: Unit Configuration")
    st.caption(
        "Choose how each product is sold. If sold by **units** (e.g., individual mangoes), "
        "enter the average weight per unit so margin can be calculated correctly."
    )

    unit_types = {}
    unit_weight_grams = {}
    unit_cols = st.columns(4)
    for i, product in enumerate(selected):
        col = unit_cols[i % 4]
        with col:
            st.markdown(f"**{PRETTY_NAMES[product]}**")
            ut = st.selectbox("Unit", ["kg", "grams", "units"], key=f"ut_{product}",
                              label_visibility="collapsed")
            unit_types[product] = ut
            if ut == "units":
                wt = st.number_input(f"Avg weight (g/unit)",
                                     min_value=1.0, max_value=5000.0,
                                     value=200.0, key=f"wt_{product}",
                                     label_visibility="collapsed")
                unit_weight_grams[product] = wt
            else:
                unit_weight_grams[product] = 1000.0 if ut == "kg" else 1.0

    # ─────────────────────────────────────────────────────────────────────
    # PRICE INPUTS
    # ─────────────────────────────────────────────────────────────────────
    st.header("Step 3: Pricing Configuration")
    price_col1, price_col2, price_col3 = st.columns(3)

    your_prices = {}
    competitor_prices = {}
    vendor_costs = {}

    for i, product in enumerate(selected):
        label = PRETTY_NAMES[product]
        unit = unit_types[product]
        hist_mean = float(df[product].mean())
        # Typical price multipliers relative to wholesale
        default_wholesale = hist_mean * 0.55 / (1000 / unit_weight_grams[product] if unit == "units" else 1)
        default_your = hist_mean * 0.75 / (1000 / unit_weight_grams[product] if unit == "units" else 1)
        default_comp = hist_mean * 0.70 / (1000 / unit_weight_grams[product] if unit == "units" else 1)
        col_idx = i % 3

        if col_idx == 0:
            col = price_col1
        elif col_idx == 1:
            col = price_col2
        else:
            col = price_col3

        with col:
            st.markdown(f"**{label}** (per {unit})")
            your_prices[product] = st.number_input(
                f"Your price (₹/{unit})", min_value=0.0,
                value=round(default_your, 1), step=0.5, key=f"yp_{product}",
            )
            competitor_prices[product] = st.number_input(
                f"Competitor (₹/{unit})", min_value=0.0,
                value=round(default_comp, 1), step=0.5, key=f"cp_{product}",
            )
            vendor_costs[product] = st.number_input(
                f"Vendor cost (₹/kg)", min_value=0.0,
                value=round(default_wholesale, 1), step=0.5, key=f"vc_{product}",
            )
            st.divider()

    # ─────────────────────────────────────────────────────────────────────
    # COMPUTE PRICING ANALYSIS
    # ─────────────────────────────────────────────────────────────────────
    records = []
    for product in selected:
        label = PRETTY_NAMES[product]
        your_price = your_prices[product]
        comp_price = competitor_prices[product]
        cost = vendor_costs[product]
        unit = unit_types[product]
        uw_g = unit_weight_grams[product]

        # Convert to per-kg equivalent
        if unit == "grams":
            price_per_kg = your_price * 1000
            comp_per_kg = comp_price * 1000
        elif unit == "units":
            price_per_kg = your_price * (1000 / uw_g) if uw_g > 0 else 0
            comp_per_kg = comp_price * (1000 / uw_g) if uw_g > 0 else 0
        else:  # kg
            price_per_kg = your_price
            comp_per_kg = comp_price

        margin_per_kg = price_per_kg - cost
        margin_pct = (margin_per_kg / price_per_kg * 100) if price_per_kg > 0 else 0
        vs_comp = ((your_price - comp_price) / comp_price * 100) if comp_price > 0 else 0

        monthly_kg = get_monthly_forecast(product)
        monthly_revenue = monthly_kg * price_per_kg
        monthly_cost = monthly_kg * cost
        monthly_profit = monthly_revenue - monthly_cost

        # Historical: last 3 months avg revenue
        recent = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=90)]
        avg_monthly_kg_hist = float(recent[product].mean() * 30)
        hist_monthly_revenue = avg_monthly_kg_hist * price_per_kg
        revenue_deviation = ((monthly_revenue - hist_monthly_revenue) / hist_monthly_revenue * 100
                             if hist_monthly_revenue > 0 else 0)

        records.append({
            "product": product,
            "label": label,
            "unit": unit,
            "your_price": your_price,
            "competitor_price": comp_price,
            "vendor_cost_per_kg": cost,
            "price_per_kg": round(price_per_kg, 2),
            "comp_per_kg": round(comp_per_kg, 2),
            "margin_per_kg": round(margin_per_kg, 2),
            "margin_pct": round(margin_pct, 1),
            "vs_competitor_pct": round(vs_comp, 1),
            "position": "↑ Above" if vs_comp > 2 else ("↓ Below" if vs_comp < -2 else "≈ Parity"),
            "monthly_forecast_kg": round(monthly_kg, 1),
            "monthly_revenue": round(monthly_revenue, 2),
            "monthly_profit": round(monthly_profit, 2),
            "revenue_deviation_pct": round(revenue_deviation, 1),
        })

    results_df = pd.DataFrame(records)
    total_rev = results_df["monthly_revenue"].sum()
    total_profit = results_df["monthly_profit"].sum()
    avg_margin = results_df["margin_pct"].mean()

    # ─────────────────────────────────────────────────────────────────────
    # SUMMARY KPIs
    # ─────────────────────────────────────────────────────────────────────
    st.header("📊 Pricing Summary")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Monthly Revenue (est.)", f"₹{total_rev:,.0f}")
    k2.metric("Monthly Profit (est.)", f"₹{total_profit:,.0f}",
              delta=f"{(total_profit/total_rev*100):.1f}% net margin" if total_rev > 0 else None)
    k3.metric("Avg Gross Margin", f"{avg_margin:.1f}%",
              delta="healthy" if avg_margin > 20 else "tight")
    k4.metric("Products Priced Below Competitor",
              str(len(results_df[results_df["vs_competitor_pct"] < -2])))

    # ─────────────────────────────────────────────────────────────────────
    # COMPETITIVE POSITIONING CHART
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("🎯 Competitive Positioning")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_comp = px.bar(
            results_df.sort_values("vs_competitor_pct"),
            x="vs_competitor_pct", y="label", orientation="h",
            color="vs_competitor_pct",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title="Your Price vs Competitor (%)",
            labels={"vs_competitor_pct": "% Difference", "label": ""},
            text=results_df.sort_values("vs_competitor_pct")["vs_competitor_pct"].round(1),
        )
        fig_comp.add_vline(x=0, line_dash="solid", line_color="gray", opacity=0.5)
        fig_comp.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_comp.update_layout(height=max(350, 30 * len(selected)), margin=dict(l=0, r=30, t=40, b=20),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_chart2:
        fig_margin = px.bar(
            results_df.sort_values("margin_pct"),
            x="margin_pct", y="label", orientation="h",
            color="margin_pct",
            color_continuous_scale="RdYlGn",
            title="Gross Margin by Product (%)",
            labels={"margin_pct": "Margin (%)", "label": ""},
            text=results_df.sort_values("margin_pct")["margin_pct"].round(1),
        )
        fig_margin.add_vline(x=20, line_dash="dash", line_color="gray",
                             annotation_text="20% target", opacity=0.5)
        fig_margin.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_margin.update_layout(height=max(350, 30 * len(selected)), margin=dict(l=0, r=30, t=40, b=20),
                                 coloraxis_showscale=False)
        st.plotly_chart(fig_margin, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # MONTHLY REVENUE & PROFIT FORECAST
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📅 Monthly Revenue & Profit Forecast")
    fig_rev = go.Figure()
    sorted_res = results_df.sort_values("monthly_revenue", ascending=True)

    fig_rev.add_trace(go.Bar(
        x=sorted_res["monthly_revenue"], y=sorted_res["label"],
        orientation="h", name="Revenue",
        marker_color="#3498DB", opacity=0.8,
    ))
    fig_rev.add_trace(go.Bar(
        x=sorted_res["monthly_profit"], y=sorted_res["label"],
        orientation="h", name="Profit",
        marker_color="#27AE60", opacity=0.8,
    ))
    fig_rev.update_layout(
        barmode="overlay", height=max(350, 30 * len(selected)),
        title="Estimated Monthly Revenue vs Profit per Product (₹)",
        xaxis_title="Amount (₹)", margin=dict(l=0, r=0, t=40, b=20),
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig_rev, use_container_width=True)

    # Revenue deviation from historical
    st.subheader("📉 Forecast Deviation from Historical Average")
    fig_dev = px.bar(
        results_df.sort_values("revenue_deviation_pct"),
        x="revenue_deviation_pct", y="label", orientation="h",
        color="revenue_deviation_pct",
        color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
        title="Forecasted Monthly Revenue vs Past 3-Month Average (%)",
        labels={"revenue_deviation_pct": "Deviation (%)", "label": ""},
        text=results_df.sort_values("revenue_deviation_pct")["revenue_deviation_pct"].round(1),
    )
    fig_dev.add_vline(x=0, line_color="gray", opacity=0.4)
    fig_dev.update_traces(texttemplate="%{text}%", textposition="outside")
    fig_dev.update_layout(height=max(350, 30 * len(selected)), margin=dict(l=0, r=30, t=40, b=20),
                          coloraxis_showscale=False)
    st.plotly_chart(fig_dev, use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────
    # FULL PRICING TABLE
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("📋 Detailed Pricing Table")
    display = results_df[[
        "label", "unit", "your_price", "competitor_price", "vendor_cost_per_kg",
        "price_per_kg", "margin_per_kg", "margin_pct",
        "vs_competitor_pct", "position",
        "monthly_forecast_kg", "monthly_revenue", "monthly_profit",
    ]].copy()
    display.columns = [
        "Product", "Unit", "Your Price", "Comp. Price", "Vendor Cost (₹/kg)",
        "Your ₹/kg", "Margin/kg (₹)", "Margin (%)",
        "vs Competitor (%)", "Position",
        "Monthly Forecast (kg)", "Monthly Revenue (₹)", "Monthly Profit (₹)",
    ]
    st.dataframe(
        display.style
        .background_gradient(subset=["Margin (%)"], cmap="RdYlGn", vmin=0, vmax=40)
        .background_gradient(subset=["vs Competitor (%)"], cmap="RdYlGn", vmin=-20, vmax=20)
        .format({
            "Your Price": "₹{:.2f}", "Comp. Price": "₹{:.2f}",
            "Vendor Cost (₹/kg)": "₹{:.2f}", "Your ₹/kg": "₹{:.2f}",
            "Margin/kg (₹)": "₹{:.2f}", "Margin (%)": "{:.1f}%",
            "vs Competitor (%)": "{:+.1f}%",
            "Monthly Forecast (kg)": "{:,.1f}",
            "Monthly Revenue (₹)": "₹{:,.0f}",
            "Monthly Profit (₹)": "₹{:,.0f}",
        }),
        use_container_width=True,
    )

    # Download
    with st.expander("⬇️ Download Pricing Report"):
        csv = display.to_csv(index=False)
        st.download_button("Download Pricing CSV", csv, "pricing_analysis.csv", "text/csv")

    # ─────────────────────────────────────────────────────────────────────
    # PRICING RECOMMENDATIONS
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("💡 Pricing Recommendations")
    low_margin = results_df[results_df["margin_pct"] < 15].sort_values("margin_pct")
    overpriced = results_df[results_df["vs_competitor_pct"] > 10].sort_values("vs_competitor_pct", ascending=False)
    underpriced = results_df[results_df["vs_competitor_pct"] < -10].sort_values("vs_competitor_pct")

    if not low_margin.empty:
        st.warning(
            f"⚠️ **Low margin (<15%)**: {', '.join(low_margin['label'].tolist())}. "
            "Consider raising prices or negotiating better vendor rates."
        )
    if not overpriced.empty:
        st.error(
            f"📈 **Significantly above competition (>10%)**: {', '.join(overpriced['label'].tolist())}. "
            "You risk volume loss. Consider targeted discounts or bundling."
        )
    if not underpriced.empty:
        st.info(
            f"📉 **Below competition (>10% cheaper)**: {', '.join(underpriced['label'].tolist())}. "
            "Opportunity to capture market share, but monitor margin carefully."
        )

    # Optimal price range suggestion
    st.subheader("🎯 Suggested Optimal Price Range")
    opt_records = []
    for _, row in results_df.iterrows():
        cost = row["vendor_cost_per_kg"]
        comp = row["comp_per_kg"]
        target_margin = 0.22
        floor_price = cost / (1 - target_margin)
        ceiling_price = comp * 1.05
        optimal = (floor_price + ceiling_price) / 2
        unit = row["unit"]
        uw_g = unit_weight_grams.get(row["product"], 1000)
        if unit == "grams":
            floor_display = floor_price / 1000
            ceil_display = ceiling_price / 1000
            opt_display = optimal / 1000
        elif unit == "units":
            floor_display = floor_price * uw_g / 1000
            ceil_display = ceiling_price * uw_g / 1000
            opt_display = optimal * uw_g / 1000
        else:
            floor_display = floor_price
            ceil_display = ceiling_price
            opt_display = optimal

        opt_records.append({
            "Product": row["label"],
            "Unit": unit,
            "Current Price": row["your_price"],
            "Floor (22% margin)": round(floor_display, 2),
            "Suggested Optimal": round(opt_display, 2),
            "Ceiling (5% above comp.)": round(ceil_display, 2),
            "Margin at Optimal (%)": round(
                (optimal - cost) / optimal * 100 if optimal > 0 else 0, 1
            ),
        })

    opt_df = pd.DataFrame(opt_records)
    st.dataframe(
        opt_df.style.format({
            "Current Price": "₹{:.2f}", "Floor (22% margin)": "₹{:.2f}",
            "Suggested Optimal": "₹{:.2f}", "Ceiling (5% above comp.)": "₹{:.2f}",
            "Margin at Optimal (%)": "{:.1f}%",
        }),
        use_container_width=True,
    )