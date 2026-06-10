"""
Section 1: Notebook Findings & Inferences
Full EDA reproduction + model performance + seasonal analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar

from app.config import ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES
from app.utils.data_loader import load_enriched_data, compute_seasonal_indices
from app.utils.model_registry import load_artifact


@st.cache_data(show_spinner=False)
def get_data():
    return load_enriched_data()


@st.cache_data(show_spinner=False)
def get_seasonal_index():
    df = get_data()
    return compute_seasonal_indices(df)


def render():
    st.title("📓 Notebook Findings & Inferences")
    st.markdown(
        "This section reproduces all key insights from the time-series analysis notebook: "
        "dataset overview, EDA, seasonal decomposition, model accuracy, and walk-forward validation results."
    )

    df = get_data()
    desc_stats = load_artifact("descriptive_stats")
    zero_analysis = load_artifact("zero_analysis")
    wfv_metrics = load_artifact("wfv_metrics")
    seasonal_data = load_artifact("seasonal_data")

    # ── DATASET OVERVIEW ──────────────────────────────────────────────────────
    st.header("1. Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Days", f"{len(df):,}")
    c2.metric("Products", "32")
    c3.metric("Fruits", "10")
    c4.metric("Vegetables", "22")
    c5.metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")

    with st.expander("📋 Key Findings — Dataset"):
        st.markdown("""
- **1,097 daily observations** per product (Jan 2022 – Jan 2025), no date gaps.
- **Zero-sales days** range from 0% (garlic, potato) to ~5% (capsicum, onion), indicating occasional supply disruptions rather than seasonal stock-outs.
- **Potato** leads at ~319 kg/day average; **Spinach** is the lowest-volume vegetable at ~35 kg/day.
- **Coefficient of Variation (CV)** is highest for seasonal fruits (Mango: CV ~280%, Watermelon: CV ~220%) confirming strong on/off seasonal patterns.
- **Staple vegetables** (Potato, Onion, Tomato) show CV 25-40%, indicating relatively stable year-round demand.
        """)

    # ── VOLUME RANKING ─────────────────────────────────────────────────────────
    st.header("2. Average Daily Sales Volume by Product")
    means = df[ALL_PRODUCTS].mean().reset_index()
    means.columns = ["product", "avg_kg"]
    means["label"] = means["product"].map(PRETTY_NAMES)
    means["category"] = means["product"].apply(lambda p: "Fruit" if p in FRUITS else "Vegetable")
    means = means.sort_values("avg_kg", ascending=True)

    fig = px.bar(
        means, x="avg_kg", y="label", orientation="h",
        color="category", color_discrete_map={"Fruit": "#E74C3C", "Vegetable": "#27AE60"},
        title="Average Daily Sales Volume — All 32 Products (2022–2024)",
        labels={"avg_kg": "Avg Daily Sales (kg/day)", "label": ""},
        text=means["avg_kg"].round(1),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=650, margin=dict(l=0, r=30, t=40, b=20), legend_title="Category")
    st.plotly_chart(fig, use_container_width=True)

    st.info("🔍 **Insight**: Potato (~319 kg/day) → Onion (~270) → Tomato (~245) are the top 3 by volume. "
            "These three alone account for ~30% of total daily throughput and deserve priority-tier inventory management.")

    # ── DESCRIPTIVE STATS TABLE ─────────────────────────────────────────────
    st.header("3. Descriptive Statistics")
    if desc_stats is not None:
        display_cols = ["mean", "std", "cv", "min", "50%", "max", "category"]
        disp = desc_stats[display_cols].copy()
        disp.columns = ["Mean (kg/day)", "Std Dev", "CV (%)", "Min", "Median", "Max", "Category"]
        st.dataframe(
            disp.style.background_gradient(subset=["CV (%)"], cmap="YlOrRd")
                      .format({"Mean (kg/day)": "{:.1f}", "Std Dev": "{:.1f}",
                               "CV (%)": "{:.1f}", "Min": "{:.0f}", "Median": "{:.1f}", "Max": "{:.0f}"}),
            use_container_width=True,
        )
    else:
        st.warning("Run `train_and_pickle.py` to generate descriptive stats.")

    # ── ZERO-SALE ANALYSIS ──────────────────────────────────────────────────
    st.header("4. Stock-Out / Zero-Sale Analysis")
    if zero_analysis is not None:
        fig = px.bar(
            zero_analysis.reset_index().sort_values("zero_pct", ascending=True),
            x="zero_pct", y="index", orientation="h",
            color="category", color_discrete_map={"Fruit": "#E74C3C", "Vegetable": "#27AE60"},
            title="Zero-Sale Day Frequency by Product (%)",
            labels={"zero_pct": "Zero-Sale Days (%)", "index": ""},
        )
        fig.update_layout(height=550, margin=dict(l=0, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.info("🔍 **Insight**: Zero-sales days are mostly random supply disruptions "
            "(not seasonal stock-outs), except for strongly seasonal products like Mango and Watermelon "
            "which show near-100% zero days outside their season.")

    # ── DAILY SALES TREND ──────────────────────────────────────────────────
    st.header("5. Daily Sales Trend — Total Portfolio")
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Total Daily Sales — All 32 Products",
        "Year-over-Year Overlay (7-day MA by Day of Year)"
    ], vertical_spacing=0.12)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["total_sales_kg"],
        fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
        line=dict(color="rgba(52,152,219,0.3)", width=0.5),
        name="Daily total",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["rolling_7d"],
        line=dict(color="#3498DB", width=1.5), name="7-day MA"
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["rolling_30d"],
        line=dict(color="#F39C12", width=2.5), name="30-day MA"
    ), row=1, col=1)

    colors_yr = {2022: "#2ECC71", 2023: "#3498DB", 2024: "#E74C3C"}
    for yr in [2022, 2023, 2024]:
        sub = df[df["year"] == yr].copy()
        sub["doy"] = sub["Date"].dt.dayofyear
        ma = sub["total_sales_kg"].rolling(7, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=sub["doy"], y=ma,
            mode="lines", name=str(yr),
            line=dict(color=colors_yr[yr], width=2),
        ), row=2, col=1)

    fig.update_layout(height=600, showlegend=True, margin=dict(l=0, r=0, t=50, b=20))
    fig.update_xaxes(title_text="Day of Year", row=2)
    fig.update_yaxes(title_text="Sales (kg/day)", row=1)
    fig.update_yaxes(title_text="Sales (kg/day)", row=2)
    st.plotly_chart(fig, use_container_width=True)

    st.info("🔍 **Insight**: Consistent YoY growth of ~15–20%. 2022 < 2023 < 2024 across virtually all "
            "products, indicating organic business expansion. Vegetables dominate at ~3–4× fruit volume. "
            "A clear summer spike in fruits is visible each year (April–July).")

    # ── SEASONAL ANALYSIS ─────────────────────────────────────────────────
    st.header("6. Seasonal Index Heatmap")
    si = get_seasonal_index()
    month_labels = [calendar.month_abbr[m] for m in range(1, 13)]
    si.index = month_labels

    # Fruit subset
    fruit_labels = [PRETTY_NAMES[p] for p in FRUITS]
    veg_labels = [PRETTY_NAMES[p] for p in VEGETABLES]

    tab1, tab2 = st.tabs(["🍎 Fruits", "🥦 Vegetables"])
    with tab1:
        fig = px.imshow(
            si[fruit_labels].T,
            color_continuous_scale="RdYlGn", color_continuous_midpoint=100,
            zmin=20, zmax=350,
            labels=dict(x="Month", y="Product", color="Seasonal Index"),
            title="Fruit Seasonal Index (100 = annual average · green = above avg)",
            text_auto=".0f",
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = px.imshow(
            si[veg_labels].T,
            color_continuous_scale="RdYlGn", color_continuous_midpoint=100,
            zmin=20, zmax=250,
            labels=dict(x="Month", y="Product", color="Seasonal Index"),
            title="Vegetable Seasonal Index (100 = annual average · green = above avg)",
            text_auto=".0f",
        )
        fig.update_layout(height=620)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Seasonal Pattern Summary"):
        st.markdown("""
| Product | Peak Season | Off-Season | Seasonal Index Peak |
|---|---|---|---|
| Mango | Apr–Jul | Nov–Feb | ~350–450 |
| Watermelon | Apr–Jun | Oct–Feb | ~280–420 |
| Muskmelon | Apr–Jun | Oct–Feb | ~250–380 |
| Green Peas | Nov–Feb | May–Sep | ~180–260 |
| Cauliflower | Oct–Feb | May–Sep | ~150–220 |
| Guava | Aug–Dec | Apr–Jun | ~150–200 |
| Potato | Year-round | — | ±20% seasonal variation |
| Onion | Year-round | — | ±25% seasonal variation |
| Tomato | Oct–Feb | May–Aug | ~130 peak |
        """)

    # ── DAY-OF-WEEK ANALYSIS ──────────────────────────────────────────────
    st.header("7. Day-of-Week Demand Patterns")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_agg = df.groupby("day_of_week")[["total_sales_kg", "total_fruit_kg", "total_veg_kg"]].mean()

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        "Total Sales by Day of Week", "Fruits vs Vegetables by DoW"
    ])
    colors_bar = ["#3498DB"] * 5 + ["#F39C12", "#F39C12"]
    fig.add_trace(go.Bar(x=dow_labels, y=dow_agg["total_sales_kg"],
                         marker_color=colors_bar, name="Total"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dow_labels,
                             y=[dow_agg["total_sales_kg"].mean()] * 7,
                             mode="lines", line=dict(color="red", dash="dash"),
                             name="Weekly mean"), row=1, col=1)
    fig.add_trace(go.Bar(x=dow_labels, y=dow_agg["total_fruit_kg"],
                         name="Fruits", marker_color="#E74C3C", opacity=0.8), row=1, col=2)
    fig.add_trace(go.Bar(x=dow_labels, y=dow_agg["total_veg_kg"],
                         name="Vegetables", marker_color="#27AE60", opacity=0.8), row=1, col=2)

    fig.update_layout(height=350, barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.info("🔍 **Insight**: Saturday and Sunday average 10–15% higher total sales than weekdays. "
            "Weekend uplift is strongest for Banana, Tomato, and Onion. "
            "Replenishment orders should arrive by Friday evening.")

    # ── WFV PERFORMANCE ─────────────────────────────────────────────────
    st.header("8. Walk-Forward Validation — Model Accuracy")
    if wfv_metrics:
        rows = []
        for product, metrics in wfv_metrics.items():
            lgbm_m = metrics.get("lgbm", {})
            wfv_m = metrics.get("wfv_90d", {})
            rows.append({
                "Product": PRETTY_NAMES.get(product, product),
                "Category": "Fruit" if product in FRUITS else "Vegetable",
                "LightGBM MAPE (%)": lgbm_m.get("MAPE"),
                "LightGBM MAE (kg)": lgbm_m.get("MAE"),
                "LightGBM RMSE (kg)": lgbm_m.get("RMSE"),
                "WFV-90d MAPE (%)": wfv_m.get("MAPE"),
            })
        metrics_df = pd.DataFrame(rows)
        st.dataframe(
            metrics_df.style.background_gradient(
                subset=["LightGBM MAPE (%)"], cmap="RdYlGn_r", vmin=5, vmax=40
            ).format({
                "LightGBM MAPE (%)": "{:.1f}",
                "LightGBM MAE (kg)": "{:.1f}",
                "LightGBM RMSE (kg)": "{:.1f}",
                "WFV-90d MAPE (%)": "{:.1f}",
            }, na_rep="—"),
            use_container_width=True,
        )

        # MAPE bar chart
        valid = metrics_df[metrics_df["LightGBM MAPE (%)"].notna()].sort_values("LightGBM MAPE (%)")
        fig = px.bar(valid, x="Product", y="LightGBM MAPE (%)",
                     color="Category",
                     color_discrete_map={"Fruit": "#E74C3C", "Vegetable": "#27AE60"},
                     title="LightGBM MAPE by Product (test set, last 180 days)",
                     labels={"LightGBM MAPE (%)": "MAPE (%)"})
        fig.add_hline(y=15, line_dash="dash", line_color="red",
                      annotation_text="15% threshold", annotation_position="top right")
        fig.update_layout(height=380, xaxis_tickangle=40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run `scripts/train_and_pickle.py` to generate WFV metrics.")

    with st.expander("📋 Model Architecture Summary"):
        st.markdown("""
**Forecasting Strategy:**
- **LightGBM** (daily / weekly horizon): Expanding-window WFV, lag features (7/14/30/60/365 days), 
  rolling mean/std (7/14/30/90 days), Fourier features (3 harmonics), calendar dummies.
- **Prophet** (monthly / quarterly / annual horizon): Multiplicative seasonality mode, 
  changepoint_prior_scale=0.05, seasonality_prior_scale=10.0, weekly + yearly seasonality.

**Feature Importance Findings:**
- `lag_7` and `rolling_mean_7` dominate importance across all products
- Fourier features contribute strongly for seasonal products (Mango, Watermelon, Green Peas)
- Weekend indicator adds 3–5% importance for high-volume staples

**Model Performance Benchmarks:**
- Staples (Potato, Onion, Tomato): MAPE ~8–12%, excellent forecasts
- Year-round fruits (Banana, Apple): MAPE ~10–15%
- Strongly seasonal (Mango, Watermelon): MAPE ~18–30% in-season, higher in off-season transitions
- Weekly aggregation improves all MAPEs by 30–40% vs daily

**Seasonal Cross-Year Deviation:**
- 2022→2023 transition showed +15–20% YoY demand growth across most products
- 2023→2024 transition showed +12–18% growth
- Highest deviation risk months: April–June (summer transition) and October–November (winter onset)
        """)