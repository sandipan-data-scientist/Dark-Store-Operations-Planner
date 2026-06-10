"""
Section 2: Analytics & Performance Dashboard
Date-range selector, sales KPIs, trends, category breakdown, correlation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
import calendar

from app.config import ALL_PRODUCTS, FRUITS, VEGETABLES, PRETTY_NAMES
from app.utils.data_loader import load_enriched_data, get_date_range_data


@st.cache_data(show_spinner=False)
def get_data():
    return load_enriched_data()


def render():
    st.title("📈 Analytics & Performance Dashboard")
    st.markdown("Explore historical sales behavior between any two dates with full product breakdown.")

    df_full = get_data()
    min_date = df_full["Date"].min().date()
    max_date = df_full["Date"].max().date()

    # ── DATE RANGE & FILTERS ───────────────────────────────────────────────
    with st.sidebar:
        st.subheader("🗓 Date Range")
        col_s, col_e = st.columns(2)
        start_date = col_s.date_input("From", value=date(2024, 1, 1), min_value=min_date, max_value=max_date)
        end_date = col_e.date_input("To", value=max_date, min_value=min_date, max_value=max_date)

        st.subheader("🛒 Products")
        cat_filter = st.radio("Category", ["All", "Fruits Only", "Vegetables Only"])
        if cat_filter == "Fruits Only":
            avail_products = FRUITS
        elif cat_filter == "Vegetables Only":
            avail_products = VEGETABLES
        else:
            avail_products = ALL_PRODUCTS

        selected_products = st.multiselect(
            "Select products",
            options=avail_products,
            default=avail_products[:6],
            format_func=lambda x: PRETTY_NAMES[x],
        )
        granularity = st.selectbox("Aggregation", ["Daily", "Weekly", "Monthly", "Quarterly"],
                                   index=2)
        gran_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME", "Quarterly": "QE"}
        gran = gran_map[granularity]

    if not selected_products:
        st.warning("Select at least one product from the sidebar.")
        return
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    df = get_date_range_data(start_date, end_date)
    n_days = (end_date - start_date).days + 1

    # Compare with previous equal period
    prev_start = start_date - timedelta(days=n_days)
    prev_end = start_date - timedelta(days=1)
    df_prev = get_date_range_data(prev_start, prev_end) if prev_start >= min_date else pd.DataFrame()

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    st.subheader("📊 Period KPIs")
    total_curr = df[selected_products].sum().sum()
    total_prev = df_prev[selected_products].sum().sum() if not df_prev.empty else None
    delta_pct = ((total_curr - total_prev) / total_prev * 100) if total_prev else None

    daily_avg_curr = df[selected_products].sum(axis=1).mean()
    peak_day = df.loc[df[selected_products].sum(axis=1).idxmax(), "Date"]
    peak_val = df[selected_products].sum(axis=1).max()

    zero_days = (df[selected_products].sum(axis=1) == 0).sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales (kg)", f"{total_curr:,.0f}",
              delta=f"{delta_pct:+.1f}% vs prev period" if delta_pct else None)
    c2.metric("Daily Average (kg)", f"{daily_avg_curr:,.1f}")
    c3.metric("Period Days", str(n_days))
    c4.metric("Peak Day", str(peak_day.date()), delta=f"{peak_val:,.0f} kg")
    c5.metric("Zero-Sale Days", str(zero_days))

    st.divider()

    # ── TIME SERIES ─────────────────────────────────────────────────────────
    st.subheader(f"📉 {granularity} Sales Trend")
    df_ts = df[["Date"] + selected_products].copy()
    if gran != "D":
        df_ts = df_ts.set_index("Date").resample(gran).sum().reset_index()

    fig = go.Figure()
    palette = px.colors.qualitative.Safe
    for i, p in enumerate(selected_products):
        label = PRETTY_NAMES[p]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=df_ts["Date"], y=df_ts[p],
            mode="lines", name=label,
            line=dict(color=color, width=1.8),
            fill="tozeroy" if len(selected_products) == 1 else None,
            fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.1)"
            if len(selected_products) == 1 and color.startswith("#") else None,
        ))
    fig.update_layout(
        title=f"{granularity} Sales: {start_date} → {end_date}",
        xaxis_title="Date", yaxis_title="Sales (kg)",
        height=380, legend=dict(orientation="h", y=-0.25),
        margin=dict(l=0, r=0, t=40, b=60),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── PRODUCT BREAKDOWN ───────────────────────────────────────────────────
    st.subheader("🏆 Product Breakdown")
    col_a, col_b = st.columns(2)

    with col_a:
        product_totals = df[selected_products].sum().reset_index()
        product_totals.columns = ["product", "total_kg"]
        product_totals["label"] = product_totals["product"].map(PRETTY_NAMES)
        product_totals["category"] = product_totals["product"].apply(
            lambda p: "Fruit" if p in FRUITS else "Vegetable"
        )
        product_totals = product_totals.sort_values("total_kg", ascending=False)
        fig = px.pie(
            product_totals, values="total_kg", names="label",
            color="category",
            color_discrete_map={"Fruit": "#E74C3C", "Vegetable": "#27AE60"},
            title="Sales Share by Product",
            hole=0.4,
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        # Month-wise heatmap if range > 30 days
        if n_days > 30 and gran in ("ME", "QE", "D", "W"):
            df_heat = df[["Date"] + selected_products].copy()
            df_heat["month"] = df_heat["Date"].dt.strftime("%b %Y")
            monthly = df_heat.groupby("month")[selected_products].sum()
            monthly.columns = [PRETTY_NAMES[c] for c in monthly.columns]
            if len(monthly) > 1:
                fig = px.imshow(
                    monthly.T,
                    color_continuous_scale="YlGn",
                    labels=dict(x="Month", y="Product", color="Sales (kg)"),
                    title="Monthly Sales Heatmap",
                    text_auto=".0f",
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Extend date range for monthly heatmap (needs > 30 days).")
        else:
            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_agg = df.groupby("day_of_week")[selected_products].mean()
            dow_agg.index = dow_labels
            dow_agg.columns = [PRETTY_NAMES[c] for c in dow_agg.columns]
            fig = px.imshow(
                dow_agg.T, color_continuous_scale="Blues",
                labels=dict(x="Day of Week", y="Product", color="Avg Sales (kg)"),
                title="Average Sales by Day of Week",
                text_auto=".0f",
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # ── GROWTH TABLE ───────────────────────────────────────────────────────
    if not df_prev.empty:
        st.subheader("📊 Period-over-Period Growth")
        growth_rows = []
        for p in selected_products:
            curr_total = float(df[p].sum())
            prev_total = float(df_prev[p].sum()) if p in df_prev.columns else 0
            pct = (curr_total - prev_total) / prev_total * 100 if prev_total > 0 else None
            growth_rows.append({
                "Product": PRETTY_NAMES[p],
                "Current Period (kg)": round(curr_total, 1),
                "Previous Period (kg)": round(prev_total, 1),
                "Growth (%)": round(pct, 1) if pct is not None else None,
                "Category": "Fruit" if p in FRUITS else "Vegetable",
            })
        growth_df = pd.DataFrame(growth_rows).sort_values("Growth (%)", ascending=False)
        st.dataframe(
            growth_df.style.background_gradient(subset=["Growth (%)"], cmap="RdYlGn")
                           .format({"Current Period (kg)": "{:,.1f}",
                                    "Previous Period (kg)": "{:,.1f}",
                                    "Growth (%)": "{:+.1f}%"}, na_rep="N/A"),
            use_container_width=True,
        )

    # ── CORRELATION MATRIX ─────────────────────────────────────────────────
    if len(selected_products) >= 3:
        st.subheader("🔗 Cross-Product Correlation")
        corr = df[selected_products].replace(0, np.nan).corr()
        corr.index = [PRETTY_NAMES[p] for p in corr.index]
        corr.columns = [PRETTY_NAMES[p] for p in corr.columns]
        fig = px.imshow(
            corr, color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
            zmin=-1, zmax=1,
            labels=dict(color="Pearson r"),
            title="Pearson Correlation Matrix (zero-days excluded)",
            text_auto=".2f",
        )
        fig.update_layout(height=max(350, 40 * len(selected_products)))
        st.plotly_chart(fig, use_container_width=True)

    # ── RAW DATA EXPORT ─────────────────────────────────────────────────────
    with st.expander("⬇️ Download Data"):
        export = df[["Date"] + selected_products].copy()
        export.columns = ["Date"] + [PRETTY_NAMES[p] for p in selected_products]
        csv = export.to_csv(index=False)
        st.download_button("Download CSV", csv, "darkstore_sales.csv", "text/csv")