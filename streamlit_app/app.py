"""
Dark Store Operations Planner — Streamlit App
Main entry point. Runs on port 7860 (HuggingFace Spaces).
"""

import sys
import os

# Ensure project root is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Dark Store Operations Planner",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar Navigation ---
st.sidebar.image(
    "https://img.icons8.com/emoji/96/convenience-store.png",
    width=60,
)
st.sidebar.title("Dark Store Planner")
st.sidebar.caption("Delhi NCR · Fruits & Vegetables · 32 SKUs")
st.sidebar.divider()

PAGES = {
    "📓 Notebook Findings": "findings",
    "📈 Analytics Dashboard": "analytics",
    "🏭 Supply Chain & Inventory": "supply_chain",
    "💰 Pricing Dashboard": "pricing",
}

selected = st.sidebar.radio(
    "Navigate",
    list(PAGES.keys()),
    index=0,
    label_visibility="collapsed",
)
st.sidebar.divider()
st.sidebar.caption(
    "Models: Prophet (monthly+) · LightGBM (daily/weekly)\n"
    "Data: 2022-01-01 → 2025-01-01 · 1097 days"
)

# --- Page Routing ---
section_name = PAGES[selected]

if section_name == "findings":
    from streamlit_app.sections.findings import render
elif section_name == "analytics":
    from streamlit_app.sections.analytics import render
elif section_name == "supply_chain":
    from streamlit_app.sections.supply_chain import render
elif section_name == "pricing":
    from streamlit_app.sections.pricing import render
else:
    render = lambda: st.error("Page not found")

render()