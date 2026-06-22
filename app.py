import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Product Demand Forecast",
    layout="wide"
)

st.markdown("""
<style>
/* Custom dashboard styling */
.dashboard-header {
    background-color: #FFFFFF;
    padding: 24px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    margin-bottom: 24px;
}
.dashboard-title {
    font-size: 28px;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.dashboard-subtitle {
    font-size: 15px;
    color: #64748B;
    margin-top: 6px;
    margin-bottom: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}
.highlights-container {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    margin-bottom: 24px;
}
.highlights-title {
    font-size: 16px;
    font-weight: 600;
    color: #334155;
    margin-bottom: 12px;
}
.highlights-list {
    margin: 0;
    padding-left: 20px;
    color: #475569;
    font-size: 14px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">Product Demand Forecasting System</div>
    <div class="dashboard-subtitle">
        A machine learning based application to forecast future product demand
        using historical sales data.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="highlights-container">
    <div class="highlights-title">Project Highlights</div>
    <ul class="highlights-list">
        <li>Upload real-world historical sales data</li>
        <li>Forecast next 6 months demand using Random Forest</li>
        <li>Supports interactive forecasting trend line analysis</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar setup
st.sidebar.header("Data Ingestion")
file = st.sidebar.file_uploader(
    "Upload Sales Dataset (CSV)",
    type=["csv"]
)

if file is None:
    st.info("Please upload a CSV file to start demand prediction.")
    st.stop()

df = pd.read_csv(file)

st.subheader("Uploaded Dataset")
st.dataframe(df, use_container_width=True)

# Data preprocessing
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")

df["month_num"] = df["Month"].dt.month
df["year"] = df["Month"].dt.year

products = df["family"].unique()

product = st.selectbox(
    "Select Product to Analyze",
    products
)

product_df = df[df["family"] == product].copy()
product_df["time_index"] = range(1, len(product_df) + 1)

X = product_df[["time_index", "month_num", "year"]]
y = product_df["sales"]

# Cached model training
@st.cache_resource
def train_forecaster(X_train, y_train):
    mdl = RandomForestRegressor(n_estimators=200, random_state=42)
    mdl.fit(X_train, y_train)
    return mdl

model = train_forecaster(X, y)

# Future dates loop execution
last_time = product_df["time_index"].max()
current_month = product_df["month_num"].iloc[-1]
current_year = product_df["year"].iloc[-1]

future_rows = []
for i in range(1, 7):
    current_month += 1
    if current_month > 12:
        current_month = 1
        current_year += 1

    future_rows.append([
        last_time + i,
        current_month,
        current_year
    ])

future_X = pd.DataFrame(
    future_rows,
    columns=["time_index", "month_num", "year"]
)

future_sales = model.predict(future_X)

future_df = pd.DataFrame({
    "Future Month": pd.date_range(
        start=product_df["Month"].iloc[-1] + pd.offsets.MonthBegin(),
        periods=6,
        freq="MS"
    ).strftime("%Y-%m"),
    "Predicted Demand": future_sales
})

# Complete, centralized UI view selector
view = st.radio(
    "Select Visualization Output",
    ["Line Graph", "Data Table"],
    horizontal=True
)

if view == "Line Graph":
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(
        future_df["Future Month"],
        future_df["Predicted Demand"],
        marker="o",
        linewidth=2,
        color="#2563EB"
    )
    ax.set_xlabel("Future Months", fontsize=10, color="#475569")
    ax.set_ylabel("Predicted Demand", fontsize=10, color="#475569")
    ax.set_title(f"Demand Trend Forecast for: {product}", fontsize=12, fontweight="bold", color="#0F172A")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.grid(True, linestyle="--", alpha=0.5, color="#E2E8F0")
    st.pyplot(fig, use_container_width=True)

elif view == "Data Table":
    st.dataframe(future_df, use_container_width=True)