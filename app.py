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
@keyframes glow-bg {
    0% { box-shadow: 0 0 15px rgba(0,198,255,0.6); }
    33% { box-shadow: 0 0 15px rgba(124,255,0,0.6); }
    66% { box-shadow: 0 0 15px rgba(255,122,0,0.6); }
    100% { box-shadow: 0 0 15px rgba(0,198,255,0.6); }
}

.glow-box {
    background: #111111;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    animation: glow-bg 6s infinite linear;
}

.glow-title {
    font-size: 48px;
    font-weight: 700;
    color: white;
    margin: 0;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-top: 10px;
}

.features {
    max-width: 900px;
    margin: auto;
    font-size: 16px;
    color: #dddddd;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glow-box">
    <div class="glow-title">Product Demand Forecasting System</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
A machine learning based application to forecast future product demand
using historical sales data.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features">
<b>Project Highlights:</b>
<ul>
<li>Upload real-world historical sales data</li>
<li>Forecast next 6 months demand using Random Forest</li>
<li>Supports multiple products automatically</li>
<li>Interactive graph and table visualization</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

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

df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month")

df["month_num"] = df["Month"].dt.month
df["year"] = df["Month"].dt.year

products = df["family"].unique()

product = st.selectbox(
    "Select Product",
    products
)

product_df = df[df["family"] == product].copy()

product_df["time_index"] = range(1, len(product_df) + 1)

X = product_df[["time_index", "month_num", "year"]]
y = product_df["sales"]

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

last_time = product_df["time_index"].max()
last_month = product_df["month_num"].iloc[-1]
last_year = product_df["year"].iloc[-1]

future_rows = []

for i in range(1, 7):
    last_month += 1
    if last_month > 12:
        last_month = 1
        last_year += 1

    future_rows.append([
        last_time + i,
        last_month,
        last_year
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

view = st.radio(
    "Select Output View",
    ["Graph", "Table"],
    horizontal=True
)

if view == "Graph":
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        future_df["Future Month"],
        future_df["Predicted Demand"],
        marker="o",
        linewidth=2
    )
    ax.set_xlabel("Future Months")
    ax.set_ylabel("Predicted Demand")
    ax.set_title(f"Demand Forecast for Product: {product}")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
else:
    st.dataframe(future_df, use_container_width=True)
