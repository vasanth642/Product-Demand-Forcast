import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
using historical sales and marketing data.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="features">
<b>Project Highlights:</b>
<ul>
<li>Upload historical sales data in CSV format</li>
<li>Predict demand for the next 6 months using machine learning</li>
<li>Interactive visualization with graph and table views</li>
<li>Supports multiple products with individual forecasting</li>
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

df["Month_Index"] = range(1, len(df) + 1)

product = st.selectbox(
    "Select Product",
    df["Product"].unique()
)

product_df = df[df["Product"] == product]

X = product_df[["Month_Index", "Price", "Ads_Spend", "Prev_Sales"]]
y = product_df["Demand"]

model = LinearRegression()
model.fit(X, y)

last_month = product_df["Month_Index"].max()
future_months = np.arange(last_month + 1, last_month + 7)

future_X = pd.DataFrame({
    "Month_Index": future_months,
    "Price": product_df["Price"].iloc[-1],
    "Ads_Spend": product_df["Ads_Spend"].mean(),
    "Prev_Sales": product_df["Prev_Sales"].iloc[-1]
})

future_demand = model.predict(future_X)

future_df = pd.DataFrame({
    "Future Month": [f"Month {i}" for i in range(1, 7)],
    "Predicted Demand": future_demand
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
    ax.set_title(f"Demand Forecast for Product {product}")
    ax.grid(alpha=0.3)
    st.pyplot(fig, use_container_width=True)
else:
    st.dataframe(future_df, use_container_width=True)
