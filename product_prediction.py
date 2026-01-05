import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ===== Step 1: Load Dataset =====
df = pd.read_csv("product_data.csv")

# ===== Step 2: Encode Categorical Variables =====
label_encoders = {}
for col in ['Month', 'Product', 'Category']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===== Step 3: Prepare Features & Target =====
X = df.drop("Demand", axis=1)
y = df["Demand"]
product_labels = df["Product"]

# ===== Step 4: Stratified Split =====
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in splitter.split(X, product_labels):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    product_test = product_labels.iloc[test_idx]

# ===== Step 5: Train Model =====
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# ===== Step 6: Predictions =====
df_test = X_test.copy()
df_test["Actual"] = y_test
df_test["Predicted"] = model.predict(X_test)
df_test["Product"] = product_test
df_test["Product"] = label_encoders["Product"].inverse_transform(df_test["Product"])

# ===== Step 7: Plot with Seaborn =====
plt.style.use("dark_background")
sns.set_theme(style="darkgrid")

plt.figure(figsize=(12, 6))
palette = {"A": "lime", "B": "magenta", "C": "orange"}

for product in ["A", "B", "C"]:
    subset = df_test[df_test["Product"] == product].reset_index(drop=True)
    sns.lineplot(x=subset.index, y=subset["Actual"], label=f"{product} - Actual",
                 color=palette[product], marker="o", linewidth=2)
    sns.lineplot(x=subset.index, y=subset["Predicted"], label=f"{product} - Predicted",
                 color=palette[product], linestyle="--", marker="x", linewidth=2)

plt.title("📊 Product Demand Forecasting ", fontsize=16, fontweight="bold")
plt.xlabel("Sample Index (Grouped by Product)")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()
plt.show()

