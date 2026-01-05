import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("product_data.csv")

# ==============================
# Encode Categorical Variables
# ==============================
label_encoders = {}

for col in ["Month", "Product", "Category"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ==============================
# Features & Target
# ==============================
X = df.drop("Demand", axis=1)
y = df["Demand"]

# ==============================
# Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# Train Model
# ==============================
model = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# Save Model & Encoders
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(label_encoders, "encoders.pkl")

print("Model and encoders saved successfully!")
