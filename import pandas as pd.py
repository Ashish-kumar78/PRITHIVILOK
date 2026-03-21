import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Synthetic dataset (hackathon friendly)
np.random.seed(42)
data_size = 1000

data = pd.DataFrame({
    "AQI": np.random.randint(50, 400, data_size),
    "WQI": np.random.randint(30, 100, data_size),
    "LandPollution": np.random.randint(20, 100, data_size)
})

# Create target (Environmental Risk Score)
data["RiskScore"] = (
    0.5 * data["AQI"] +
    -0.3 * data["WQI"] +
    0.4 * data["LandPollution"] +
    np.random.normal(0, 10, data_size)
)

X = data[["AQI", "WQI", "LandPollution"]]
y = data["RiskScore"]

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "env_model.pkl")

print("✅ Model trained & saved!")