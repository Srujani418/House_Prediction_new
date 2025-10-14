import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import json
import os

# Load data
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Encode categorical columns (same logic as training)
for col in X_test.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_test[col] = le.fit_transform(X_test[col])

# Load trained model
model = joblib.load("models/random_forest.joblib")

# Predict
y_pred = model.predict(X_test)

# Evaluate metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print for quick view
print(f"✅ RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")

# Save metrics for DVC tracking
os.makedirs("metrics", exist_ok=True)
metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)
