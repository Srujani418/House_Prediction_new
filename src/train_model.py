import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()  # ensures it's a 1D array

# Encode categorical features automatically
X_train = X_train.copy()
for col in X_train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
import joblib
import os
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest.joblib")

print("✅ Model trained successfully and saved as 'models/random_forest.joblib'")
