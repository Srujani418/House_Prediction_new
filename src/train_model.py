# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder

# # Load processed data
# X_train = pd.read_csv("data/processed/X_train.csv")
# y_train = pd.read_csv("data/processed/y_train.csv").squeeze()  # ensures it's a 1D array

# # Encode categorical features automatically
# X_train = X_train.copy()
# for col in X_train.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     X_train[col] = le.fit_transform(X_train[col])

# # Train model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Save model
# import joblib
# import os
# os.makedirs("models", exist_ok=True)
# joblib.dump(model, "models/random_forest.joblib")

# print("✅ Model trained successfully and saved as 'models/random_forest.joblib'")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn

# Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

# Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# Define preprocessing: One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep numeric columns as-is
)

# Define model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Combine preprocessing + model into a pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Train model with MLflow tracking
with mlflow.start_run():
    pipeline.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics and parameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)

    # Log model
    mlflow.sklearn.log_model(pipeline, "random_forest_model")

    # Save locally
    joblib.dump(pipeline, "models/random_forest_pipeline.joblib")

print("✅ Model trained successfully, categorical columns encoded, and metrics logged in MLflow.")


