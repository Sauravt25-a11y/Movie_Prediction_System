# train_model.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------
# 1) Load & clean dataset
# -------------------------
DATA_PATH = "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Movie_Rating_Prediction/IMDb_Movies_India.csv"
SAVE_DIR = "C:/Users/SAURAV THAKUR/Desktop/Codsoft/Movie_Rating_Prediction"

df = pd.read_csv(DATA_PATH, encoding="latin1")
df = df.dropna(subset=["Rating"]).copy()

# Clean numeric-like columns
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Duration"] = df["Duration"].astype(str).str.replace(" min", "", regex=False)
df["Duration"] = pd.to_numeric(df["Duration"], errors="coerce")
df["Votes"] = df["Votes"].astype(str).str.replace(",", "", regex=False)
df["Votes"] = pd.to_numeric(df["Votes"], errors="coerce")

# ==========================
# Extra Feature Engineering
# ==========================

# 1. Log-transform Votes (reduces skew)
df["Votes"] = np.log1p(df["Votes"])

# 2. Create "Decade" from Year
df["Decade"] = (df["Year"] // 10) * 10

# 3. Flag long movies
df["Is_Long_Movie"] = (df["Duration"] > 120).astype(int)

# 4. Group rare directors/actors into "Other" (top 50 only)
for col in ["Director", "Actor 1", "Actor 2", "Actor 3"]:
    top_items = df[col].value_counts().nlargest(50).index
    df[col] = df[col].where(df[col].isin(top_items), "Other")

# Features & target
X = df.drop(["Name", "Rating"], axis=1)
y = df["Rating"]

# -------------------------
# 2) Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 3) Preprocessing pipelines
# -------------------------
from category_encoders.target_encoder import TargetEncoder  # NEW

numeric_features = ["Duration", "Votes", "Is_Long_Movie"]

if df["Year"].notna().sum() > 0:
    numeric_features.append("Year")
else:
    print("Note: dropping 'Year' because it contains no observed values.")

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_low_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# 🔹 Use TargetEncoder instead of OrdinalEncoder
cat_high_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("target", TargetEncoder(handle_unknown="value", handle_missing="value"))
])

categorical_low = ["Genre"]
categorical_high = ["Director", "Actor 1", "Actor 2", "Actor 3"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat_low", cat_low_pipeline, categorical_low),
        ("cat_high", cat_high_pipeline, categorical_high),
    ],
    remainder="drop",
    sparse_threshold=0
)

# -------------------------
# 4) Define models
# -------------------------
from xgboost import XGBRegressor

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_model = None
best_score = -np.inf

# -------------------------
# 5) Train & evaluate models
# -------------------------
for name, regressor in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"RMSE": rmse, "R2": r2}
    print(f"🔹 {name} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    
    # Track best model
    if r2 > best_score:
        best_score = r2
        best_model = pipeline

# -------------------------
# 6) Save the best model
# -------------------------
MODEL_PATH = os.path.join(SAVE_DIR, "movie_model.pkl")
joblib.dump(best_model, MODEL_PATH)
print(f"🎉 Best model saved at: {MODEL_PATH}")
print("Best Model Performance:", results)
