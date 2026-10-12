import pandas as pd
import numpy as np
import time
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1) Load CSV
# =========================
csv_path = "data/raw/smartfarm_soil_data.csv"   # เปลี่ยนชื่อตามไฟล์จริง

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print("Raw records:", len(df))

# =========================
# 2) Parse timestamp
# ใช้ ts_text เพราะเป็นรูปแบบ day/month/year เช่น 7/9/2025 = 7 Sep 2025
# =========================
df["timestamp"] = pd.to_datetime(df["ts_text"], dayfirst=True, errors="coerce")

# ถ้า ts_text มีปัญหา ค่อย fallback ไปใช้ ts
if df["timestamp"].isna().sum() > 0:
    df.loc[df["timestamp"].isna(), "timestamp"] = pd.to_datetime(
        df.loc[df["timestamp"].isna(), "ts"],
        errors="coerce"
    )

# =========================
# 3) Clean basic data
# =========================
soil_cols = ["soil1", "soil2", "soil3", "soil4"]

for col in soil_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["timestamp"] + soil_cols)
df = df.sort_values("timestamp")
df = df.drop_duplicates(subset=["timestamp"], keep="last")

print("Records after basic cleaning:", len(df))
print("Date range:", df["timestamp"].min(), "to", df["timestamp"].max())

# =========================
# 4) Convert to hourly time-series
# ใช้ค่าเฉลี่ยรายชั่วโมง
# =========================
hourly = (
    df.set_index("timestamp")[soil_cols]
    .resample("1H")
    .mean()
    .dropna()
)

print("Hourly records:", len(hourly))

# =========================
# 5) Feature engineering
# =========================
data = hourly.copy()

# Current soil features
data["soil_avg"] = data[soil_cols].mean(axis=1)
data["soil_min"] = data[soil_cols].min(axis=1)
data["soil_max"] = data[soil_cols].max(axis=1)
data["soil_std"] = data[soil_cols].std(axis=1)
data["soil_range"] = data["soil_max"] - data["soil_min"]

# Lag features
# ใช้ lag_1 และ lag_2 เพื่อให้จำนวน supervised samples ใกล้กับที่เขียนไว้ใน paper
data["soil_avg_lag_1"] = data["soil_avg"].shift(1)
data["soil_avg_lag_2"] = data["soil_avg"].shift(2)

# Rolling statistics, 3-hour window
data["soil_avg_roll_mean_3"] = data["soil_avg"].rolling(window=3).mean()
data["soil_avg_roll_min_3"] = data["soil_avg"].rolling(window=3).min()
data["soil_avg_roll_max_3"] = data["soil_avg"].rolling(window=3).max()
data["soil_avg_roll_std_3"] = data["soil_avg"].rolling(window=3).std()

# Moisture-change indicators
data["moisture_diff"] = data["soil_avg"] - data["soil_avg_lag_1"]
data["moisture_change_rate"] = data["moisture_diff"]

# Time variables
data["hour"] = data.index.hour
data["day"] = data.index.day

# Target variable: future average soil moisture at t+1 hour
data["target_t_plus_1h"] = data["soil_avg"].shift(-1)

# Drop rows with NaN from lag/rolling/target
model_df = data.dropna().copy()

print("Supervised samples:", len(model_df))

# =========================
# 6) Prepare X and y
# =========================
feature_cols = [
    "soil1", "soil2", "soil3", "soil4",
    "soil_avg", "soil_min", "soil_max", "soil_std", "soil_range",
    "soil_avg_lag_1", "soil_avg_lag_2",
    "soil_avg_roll_mean_3", "soil_avg_roll_min_3",
    "soil_avg_roll_max_3", "soil_avg_roll_std_3",
    "moisture_diff", "moisture_change_rate",
    "hour", "day"
]

X = model_df[feature_cols]
y = model_df["target_t_plus_1h"]

# =========================
# 7) Chronological split
# ห้าม shuffle
# =========================
n = len(model_df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]

X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))

# สำหรับรอบนี้ train ด้วย train+validation แล้ว test ด้วยช่วงท้าย
X_trainval = X.iloc[:val_end]
y_trainval = y.iloc[:val_end]

# =========================
# 8) Define models
# =========================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))
    ])
}

# =========================
# 9) Train and evaluate
# =========================
results = []
predictions = {}

for name, model in models.items():
    model.fit(X_trainval, y_trainval)

    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)

    # Inference time per sample
    repeat = 100
    start = time.perf_counter()
    for _ in range(repeat):
        _ = model.predict(X_test)
    end = time.perf_counter()

    inference_ms_per_sample = ((end - start) / (repeat * len(X_test))) * 1000

    # Model size
    model_size_kb = len(pickle.dumps(model)) / 1024

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE (%)": mape,
        "R2": r2,
        "Inference time (ms/sample)": inference_ms_per_sample,
        "Model size (KB)": model_size_kb
    })

results_df = pd.DataFrame(results)

# Sort by RMSE first
results_df = results_df.sort_values("RMSE").reset_index(drop=True)

print("\nModel results:")
print(results_df)

# =========================
# 10) Save outputs
# =========================
results_df.to_csv("results/model_results_table7.csv", index=False, encoding="utf-8-sig")

# Select best model by RMSE
best_model_name = results_df.iloc[0]["Model"]
best_pred = predictions[best_model_name]

actual_pred_df = pd.DataFrame({
    "timestamp": X_test.index,
    "actual_soil_moisture": y_test.values,
    "predicted_soil_moisture": best_pred
})

actual_pred_df.to_csv(
    "results/actual_vs_predicted_selected_model.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nBest model by RMSE:", best_model_name)
print("Saved: model_results_table7.csv")
print("Saved: actual_vs_predicted_selected_model.csv")
