#!/usr/bin/env python3
"""SmartFarm soil-moisture forecasting: reproducible, gap-aware v2 evaluation.

Run from the SmartFarm repository root on the Raspberry Pi 4:
    python3 scripts/run_soil_prediction_v2.py

Or supply an input path and output directory explicitly:
    python3 run_soil_prediction_v2.py --input data/raw/smartfarm_soil_data.csv --output-dir results

Key differences from v1:
  * Keep missing hours in the hourly time axis until ALL features and targets
    have been constructed. Therefore, shift(-1) means exactly +1 clock hour.
  * Keep only samples with a complete 3-hour feature window and t+1 target.
  * Remove the duplicate moisture_change_rate feature (18 features, not 19).
  * Select the algorithm by VALIDATION RMSE only, then refit on pre-test data.
  * Purge samples whose target time reaches the next split's first origin.
  * Evaluate all five refitted models once on the untouched chronological test.
  * Export new *_v2 files; never overwrite the original experiment's results.

The 100-repeat batch benchmark measures model-level time per sample, NOT
single-observation latency or an end-to-end irrigation-system response time.
Run the final benchmark on the Raspberry Pi 4 to report Pi-specific timings.
"""

import argparse
import json
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SOIL_COLS = ["soil1", "soil2", "soil3", "soil4"]
FEATURE_COLS = [
    "soil1", "soil2", "soil3", "soil4",
    "soil_avg", "soil_min", "soil_max", "soil_std", "soil_range",
    "soil_avg_lag_1", "soil_avg_lag_2",
    "soil_avg_roll_mean_3", "soil_avg_roll_min_3",
    "soil_avg_roll_max_3", "soil_avg_roll_std_3",
    "moisture_diff", "hour", "day",
]
TARGET_COL = "target_t_plus_1h"
DELTA_1H = pd.Timedelta(hours=1)


def make_models():
    """Keep exactly the five v1 algorithms and v1 hyperparameters."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1)),
        ]),
    }


def load_records(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            "CSV not found: {}\nRun from your repository root or use --input."
            .format(csv_path)
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    required = ["ts_text", *SOIL_COLS]
    missing_columns = [col for col in required if col not in df.columns]
    if missing_columns:
        raise ValueError("Missing required CSV columns: {}".format(missing_columns))

    raw_n = len(df)
    # ts_text uses DD/MM/YYYY; ts, if present, uses MM/DD/YYYY.
    # Do not infer ambiguous dates by mixing both formats in one column.
    df["timestamp"] = pd.to_datetime(
        df["ts_text"], format="%d/%m/%Y %H:%M", errors="coerce"
    )
    if "ts" in df.columns:
        missing_text = df["timestamp"].isna()
        if missing_text.any():
            df.loc[missing_text, "timestamp"] = pd.to_datetime(
                df.loc[missing_text, "ts"], errors="coerce", dayfirst=False
            )

    for col in SOIL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    incomplete_n = int(df[["timestamp", *SOIL_COLS]].isna().any(axis=1).sum())
    df = df.dropna(subset=["timestamp", *SOIL_COLS])
    df = df.sort_values("timestamp", kind="mergesort")
    duplicate_n = int(df.duplicated(subset=["timestamp"]).sum())
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    if df.empty:
        raise ValueError("No valid sensor records remain after cleaning")

    audit = {
        "raw_records": raw_n,
        "excluded_incomplete_raw_records": incomplete_n,
        "excluded_duplicate_timestamps": duplicate_n,
        "cleaned_raw_records": len(df),
        "first_raw_timestamp": str(df["timestamp"].min()),
        "last_raw_timestamp": str(df["timestamp"].max()),
    }
    return df, audit


def make_supervised(df):
    # IMPORTANT: do NOT add .dropna() here. Keep every empty clock hour in
    # the index so shift(1), shift(2), rolling(3) and shift(-1) mean hours.
    hourly = (
        df.set_index("timestamp")[SOIL_COLS]
        .resample("1h")
        .mean()
    )
    missing_hours = int(hourly[SOIL_COLS].isna().any(axis=1).sum())
    observed_hours = len(hourly) - missing_hours

    data = hourly.copy()
    data["soil_avg"] = data[SOIL_COLS].mean(axis=1)
    data["soil_min"] = data[SOIL_COLS].min(axis=1)
    data["soil_max"] = data[SOIL_COLS].max(axis=1)
    data["soil_std"] = data[SOIL_COLS].std(axis=1)
    data["soil_range"] = data["soil_max"] - data["soil_min"]

    data["soil_avg_lag_1"] = data["soil_avg"].shift(1)
    data["soil_avg_lag_2"] = data["soil_avg"].shift(2)
    roll = data["soil_avg"].rolling(window=3, min_periods=3)
    data["soil_avg_roll_mean_3"] = roll.mean()
    data["soil_avg_roll_min_3"] = roll.min()
    data["soil_avg_roll_max_3"] = roll.max()
    data["soil_avg_roll_std_3"] = roll.std()

    data["moisture_diff"] = data["soil_avg"] - data["soil_avg_lag_1"]
    # v1 moisture_change_rate was an exact copy of moisture_diff.
    # Remove that duplicate predictor rather than fabricate a different rate.
    data["hour"] = data.index.hour
    data["day"] = data.index.day

    # With all clock hours retained, next row ALWAYS corresponds to t+1 hour.
    data[TARGET_COL] = data["soil_avg"].shift(-1)
    data["target_timestamp"] = data.index + DELTA_1H
    model_df = data.dropna(subset=[*FEATURE_COLS, TARGET_COL]).copy()
    if model_df.empty:
        raise ValueError("No complete t+1 supervised samples remain")

    # Explicit sanity checks: complete feature window and observed t+1 label.
    assert model_df.index.is_monotonic_increasing
    assert model_df.index.is_unique
    assert (model_df["target_timestamp"] - model_df.index == DELTA_1H).all()
    assert not model_df[FEATURE_COLS + [TARGET_COL]].isna().any().any()

    audit = {
        "hourly_slots_including_missing": len(hourly),
        "observed_hourly_records": observed_hours,
        "missing_hourly_intervals": missing_hours,
        "supervised_samples": len(model_df),
        "feature_count": len(FEATURE_COLS),
        "feature_columns": FEATURE_COLS,
        "first_supervised_origin": str(model_df.index.min()),
        "last_supervised_origin": str(model_df.index.max()),
    }
    return model_df, audit


def split_chronologically(model_df):
    n = len(model_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if train_end < 5 or val_end - train_end < 2 or n - val_end < 2:
        raise ValueError("Not enough samples for train/validation/test splits")

    train_candidates = model_df.iloc[:train_end]
    val_candidates = model_df.iloc[train_end:val_end]
    test_df = model_df.iloc[val_end:]

    first_val_origin = val_candidates.index.min()
    first_test_origin = test_df.index.min()

    # Conservative one-hour boundary purging: a target used in an earlier
    # phase cannot reach or exceed the origin time of the next phase.
    train_df = train_candidates.loc[
        train_candidates["target_timestamp"] < first_val_origin
    ]
    val_df = val_candidates.loc[
        val_candidates["target_timestamp"] < first_test_origin
    ]

    # Final fit uses all PRE-test origins with labels strictly before the
    # first test origin, regardless of the earlier train/validation boundary.
    trainval_candidates = model_df.iloc[:val_end]
    trainval_df = trainval_candidates.loc[
        trainval_candidates["target_timestamp"] < first_test_origin
    ]

    if min(len(train_df), len(val_df), len(test_df), len(trainval_df)) < 2:
        raise ValueError("Boundary purge leaves an insufficient split")
    assert train_df["target_timestamp"].max() < first_val_origin
    assert val_df["target_timestamp"].max() < first_test_origin
    assert trainval_df["target_timestamp"].max() < first_test_origin
    assert trainval_df.index.max() < first_test_origin

    audit = {
        "split_strategy": "chronological_70_15_15_by_forecast_origin_before_boundary_purge",
        "initial_train_candidates": len(train_candidates),
        "initial_validation_candidates": len(val_candidates),
        "initial_test_candidates": len(test_df),
        "purged_train_boundary_samples": len(train_candidates) - len(train_df),
        "purged_validation_boundary_samples": len(val_candidates) - len(val_df),
        "training_samples_for_validation_selection": len(train_df),
        "validation_samples_for_model_selection": len(val_df),
        "refit_samples_before_test": len(trainval_df),
        "test_samples": len(test_df),
        "first_validation_origin": str(first_val_origin),
        "first_test_origin": str(first_test_origin),
    }
    return train_df, val_df, trainval_df, test_df, audit


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        raise ValueError("MAPE undefined: zero observed soil moisture found")
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE (%)": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        "R2": float(r2_score(y_true, y_pred)),
    }


def benchmark_ms_per_sample(model, X_test, repetitions):
    # Warm up prediction to reduce one-time initialization effects.
    for _ in range(3):
        model.predict(X_test)
    start = time.perf_counter()
    for _ in range(repetitions):
        model.predict(X_test)
    elapsed = time.perf_counter() - start
    return float(elapsed * 1000.0 / (repetitions * len(X_test)))


def get_device_info():
    model_file = Path("/proc/device-tree/model")
    device_model = None
    if model_file.is_file():
        try:
            device_model = model_file.read_bytes().decode("utf-8", errors="replace").strip("\x00\n")
        except OSError:
            pass
    return {
        "hardware_model_from_os": device_model,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def save_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print("Saved:", path)


def run(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df, raw_audit = load_records(Path(args.input))
    model_df, hourly_audit = make_supervised(df)
    train_df, val_df, trainval_df, test_df, split_audit = split_chronologically(model_df)

    def xy(frame):
        return frame[FEATURE_COLS], frame[TARGET_COL]

    X_train, y_train = xy(train_df)
    X_val, y_val = xy(val_df)
    X_trainval, y_trainval = xy(trainval_df)
    X_test, y_test = xy(test_df)

    print("\n=== Dataset audit ===")
    for key, val in {**raw_audit, **hourly_audit, **split_audit}.items():
        if key != "feature_columns":
            print("{}: {}".format(key, val))
    print("Features:", ", ".join(FEATURE_COLS))

    # Stage A: select the MODEL FAMILY using validation data only.
    validation_rows = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        validation_rows.append({"Model": name, **metrics(y_val, val_pred)})
    validation_df = pd.DataFrame(validation_rows).sort_values(
        ["RMSE", "Model"], ascending=[True, True]
    ).reset_index(drop=True)
    selected_name = str(validation_df.loc[0, "Model"])

    print("\n=== Validation results (training-only fits) ===")
    print(validation_df.to_string(index=False, float_format=lambda x: "{:.6f}".format(x)))
    print("Selected by validation RMSE:", selected_name)

    # Stage B: refit EACH fixed-configuration model using pre-test data.
    # Then evaluate exactly once on the untouched test set. Test RMSE does
    # NOT influence selected_name, although all test results are reported.
    test_rows = []
    selected_pred = None
    selected_pickle = None
    for name, model in make_models().items():
        model.fit(X_trainval, y_trainval)
        pred = model.predict(X_test)
        serialized = pickle.dumps(model)  # Same default serialization as v1
        row = {
            "Model": name,
            **metrics(y_test, pred),
            "Inference time (ms/sample)": benchmark_ms_per_sample(
                model, X_test, args.repeats
            ),
            "Model size (KiB)": len(serialized) / 1024.0,
        }
        test_rows.append(row)
        if name == selected_name:
            selected_pred = pred
            selected_pickle = serialized

    test_df_results = pd.DataFrame(test_rows).sort_values(
        ["RMSE", "Model"], ascending=[True, True]
    ).reset_index(drop=True)
    test_df_results["Selected by validation RMSE"] = (
        test_df_results["Model"] == selected_name
    )

    print("\n=== Final test results (all five refitted models) ===")
    print(test_df_results.to_string(index=False, float_format=lambda x: "{:.6f}".format(x)))

    # The timestamp column means the TARGET hour for plotting; keep the
    # forecast-origin timestamp separately to make horizon auditable.
    selected_predictions = pd.DataFrame({
        "timestamp": test_df["target_timestamp"].values,
        "forecast_origin": test_df.index,
        "actual_soil_moisture": y_test.to_numpy(),
        "predicted_soil_moisture": selected_pred,
        "selected_model": selected_name,
    })
    assert (pd.to_datetime(selected_predictions["timestamp"]) -
            pd.to_datetime(selected_predictions["forecast_origin"]) == DELTA_1H).all()

    save_csv(validation_df, output_dir / "validation_metrics_v2.csv")
    save_csv(test_df_results, output_dir / "model_results_table7_v2.csv")
    save_csv(selected_predictions,
             output_dir / "actual_vs_predicted_selected_model_v2.csv")

    # Optional research artifact: only unpickle files you created and trust.
    (output_dir / "selected_model_v2.pkl").write_bytes(selected_pickle)
    print("Saved:", output_dir / "selected_model_v2.pkl")

    device_info = get_device_info()
    report = {
        "script_version": "2.0",
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(Path(args.input).resolve()),
        "preprocessing": "hourly_mean_no_imputation_full_hour_grid_lag2_rolling3_t_plus_1h",
        "missing_value_handling": "drop incomplete supervised windows after feature/target construction",
        "outlier_filtering": "none (no unverified sensor values removed)",
        "selection_metric": "validation_RMSE",
        "refit_strategy": "all_available_pretest_labels_purged_at_test_boundary",
        "inference_timing": "warmup_3_then_{}x_full_test_batch_divided_by_{}_samples".format(
            args.repeats, len(X_test)
        ),
        "model_size_unit": "KiB = pickle byte count / 1024",
        "pickle_protocol": pickle.DEFAULT_PROTOCOL,
        "selected_model": selected_name,
        "raw_data": raw_audit,
        "hourly_and_features": hourly_audit,
        "splits": split_audit,
        "device_and_packages": device_info,
    }
    metadata_path = output_dir / "run_metadata_v2.json"
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print("Saved:", metadata_path)

    if not device_info["hardware_model_from_os"] or (
        "Raspberry Pi" not in device_info["hardware_model_from_os"]
    ):
        print("\nWARNING: OS did not identify this host as a Raspberry Pi. ")
        print("Use numbers from a run on your Raspberry Pi 4 in the manuscript.")
    print("\nSelected model (based on VALIDATION RMSE):", selected_name)
    print("All outputs are *_v2; original result files remain unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/raw/smartfarm_soil_data.csv",
        help="Path to the original sensor-data CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Where to save v2 outputs (default: %(default)s)",
    )
    parser.add_argument(
        "--repeats", type=int, default=100,
        help="Number of repeated batch inference runs (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    run(args)


if __name__ == "__main__":
    main()
