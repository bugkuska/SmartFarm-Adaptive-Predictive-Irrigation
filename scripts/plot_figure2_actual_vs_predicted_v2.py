#!/usr/bin/env python3
"""Publication-ready actual vs. predicted figure from Version 2 test predictions.

Run from project root:
    python3 scripts/plot_figure2_actual_vs_predicted_v2.py

Optional:
    python3 scripts/plot_figure2_actual_vs_predicted_v2.py --input results/actual_vs_predicted_selected_model_v2.csv --output-dir figures

The Version 2 CSV 'timestamp' is ALREADY the prediction target timestamp;
do not shift it by one hour again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save files without requiring an X11 display on Raspberry Pi.
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def project_root() -> Path:
    here = Path(__file__).resolve().parent
    return here.parent if here.name == "scripts" else here


def get_input_path(value: str, root: Path) -> Path:
    supplied = Path(value).expanduser()
    attempts = [supplied, root / supplied, Path(__file__).resolve().parent / supplied]
    for path in attempts:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"CSV not found. Tried: {[str(p) for p in attempts]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Figure 2 from selected SVR test predictions (Version 2).")
    parser.add_argument("--input", default="results/actual_vs_predicted_selected_model_v2.csv", help="Path to Version 2 prediction CSV")
    parser.add_argument("--output-dir", default=None, help="Directory to save PNG/PDF/TIFF (default: PROJECT_ROOT/figures)")
    args = parser.parse_args()

    root = project_root()
    input_path = get_input_path(args.input, root)
    output_dir = (Path(args.output_dir).expanduser() if args.output_dir else root / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    required = {"timestamp", "forecast_origin", "actual_soil_moisture", "predicted_soil_moisture", "selected_model"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.empty:
        raise ValueError("CSV is empty")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df["forecast_origin"] = pd.to_datetime(df["forecast_origin"], errors="raise")
    df["actual_soil_moisture"] = pd.to_numeric(df["actual_soil_moisture"], errors="raise")
    df["predicted_soil_moisture"] = pd.to_numeric(df["predicted_soil_moisture"], errors="raise")
    if df[list(required)].isna().any().any():
        raise ValueError("Missing required observations")
    if set(df["selected_model"].astype(str).str.strip()) != {"SVR"}:
        raise ValueError("Expected only SVR test predictions selected by validation RMSE")
    if not ((df["timestamp"] - df["forecast_origin"]) == pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast origin and target timestamp are not exactly one hour apart")
    df = df.sort_values("timestamp").reset_index(drop=True)
    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate target timestamps found")

    actual = df["actual_soil_moisture"].to_numpy(dtype=float)
    predicted = df["predicted_soil_moisture"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite observed or predicted values")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.labelsize": 10,
        "axes.titlesize": 10.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4), gridspec_kw={"height_ratios": [1.1, 1.0]})

    # Panel a: target timestamps are already in the CSV (no +1 hour shift).
    ax1.plot(df["timestamp"], actual, linewidth=1.65, label="Observed")
    ax1.plot(df["timestamp"], predicted, linewidth=1.65, linestyle="--", label="Predicted (SVR)")
    ax1.set_title("(a) One-hour-ahead soil-moisture predictions", loc="left")
    ax1.set_xlabel("Target timestamp")
    ax1.set_ylabel("Mean soil moisture (%)")
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax1.legend(loc="best", frameon=False)
    ax1.grid(linestyle="--", linewidth=0.5, alpha=0.35)

    # Panel b: scatter with a 1:1 reference line.
    ax2.scatter(actual, predicted, s=28, alpha=0.8, label="Test samples")
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    margin = max((high - low) * 0.065, 0.15)
    low, high = low - margin, high + margin
    ax2.plot([low, high], [low, high], linestyle="--", linewidth=1.2, label="1:1 reference")
    ax2.set_xlim(low, high)
    ax2.set_ylim(low, high)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_title("(b) Observed versus predicted soil moisture", loc="left")
    ax2.set_xlabel("Observed mean soil moisture (%)")
    ax2.set_ylabel("Predicted mean soil moisture (%)")
    ax2.legend(loc="best", frameon=False)
    ax2.grid(linestyle="--", linewidth=0.5, alpha=0.35)

    fig.tight_layout(h_pad=1.65)
    output_base = output_dir / "Fig2_actual_vs_predicted_v2"
    for ext in ("png", "pdf", "tiff"):
        fig.savefig(output_base.with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close(fig)

    errors = actual - predicted
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mape = float(np.mean(np.abs(errors / actual)) * 100) if (actual != 0).all() else float("nan")
    r2 = 1 - float(np.sum(errors ** 2) / np.sum((actual - actual.mean()) ** 2))
    print(f"Input: {input_path}")
    print(f"Selected model: SVR; test samples: {len(df)}; horizon: exactly 1 hour")
    print(f"Test MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.4f}%, R2={r2:.4f}")
    print(f"Saved PNG/PDF/TIFF: {output_base}.*")


if __name__ == "__main__":
    main()
