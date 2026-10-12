import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Paths
# =========================
base_dir = Path(".")
input_path = base_dir / "results" / "actual_vs_predicted_selected_model.csv"

if not input_path.exists():
    input_path = base_dir / "actual_vs_predicted_selected_model.csv"

output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "figure5_actual_vs_predicted_publication.png"
pdf_path = output_dir / "figure5_actual_vs_predicted_publication.pdf"
tiff_path = output_dir / "figure5_actual_vs_predicted_publication.tiff"

# =========================
# Load data
# =========================
df = pd.read_csv(input_path)
df.columns = [c.strip() for c in df.columns]

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

actual = df["actual_soil_moisture"].astype(float)
predicted = df["predicted_soil_moisture"].astype(float)

# =========================
# Metrics
# =========================
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100
r2 = r2_score(actual, predicted)

# =========================
# General figure settings
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# =========================
# Create figure
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(7.2, 7.0),
    gridspec_kw={"height_ratios": [1.25, 1.0]}
)

# =========================
# Panel A: Time-series plot
# =========================
ax1 = axes[0]

ax1.plot(
    df["timestamp"],
    actual,
    linewidth=1.8,
    label="Actual soil moisture"
)

ax1.plot(
    df["timestamp"],
    predicted,
    linewidth=1.8,
    linestyle="--",
    label="Predicted soil moisture"
)

ax1.set_title("(a) One-hour-ahead soil-moisture prediction")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Soil moisture (%)")
ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax1.legend(loc="best", frameon=False)

locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
formatter = mdates.DateFormatter("%d %b\n%H:%M")
ax1.xaxis.set_major_locator(locator)
ax1.xaxis.set_major_formatter(formatter)

# Add metric box
metric_text = (
    f"Ridge Regression\n"
    f"MAE = {mae:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"MAPE = {mape:.2f}%\n"
    f"R² = {r2:.4f}"
)

ax1.text(
    0.02,
    0.95,
    metric_text,
    transform=ax1.transAxes,
    va="top",
    ha="left",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9)
)

# =========================
# Panel B: Actual vs predicted scatter
# =========================
ax2 = axes[1]

ax2.scatter(
    actual,
    predicted,
    s=22,
    alpha=0.8,
    label="Test samples"
)

min_val = min(actual.min(), predicted.min())
max_val = max(actual.max(), predicted.max())

ax2.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=1.2,
    label="1:1 reference"
)

ax2.set_title("(b) Agreement between actual and predicted values")
ax2.set_xlabel("Actual soil moisture (%)")
ax2.set_ylabel("Predicted soil moisture (%)")
ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax2.legend(loc="best", frameon=False)

# Equal visual range
padding = (max_val - min_val) * 0.08
ax2.set_xlim(min_val - padding, max_val + padding)
ax2.set_ylim(min_val - padding, max_val + padding)

plt.tight_layout()

# =========================
# Save outputs
# =========================
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved Figure 5:")
print(f"PNG : {png_path}")
print(f"PDF : {pdf_path}")
print(f"TIFF: {tiff_path}")
