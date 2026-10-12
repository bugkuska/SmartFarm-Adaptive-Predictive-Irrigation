import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# =========================
# Paths
# =========================
input_path = Path("results/actual_vs_predicted_selected_model.csv")
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_path_png = output_dir / "figure5_actual_vs_predicted.png"
output_path_pdf = output_dir / "figure5_actual_vs_predicted.pdf"

# =========================
# Load data
# =========================
df = pd.read_csv(input_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# =========================
# Plot
# =========================
plt.figure(figsize=(12, 6))

plt.plot(
    df["timestamp"],
    df["actual_soil_moisture"],
    linewidth=2,
    label="Actual soil moisture"
)

plt.plot(
    df["timestamp"],
    df["predicted_soil_moisture"],
    linewidth=2,
    linestyle="--",
    label="Predicted soil moisture"
)

plt.xlabel("Timestamp")
plt.ylabel("Soil moisture (%)")
plt.title("Actual and predicted one-hour-ahead soil-moisture values\nusing the selected Ridge Regression model")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)

# Date formatting
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))

plt.xticks(rotation=0)
plt.tight_layout()

# Save
plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
plt.savefig(output_path_pdf, bbox_inches="tight")
plt.show()

print(f"Saved PNG: {output_path_png}")
print(f"Saved PDF: {output_path_pdf}")
