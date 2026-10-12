import pandas as pd
import matplotlib.pyplot as plt

# File path
csv_path = "results/actual_vs_predicted_selected_model.csv"

# Load data
df = pd.read_csv(csv_path)

# Parse timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Create plot
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["actual_soil_moisture"], label="Actual soil moisture")
plt.plot(df["timestamp"], df["predicted_soil_moisture"], label="Predicted soil moisture")

plt.xlabel("Timestamp")
plt.ylabel("Soil moisture (%)")
plt.title("Actual and predicted one-hour-ahead soil-moisture values using the selected Ridge Regression model")
plt.legend()
plt.tight_layout()

# Save figure
output_path = "figures/figure5_actual_vs_predicted.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Saved figure to: {output_path}")
