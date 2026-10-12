import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# Paths
# =========================
input_path = Path("results/model_results_table7.csv")
output_dir = Path("figures")
output_dir.mkdir(parents=True, exist_ok=True)

output_path_png = output_dir / "table7_model_comparison_bar_chart.png"
output_path_pdf = output_dir / "table7_model_comparison_bar_chart.pdf"

# =========================
# Load data
# =========================
df = pd.read_csv(input_path)

# Optional: rename column if needed
df.columns = [c.strip() for c in df.columns]

# Sort by RMSE ascending for clearer comparison
df = df.sort_values("RMSE").reset_index(drop=True)

models = df["Model"]
rmse = df["RMSE"]
r2 = df["R2"]

# =========================
# Plot
# =========================
fig, axes = plt.subplots(2, 1, figsize=(12, 9))

# ---- Top chart: RMSE ----
axes[0].bar(models, rmse)
axes[0].set_title("Comparison of model RMSE values")
axes[0].set_ylabel("RMSE")
axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

for i, v in enumerate(rmse):
    axes[0].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

# ---- Bottom chart: R² ----
axes[1].bar(models, r2)
axes[1].set_title("Comparison of model R² values")
axes[1].set_ylabel("R²")
axes[1].set_xlabel("Model")
axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

for i, v in enumerate(r2):
    axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

# Rotate model names for readability
for ax in axes:
    ax.tick_params(axis="x", rotation=20)

plt.tight_layout()

# Save
plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
plt.savefig(output_path_pdf, bbox_inches="tight")
plt.show()

print(f"Saved PNG: {output_path_png}")
print(f"Saved PDF: {output_path_pdf}")
