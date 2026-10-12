import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# =========================
# Paths
# =========================
base_dir = Path(".")
input_path = base_dir / "results" / "model_results_table7.csv"

if not input_path.exists():
    input_path = base_dir / "model_results_table7.csv"

output_dir = base_dir / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

png_path = output_dir / "figure6_model_comparison_publication.png"
pdf_path = output_dir / "figure6_model_comparison_publication.pdf"
tiff_path = output_dir / "figure6_model_comparison_publication.tiff"

# =========================
# Load data
# =========================
df = pd.read_csv(input_path)
df.columns = [c.strip() for c in df.columns]

# Make sure expected columns exist
required_cols = [
    "Model",
    "MAE",
    "RMSE",
    "MAPE (%)",
    "R2",
    "Inference time (ms/sample)",
    "Model size (KB)"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in result file: {missing}")

# Sort by RMSE for consistent model order
df = df.sort_values("RMSE").reset_index(drop=True)

# Short labels for plot readability
def wrap_label(text, width=14):
    return "\n".join(textwrap.wrap(str(text), width=width))

df["Model_label"] = df["Model"].apply(lambda x: wrap_label(x, width=14))

# =========================
# General figure settings
# =========================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "figure.dpi": 300
})

# =========================
# Create figure
# =========================
fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(8.2, 6.8)
)

axes = axes.flatten()

metrics = [
    {
        "col": "MAE",
        "title": "(a) MAE comparison",
        "ylabel": "MAE",
        "format": "{:.4f}",
        "lower_is_better": True
    },
    {
        "col": "RMSE",
        "title": "(b) RMSE comparison",
        "ylabel": "RMSE",
        "format": "{:.4f}",
        "lower_is_better": True
    },
    {
        "col": "R2",
        "title": "(c) R² comparison",
        "ylabel": "R²",
        "format": "{:.4f}",
        "lower_is_better": False
    },
    {
        "col": "Model size (KB)",
        "title": "(d) Model size comparison",
        "ylabel": "Model size (KB)",
        "format": "{:.2f}",
        "lower_is_better": True
    }
]

x = range(len(df))

for ax, metric in zip(axes, metrics):
    values = df[metric["col"]].astype(float)

    bars = ax.bar(x, values)

    ax.set_title(metric["title"])
    ax.set_ylabel(metric["ylabel"])
    ax.set_xticks(x)
    ax.set_xticklabels(df["Model_label"])
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # Add value labels
    for i, v in enumerate(values):
        ax.text(
            i,
            v,
            metric["format"].format(v),
            ha="center",
            va="bottom",
            fontsize=8
        )

    # Mark best model per metric with an asterisk above value
    if metric["lower_is_better"]:
        best_idx = values.idxmin()
    else:
        best_idx = values.idxmax()

    best_pos = list(df.index).index(best_idx)
    best_value = values.iloc[best_pos]

    ax.text(
        best_pos,
        best_value,
        "*",
        ha="center",
        va="top",
        fontsize=14
    )

# Use log scale for model size if values are highly different
if df["Model size (KB)"].max() / max(df["Model size (KB)"].min(), 1e-9) > 50:
    axes[3].set_yscale("log")
    axes[3].set_ylabel("Model size (KB, log scale)")

fig.suptitle(
    "Model comparison for one-hour-ahead soil-moisture prediction",
    fontsize=12,
    y=1.02
)

fig.text(
    0.5,
    0.005,
    "Asterisk indicates the best model for each metric. Models are ordered by RMSE.",
    ha="center",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.03, 1, 1])

# =========================
# Save outputs
# =========================
fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(tiff_path, dpi=300, bbox_inches="tight")

plt.close(fig)

print("Saved Figure 6:")
print(f"PNG : {png_path}")
print(f"PDF : {pdf_path}")
print(f"TIFF: {tiff_path}")
