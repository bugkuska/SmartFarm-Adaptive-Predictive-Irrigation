## SmartFarm Adaptive Predictive Irrigation
## Research companion repository — experimental Version 2 (V2)
This repository contains the soil-moisture dataset, Python source code, evaluation outputs, and figures associated with the manuscript Adaptive Predictive Smart Irrigation Framework Using Edge AI and Workflow Automation, prepared for submission to Precision Agriculture (Springer Nature).

The evaluated scope is one-hour-ahead soil-moisture prediction and model-level edge-inference feasibility. Sensor acquisition and historical data storage used an ESP32-based sensing system, n8n-supported telemetry ingestion, and InfluxDB. The predictive models were evaluated offline using exported historical measurements. Continuous prediction-to-workflow integration, irrigation decision execution, closed-loop actuation, water savings, and agronomic outcomes were not experimentally evaluated.

## Repository structure

text
SmartFarm-Adaptive-Predictive-Irrigation/
├── data/
│   └── raw/
│       └── smartfarm_soil_data_v2.csv
├── src/
│   └── run_soil_prediction_v2.py
├── scripts/
│   └── plot_figure2_actual_vs_predicted_v2.py
├── results/
│   └── V2/
│       ├── actual_vs_predicted_selected_model_v2.csv
│       ├── model_comparison_test_v2.csv
│       ├── validation_metrics_v2.csv
│       ├── run_metadata_v2.json
│       └── selected_model_v2.pkl
├── figures/
│   ├── Fig1_framework_scope.pdf
│   └── Fig2_actual_vs_predicted_v2.pdf
└── README.md


This tree describes the intended release layout. Verify that each listed file has been uploaded before making the repository public. The filenames smartfarm_soil_data_v2.csv and model_comparison_test_v2.csv are publication-oriented renames; renaming must not alter the underlying data or reported values.

## Dataset and experimental design

- Testbed: closed greenhouse used for grape cultivation.
- Data collection: 7–29 September 2025; four soil-moisture channels (soil1–soil4).
- Raw records: 492.
- Hourly timeline: 529 hourly slots, including 475 observed and 54 missing intervals; missing values were not imputed.
- Supervised samples: 466, with 18 predictor features derived from soil-moisture observations and temporal information.
- Prediction target: mean moisture across the four sensors exactly one hour ahead.
- Model selection: chronological split; 325 training and 69 validation samples after boundary purging. The independent test set contains 70 samples. All five candidate models were refitted on 395 eligible pretest samples for final comparison.
- Algorithms: Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and Support Vector Regression (SVR).
- Selection criterion: lowest validation RMSE. SVR was selected without inspecting test-set performance.

## Reproducing the experiment

Run commands from the repository root. For the original Raspberry Pi benchmark, the recorded environment was Raspberry Pi 4 Model B Rev 1.4, 64-bit Linux (aarch64), Python 3.13.5, NumPy 2.2.4, pandas 2.2.3, and scikit-learn 1.4.2. The figure script additionally requires Matplotlib. Results, especially timing, may vary on other hardware or software versions.

1. Set up a Python environment and install dependencies:

   bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install numpy==2.2.4 pandas==2.2.3 scikit-learn==1.4.2 matplotlib
   

2. Run preprocessing, model selection, refitting, test evaluation, and the inference benchmark:

   bash
   python src/run_soil_prediction_v2.py \
     --input data/raw/smartfarm_soil_data_v2.csv \
     --output-dir results
   

3. Generate the selected-model prediction figure:

   bash
   python scripts/plot_figure2_actual_vs_predicted_v2.py \
     --input results/actual_vs_predicted_selected_model_v2.csv \
     --output-dir figures
   
## Main results (held-out test set)

| Model | MAE | RMSE | MAPE (%) | R² | Batch-derived inference time (ms/sample) | Serialized size (KiB) |
|---|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.7508 | 1.4723 | 2.4381 | 0.2535 | 0.03046 | 1.00 |
| Ridge Regression | 0.7039 | 1.4472 | 2.2901 | 0.2787 | 0.03045 | 0.87 |
| Random Forest | 0.5336 | 1.5289 | 1.6791 | 0.1950 | 0.35575 | 1031.00 |
| Gradient Boosting | 0.5727 | 1.5159 | 1.8314 | 0.2086 | 0.04968 | 246.33 |
| SVR (validation-selected) | 0.5094 | 1.5844 | 1.5661 | 0.1355 | 0.07311 | 35.59 |

SVR had the lowest validation RMSE (1.0642) and was selected according to the prespecified criterion. Ridge Regression, not SVR, achieved the lowest test RMSE. The measured times are averages computed from 100 repeated predictions of the complete 70-sample test batch after three warm-up runs, divided by the number of samples; they are not single-request or end-to-end sensor-to-decision latencies. Serialized sizes are measured in KiB (bytes / 1024).

## Files and interpretation

- validation_metrics_v2.csv: validation-set comparison used for model selection.
- model_comparison_test_v2.csv: independently held-out test metrics and Raspberry Pi model-level measurements.
- actual_vs_predicted_selected_model_v2.csv: SVR test observations and predictions used in Figure 2.
- run_metadata_v2.json: data audit, split sizes, model-selection rule, timing methodology, and recorded execution environment.
- selected_model_v2.pkl: serialized selected SVR pipeline; never unpickle a file from an untrusted source.

## Scope and limitations

The dataset represents one greenhouse and one relatively short observation period. Environmental variables and irrigation-event records were not included in the evaluated predictor data. Test-set R² values were modest, and no repeated seasonal or external-site validation was conducted. The Raspberry Pi assessment covers batch-derived model inference and serialized size only; it does not establish continuous deployment reliability or irrigation efficiency.

## Manuscript and citation

Manuscript: Adaptive Predictive Smart Irrigation Framework Using Edge AI and Workflow Automation (under preparation for submission to Precision Agriculture). Bibliographic publication metadata and a DOI should be added only when available.

## Release and security checklist

Before public release, confirm that the published data, scripts, and V2 results correspond to the manuscript; document any output filename changes; verify the reproduction instructions on the target machine; and inspect all current files and Git history for credentials, API tokens, passwords, private endpoints, and other sensitive configuration. Do not publish raw n8n workflow exports without sanitization.
