# MSE 433 Final Project — Demand Forecasting & Inventory Optimisation

An end-to-end logistics pipeline that transforms raw warehouse management system (WMS) exports into actionable reorder recommendations. The system generates 14-day SKU-level demand forecasts using a from-scratch histogram-based gradient boosted tree model, then computes dynamic inventory policies (safety stock, reorder point, economic order quantity) that reduce total annual inventory costs by approximately 46% compared to a static baseline.

## Folder Structure

```
433 final project/
├── run_pipeline.py              # One-command pipeline runner
├── requirements.txt             # Python dependencies
├── README.md
├── data/
│   ├── raw/                     # Raw WMS CSV exports go here
│   │   └── logistics_dataset.csv
│   └── cleaned/
│       ├── daily_demand.csv     # Cleaned SKU data (2,846 SKUs)
│       └── features.csv         # Model-ready feature matrix (130,916 rows)
├── src/
│   ├── 01_data_audit.py         # Load & audit raw CSV
│   ├── 02_clean_and_eda.py      # Clean data, generate 5 EDA plots
│   ├── 03_feature_engineering.py # Build SKU-day features + intermittency classification
│   ├── 04_forecast_model.py     # Train & evaluate forecast models (hist-GBT)
│   └── 05_optimisation.py       # Dynamic inventory policy + cost comparison
├── dashboard/
│   ├── app.py                   # Streamlit interactive dashboard
│   └── index.html               # Standalone HTML dashboard (no server required)
├── outputs/
│   ├── recommendations.csv      # Final reorder recommendations (2,846 SKUs)
│   ├── best_model.joblib        # Trained XGBoost histogram-GBT model
│   ├── model_comparison.csv     # Naive / MA-7 / XGBoost / LightGBM metrics
│   ├── data_audit.txt           # Raw data audit summary
│   ├── sku_intermittency.csv    # Syntetos-Boylan classification per SKU
│   ├── eda/                     # 5 exploratory analysis plots
│   └── error_analysis/          # 4 model diagnostic plots
└── notebooks/                   # (reserved for exploratory notebooks)
```

## Setup

```bash
pip install -r requirements.txt
```

## How to Run the Pipeline

Warehouse managers export a fresh CSV from their WMS weekly, place it in `data/raw/`, and run:

```bash
python run_pipeline.py --data data/raw/logistics_dataset.csv
```

This executes all four pipeline stages in sequence (cleaning, feature engineering, forecasting, optimisation) and writes the final `recommendations.csv` to `outputs/`. The script prints a completion summary with timestamp and a checklist of all generated output files.

To run individual stages:

```bash
python src/01_data_audit.py         # Audit raw data
python src/02_clean_and_eda.py      # Clean + EDA
python src/03_feature_engineering.py # Feature matrix
python src/04_forecast_model.py     # Train models
python src/05_optimisation.py       # Inventory optimisation
```

## How to Launch the Dashboard

**Streamlit (interactive, full-featured):**

```bash
streamlit run dashboard/app.py
```

**Standalone HTML (no install required):**

```bash
cd "433 final project"
python -m http.server 8000
# Open http://localhost:8000/dashboard/index.html
```

## Modelling Approach

**Forecasting:** Each SKU's demand is forecasted 14 days ahead using a recursive multi-step strategy with a histogram-based gradient boosted tree (HistogramGBT) implemented from scratch in NumPy. The model uses 21 features including lagged demand (1, 3, 7, 14 days), rolling statistics (7- and 14-day mean/std/CV), calendar indicators, and per-SKU metadata. Hyperparameters are tuned via randomised search over an expanding-window time-series cross-validation with a 14-day gap to prevent look-ahead leakage.

**Optimisation:** Dynamic inventory policies are computed per SKU using forecast-derived demand statistics and actual lead times from the dataset. Safety stock follows the standard formula incorporating both demand uncertainty and lead-time variability (z = 1.645 for 95% service level). The economic order quantity (EOQ) uses per-SKU holding costs from the WMS data. Policies are benchmarked against a static baseline that uses historical averages with a fixed 7-day demand buffer as safety stock.

**Intermittency:** SKUs are classified into Syntetos-Boylan demand categories (Smooth, Erratic, Intermittent, Lumpy) using average demand interval and squared coefficient of variation thresholds. This classification is used as a model feature and for stratified error analysis.
