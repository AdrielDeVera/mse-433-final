#!/usr/bin/env python3
"""
run_pipeline.py — End-to-end demand forecasting & inventory optimisation pipeline
==================================================================================

Usage
-----
    python run_pipeline.py --data path/to/new_data.csv

Workflow for warehouse managers
-------------------------------
Warehouse managers export a fresh CSV from their Warehouse Management System
(WMS) on a weekly basis, place it in ``data/raw/``, and run this script to
obtain updated 14-day demand forecasts and reorder recommendations.  The
script executes the full pipeline (cleaning → feature engineering → forecast
→ inventory optimisation) and writes the final ``recommendations.csv`` to
``outputs/``.  No intermediate manual steps are required.

Steps executed
--------------
  02  Clean raw data and generate EDA plots
  03  Build SKU-day feature matrix with lag/rolling/calendar features
  04  Train Naive, MA-7, XGBoost (hist-GBT) & LightGBM models;
      evaluate with recursive multi-step forecasting
  05  Compute dynamic inventory policy (safety stock, ROP, EOQ)
      and compare against a static baseline
"""

import argparse
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"

STEPS = [
    ("02_clean_and_eda.py",        "Cleaning & EDA"),
    ("03_feature_engineering.py",  "Feature engineering"),
    ("04_forecast_model.py",       "Forecast modelling"),
    ("05_optimisation.py",         "Inventory optimisation"),
]


def run_step(script_name: str, label: str, step_no: int, total: int) -> float:
    """Run a single pipeline step and return elapsed seconds."""
    path = SRC_DIR / script_name
    if not path.exists():
        print(f"\n  ERROR: {path} not found — aborting.")
        sys.exit(1)

    header = f"  STEP {step_no}/{total} — {label}  ({script_name})"
    print("\n" + "─" * 70)
    print(header)
    print("─" * 70)

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED: {script_name} exited with code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n  Completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Run the full demand-forecasting & inventory-optimisation pipeline."
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the raw WMS-exported CSV file (e.g. data/raw/logistics_dataset.csv)",
    )
    args = parser.parse_args()

    # ── resolve and validate input file ───────────────────────────────────
    src_csv = Path(args.data).resolve()
    if not src_csv.exists():
        print(f"ERROR: Data file not found: {src_csv}")
        sys.exit(1)

    # Copy to canonical location expected by the scripts
    canonical = RAW_DIR / "logistics_dataset.csv"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if src_csv != canonical:
        shutil.copy2(src_csv, canonical)
        print(f"  Copied {src_csv.name} → {canonical.relative_to(ROOT)}")
    else:
        print(f"  Using {canonical.relative_to(ROOT)}")

    # ── run pipeline ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  MSE 433 — DEMAND FORECASTING & INVENTORY OPTIMISATION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    total_time = 0.0
    for i, (script, label) in enumerate(STEPS, 1):
        total_time += run_step(script, label, i, len(STEPS))

    # ── completion summary ────────────────────────────────────────────────
    rec_path = OUT_DIR / "recommendations.csv"
    rec_exists = rec_path.exists()

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"""
  Timestamp    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  Total runtime: {total_time:.1f}s ({total_time/60:.1f} min)
  Input data   : {src_csv.name}

  Key outputs:
    outputs/recommendations.csv  {'✓ saved' if rec_exists else '✗ NOT found'}
    outputs/best_model.joblib    {'✓' if (OUT_DIR/'best_model.joblib').exists() else '✗'}
    outputs/model_comparison.csv {'✓' if (OUT_DIR/'model_comparison.csv').exists() else '✗'}
    outputs/eda/                 {'✓' if (OUT_DIR/'eda').exists() else '✗'}
    outputs/error_analysis/      {'✓' if (OUT_DIR/'error_analysis').exists() else '✗'}

  Next steps:
    • Review recommendations:  cat outputs/recommendations.csv | head
    • Launch dashboard:        streamlit run dashboard/app.py
""")
    print("=" * 70)


if __name__ == "__main__":
    main()
