"""
01_data_audit.py
────────────────
Loads the raw logistics warehouse CSV and produces a plain-text audit summary.
Outputs are printed to stdout AND saved to /outputs/data_audit.txt.
"""

import os, sys, textwrap
from pathlib import Path
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV      = PROJECT_ROOT / "data" / "raw" / "logistics_dataset.csv"
OUTPUT_TXT   = PROJECT_ROOT / "outputs" / "data_audit.txt"

# ── load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)

lines: list[str] = []

def section(title: str):
    lines.append("")
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)

def out(text: str = ""):
    lines.append(text)

# ── 1. Basic shape, dtypes, columns, head(10) ─────────────────────────────
section("1. BASIC DATASET INFO")
out(f"Rows   : {df.shape[0]:,}")
out(f"Columns: {df.shape[1]}")
out()
out("Column names:")
for i, col in enumerate(df.columns, 1):
    out(f"  {i:>2}. {col}")
out()
out("Data types:")
for col in df.columns:
    out(f"  {col:<40s} {str(df[col].dtype)}")
out()
out("First 10 rows:")
out(df.head(10).to_string(index=False))

# ── 2. Missing values ─────────────────────────────────────────────────────
section("2. MISSING VALUES")
miss_count = df.isnull().sum()
miss_pct   = (df.isnull().mean() * 100).round(2)
miss_df    = pd.DataFrame({"missing_count": miss_count, "missing_pct": miss_pct})
miss_df    = miss_df.sort_values("missing_count", ascending=False)

if miss_df["missing_count"].sum() == 0:
    out("No missing values detected in any column.")
else:
    out(f"{'Column':<40s} {'Count':>8s} {'Pct (%)':>8s}")
    out("-" * 58)
    for col, row in miss_df.iterrows():
        out(f"{col:<40s} {int(row.missing_count):>8,} {row.missing_pct:>8.2f}")
out()
out(f"Total missing cells : {int(miss_count.sum()):,}")
out(f"Total cells         : {df.shape[0] * df.shape[1]:,}")

# ── 3. SKU / date / order volume stats ────────────────────────────────────
section("3. KEY BUSINESS METRICS")

# Unique SKUs (item_id)
sku_col = "item_id" if "item_id" in df.columns else None
if sku_col:
    out(f"Unique SKUs (item_id): {df[sku_col].nunique():,}")
else:
    out("No 'item_id' column found — cannot count SKUs.")

# Date range (last_restock_date)
date_col = "last_restock_date" if "last_restock_date" in df.columns else None
if date_col:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    out(f"Date column          : {date_col}")
    out(f"  Earliest date      : {dates.min()}")
    out(f"  Latest date        : {dates.max()}")
    out(f"  Date span          : {(dates.max() - dates.min()).days} days")
    out(f"  Unparseable dates  : {dates.isna().sum()}")
else:
    out("No date column found.")

# Order volume stats (total_orders_last_month)
order_col = "total_orders_last_month" if "total_orders_last_month" in df.columns else None
if order_col:
    out()
    out(f"Order volume stats ({order_col}):")
    desc = df[order_col].describe()
    for stat in desc.index:
        out(f"  {stat:<10s}: {desc[stat]:>12,.2f}")
    out(f"  total     : {df[order_col].sum():>12,.0f}")

# Daily demand stats
demand_col = "daily_demand" if "daily_demand" in df.columns else None
if demand_col:
    out()
    out(f"Daily demand stats ({demand_col}):")
    desc = df[demand_col].describe()
    for stat in desc.index:
        out(f"  {stat:<10s}: {desc[stat]:>12,.2f}")

# Category breakdown
cat_col = "category" if "category" in df.columns else None
if cat_col:
    out()
    out("Category distribution:")
    for cat, cnt in df[cat_col].value_counts().items():
        out(f"  {cat:<25s} {cnt:>6,} ({cnt/len(df)*100:.1f}%)")

# ── write to file ─────────────────────────────────────────────────────────
OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
report = "\n".join(lines)
OUTPUT_TXT.write_text(report)

print(report)
print(f"\n✓ Audit summary saved to {OUTPUT_TXT.relative_to(PROJECT_ROOT)}")
