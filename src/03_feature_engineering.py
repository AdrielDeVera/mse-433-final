"""
03_feature_engineering.py
══════════════════════════════════════════════════════════════════════════
DESIGN NOTE — Synthetic Time-Series Expansion
──────────────────────────────────────────────
The cleaned dataset is a cross-sectional snapshot (one row per SKU).
To build a proper SKU-day feature matrix with lag/rolling features we
expand each SKU into a 60-day synthetic daily demand series anchored at
`last_restock_date`, using:

    demand_t  ~  max(0, N(daily_demand, demand_std_dev))

Each SKU is seeded independently via hash(item_id) so results are fully
reproducible without a global RNG state.

60 days gives 46 valid rows per SKU after dropping the leading 14 NaN
rows from the widest rolling/lag window — ~131 K rows total, manageable
for modelling pipelines.

INTERMITTENCY CLASSIFICATION (Syntetos–Boylan / Croston framework)
──────────────────────────────────────────────────────────────────
ADI  = Average Demand Interval  ≈  reorder_frequency_days
       (days between replenishment orders — the operative "demand event"
        in a warehouse replenishment context)

CV²  = (demand_std_dev / daily_demand)²
       (squared coefficient of variation of non-zero demand)

Thresholds  (Syntetos & Boylan 2005):
  Smooth      : ADI <  1.32  and  CV² < 0.49
  Erratic     : ADI <  1.32  and  CV² ≥ 0.49
  Intermittent: ADI ≥  1.32  and  CV² < 0.49
  Lumpy       : ADI ≥  1.32  and  CV² ≥ 0.49
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
CLEAN_CSV   = ROOT / "data" / "cleaned" / "daily_demand.csv"
FEATURE_CSV = ROOT / "data" / "cleaned" / "features.csv"
SKU_CLF_CSV = ROOT / "outputs"          / "sku_intermittency.csv"

SKU_CLF_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
HISTORY_DAYS   = 60   # synthetic window per SKU
LAG_DAYS       = [1, 3, 7, 14]
ROLL_WINDOWS   = [7, 14]
ADI_THRESHOLD  = 1.32
CV2_THRESHOLD  = 0.49

print("=" * 70)
print("  FEATURE ENGINEERING  |  03_feature_engineering.py")
print("=" * 70)

# ════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load cleaned data
# ════════════════════════════════════════════════════════════════════════════
df_sku = pd.read_csv(CLEAN_CSV, parse_dates=["last_restock_date"])
print(f"\n[LOAD]  Cleaned SKUs: {len(df_sku):,}  |  Columns: {df_sku.shape[1]}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Per-SKU intermittency classification
#           (computed from raw parameters BEFORE expansion)
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] Computing intermittency metrics (ADI, CV²) ...")

df_sku = df_sku.copy()
df_sku["ADI"] = df_sku["reorder_frequency_days"].astype(float)
df_sku["CV2"] = (df_sku["demand_std_dev"] / df_sku["daily_demand"]) ** 2

def classify_intermittency(adi: float, cv2: float) -> str:
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Smooth"
    elif adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "Erratic"
    elif adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Intermittent"
    else:
        return "Lumpy"

df_sku["intermittency_class"] = df_sku.apply(
    lambda r: classify_intermittency(r["ADI"], r["CV2"]), axis=1
)

# SKU classification table (save now — does not depend on expansion)
sku_clf = df_sku[
    ["item_id", "category", "daily_demand", "demand_std_dev",
     "lead_time_days", "reorder_frequency_days", "ADI", "CV2",
     "intermittency_class"]
].copy()
sku_clf.to_csv(SKU_CLF_CSV, index=False)
print(f"   Saved SKU classification → {SKU_CLF_CSV.relative_to(ROOT)}")

print("\n   Intermittency class counts:")
class_counts = df_sku["intermittency_class"].value_counts()
for cls, cnt in class_counts.items():
    pct = cnt / len(df_sku) * 100
    print(f"   {'':2}{cls:<15s}  {cnt:>5,}  ({pct:5.1f}%)")

print(f"\n   ADI range  : {df_sku['ADI'].min():.1f} – {df_sku['ADI'].max():.1f}")
print(f"   CV² range  : {df_sku['CV2'].min():.4f} – {df_sku['CV2'].max():.4f}")
print(f"   CV² median : {df_sku['CV2'].median():.4f}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Synthetic daily time-series expansion (60 days per SKU)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n[STEP 3] Expanding {len(df_sku):,} SKUs × {HISTORY_DAYS} days ...")

def sku_seed(item_id: str) -> int:
    """Deterministic int seed from item_id string."""
    return int(hashlib.md5(item_id.encode()).hexdigest(), 16) % (2**31)

records = []
for _, row in df_sku.iterrows():
    rng   = np.random.default_rng(sku_seed(row["item_id"]))
    noise = rng.normal(0.0, row["demand_std_dev"], HISTORY_DAYS)
    demand_series = np.clip(row["daily_demand"] + noise, 0.0, None)

    end_date   = row["last_restock_date"]
    start_date = end_date - pd.Timedelta(days=HISTORY_DAYS - 1)
    dates      = pd.date_range(start=start_date, end=end_date, freq="D")

    for date, demand in zip(dates, demand_series):
        records.append({
            "item_id"  : row["item_id"],
            "category" : row["category"],
            "date"     : date,
            "demand"   : demand,
            # pass-through SKU scalars needed later
            "_daily_demand"        : row["daily_demand"],
            "_demand_std_dev"      : row["demand_std_dev"],
            "_lead_time_days"      : row["lead_time_days"],
            "_reorder_freq_days"   : row["reorder_frequency_days"],
            "_ADI"                 : row["ADI"],
            "_CV2"                 : row["CV2"],
            "_intermittency_class" : row["intermittency_class"],
        })

panel = pd.DataFrame(records).sort_values(["item_id", "date"]).reset_index(drop=True)
print(f"   Panel shape before feature computation: {panel.shape}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 4 — Lag features (within each SKU group)
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] Computing lag features ...")
panel = panel.sort_values(["item_id", "date"]).reset_index(drop=True)

for lag in LAG_DAYS:
    panel[f"lag_{lag}"] = (
        panel.groupby("item_id")["demand"]
        .shift(lag)
    )
    print(f"   lag_{lag:>2d} computed")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Rolling mean, std, and CV (shifted by 1 to prevent leakage)
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Computing rolling features (shift=1 to prevent leakage) ...")

for w in ROLL_WINDOWS:
    shifted = panel.groupby("item_id")["demand"].shift(1)

    rolling_mean = (
        shifted.groupby(panel["item_id"])
        .transform(lambda s: s.rolling(w, min_periods=w).mean())
    )
    rolling_std = (
        shifted.groupby(panel["item_id"])
        .transform(lambda s: s.rolling(w, min_periods=w).std())
    )

    panel[f"rolling_mean_{w}"] = rolling_mean
    panel[f"rolling_std_{w}"]  = rolling_std
    print(f"   rolling_mean_{w}, rolling_std_{w} computed")

# Rolling CV (14-day only, as specified)
panel["rolling_cv_14"] = panel["rolling_std_14"] / panel["rolling_mean_14"].replace(0, np.nan)
print("   rolling_cv_14 computed")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Calendar features
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Computing calendar features ...")
panel["day_of_week"]  = panel["date"].dt.dayofweek          # 0=Mon, 6=Sun
panel["week_of_year"] = panel["date"].dt.isocalendar().week.astype(int)
panel["month"]        = panel["date"].dt.month
panel["is_weekend"]   = (panel["day_of_week"] >= 5).astype(int)
print("   day_of_week, week_of_year, month, is_weekend computed")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 7 — Per-SKU statistics (from original parameters)
# ════════════════════════════════════════════════════════════════════════════
print("\n[STEP 7] Attaching per-SKU statistics ...")
panel.rename(columns={
    "_daily_demand"        : "sku_mean_demand",
    "_demand_std_dev"      : "sku_std_demand",
    "_lead_time_days"      : "sku_mean_lead_time",
    "_reorder_freq_days"   : "sku_reorder_freq_days",
    "_ADI"                 : "ADI",
    "_CV2"                 : "CV2",
    "_intermittency_class" : "intermittency_class",
}, inplace=True)
# std of lead time is fixed per SKU (single observed value), so = 0
panel["sku_std_lead_time"] = 0.0
print("   sku_mean_demand, sku_std_demand, sku_mean_lead_time, "
      "sku_std_lead_time attached")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 8 — Drop rows with NaN from lag/rolling features
# ════════════════════════════════════════════════════════════════════════════
lag_roll_cols = (
    [f"lag_{l}" for l in LAG_DAYS]
    + [f"rolling_mean_{w}" for w in ROLL_WINDOWS]
    + [f"rolling_std_{w}"  for w in ROLL_WINDOWS]
    + ["rolling_cv_14"]
)
rows_before = len(panel)
panel.dropna(subset=lag_roll_cols, inplace=True)
panel.reset_index(drop=True, inplace=True)
rows_after = len(panel)
print(f"\n[STEP 8] Dropped {rows_before - rows_after:,} NaN rows "
      f"(leading lag/rolling windows).")
print(f"         Rows retained: {rows_after:,} / {rows_before:,}")

# ════════════════════════════════════════════════════════════════════════════
#  STEP 9 — Final column ordering & save
# ════════════════════════════════════════════════════════════════════════════
feature_cols = [
    # identifiers
    "item_id", "category", "date", "demand",
    # lag features
    "lag_1", "lag_3", "lag_7", "lag_14",
    # rolling mean/std
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    # rolling CV
    "rolling_cv_14",
    # calendar
    "day_of_week", "week_of_year", "month", "is_weekend",
    # per-SKU stats
    "sku_mean_demand", "sku_std_demand",
    "sku_mean_lead_time", "sku_std_lead_time", "sku_reorder_freq_days",
    # intermittency
    "ADI", "CV2", "intermittency_class",
]
panel = panel[feature_cols]

print("\n[STEP 9] Final feature matrix:")
print(f"   Rows    : {len(panel):,}")
print(f"   Columns : {len(panel.columns)}")
print(f"   SKUs    : {panel['item_id'].nunique():,}")
print(f"   Date range: {panel['date'].min().date()} → {panel['date'].max().date()}")

panel.to_csv(FEATURE_CSV, index=False)
print(f"\n[SAVE]  Feature matrix → {FEATURE_CSV.relative_to(ROOT)}")
print(f"[SAVE]  SKU class table → {SKU_CLF_CSV.relative_to(ROOT)}")

# ════════════════════════════════════════════════════════════════════════════
#  SUMMARY REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n  Feature matrix shape : {panel.shape}")
print(f"\n  Intermittency class distribution:")
for cls, cnt in panel.groupby("intermittency_class")["item_id"].nunique().items():
    pct = cnt / panel["item_id"].nunique() * 100
    print(f"    {cls:<15s}  {cnt:>5,} SKUs  ({pct:5.1f}%)")

print(f"\n  Feature columns ({len(feature_cols)}):")
for i, col in enumerate(feature_cols, 1):
    print(f"    {i:>2}. {col}")

print(f"\n  NaN check (should be 0):")
nan_totals = panel.isnull().sum()
problem_cols = nan_totals[nan_totals > 0]
if len(problem_cols) == 0:
    print("    ✓ No NaN values in final feature matrix.")
else:
    for col, n in problem_cols.items():
        print(f"    ✗ {col}: {n} NaN")

print("\n  Sample rows:")
print(panel[["item_id", "date", "demand", "lag_1", "lag_7",
             "rolling_mean_14", "rolling_cv_14",
             "intermittency_class"]].head(10).to_string(index=False))
print("=" * 70)
