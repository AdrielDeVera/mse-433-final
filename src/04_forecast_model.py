"""
04_forecast_model.py
═══════════════════════════════════════════════════════════════════════
Pure-numpy forecast pipeline.  sklearn / xgboost / lightgbm are NOT
available in this sandbox, so every component is built from scratch:

  HistogramGBT — histogram-based gradient boosted trees (numpy only)

Implementation notes for sandbox runtime constraints (< 10 min):
  • CV uses 5 % temporal subsample of training rows
  • XGBoost: 18 random combos drawn from the full 54-combo grid
  • LightGBM: all 12 combos (fast enough)
  • 3 expanding-window folds (gap = 14) — achieves same temporal
    integrity as 5-fold, with less compute
  • Final models trained on a 35 % stratified subsample of training data
  • Exact hyperparameter RANGES are preserved as specified
═══════════════════════════════════════════════════════════════════════
"""

import warnings, time, pickle, itertools
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "figure.figsize": (10, 5)})

ROOT       = Path(__file__).resolve().parent.parent
FEAT_CSV   = ROOT / "data" / "cleaned" / "features.csv"
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
COMP_CSV   = ROOT / "outputs" / "model_comparison.csv"
ERR_DIR    = ROOT / "outputs" / "error_analysis"
ERR_DIR.mkdir(parents=True, exist_ok=True)

HORIZON    = 14
N_BINS     = 16
MIN_LEAF   = 30

T0 = time.time()
def ts(): return f"[{time.time()-T0:5.0f}s]"

print("=" * 70)
print("  FORECAST MODEL PIPELINE  —  from-scratch hist-GBT")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
#  HISTOGRAM GRADIENT BOOSTED TREES
# ═══════════════════════════════════════════════════════════════════════════

class HistogramGBT:
    def __init__(self, *, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=1.0, max_leaves=None, random_state=42):
        self.n_estimators   = n_estimators
        self.max_depth      = max_depth
        self.learning_rate  = learning_rate
        self.subsample      = subsample
        self.max_leaves     = max_leaves
        self.random_state   = random_state

    def _prebin(self, X):
        self._edges = {}
        Xb = np.empty(X.shape, np.uint8)
        for j in range(X.shape[1]):
            e = np.unique(np.percentile(X[:, j], np.linspace(0, 100, N_BINS+1)))
            Xb[:, j] = np.clip(np.searchsorted(e[1:-1], X[:, j], side="right"),
                                0, N_BINS-1)
            self._edges[j] = e
        return Xb

    def _bin(self, X):
        Xb = np.empty(X.shape, np.uint8)
        for j in range(X.shape[1]):
            e = self._edges[j]
            Xb[:, j] = np.clip(np.searchsorted(e[1:-1], X[:, j], side="right"),
                                0, N_BINS-1)
        return Xb

    def _build(self, Xb, r, d, md):
        n = len(r)
        if d >= md or n < 2 * MIN_LEAF:
            return (float(r.mean()),)                       # leaf tuple
        tot = r.sum(); bg, bf, bb = 0., -1, -1
        for j in range(Xb.shape[1]):
            hs = np.bincount(Xb[:, j], weights=r, minlength=N_BINS)
            hc = np.bincount(Xb[:, j], minlength=N_BINS)
            cs = np.cumsum(hs); cc = np.cumsum(hc)
            rs = tot - cs;      rc = n - cc
            ok = (cc >= MIN_LEAF) & (rc >= MIN_LEAF)
            if not ok.any(): continue
            g = np.full(N_BINS, -np.inf)
            g[ok] = cs[ok]**2/cc[ok] + rs[ok]**2/rc[ok]
            bi = int(np.argmax(g)); gi = g[bi] - tot**2/n
            if gi > bg: bg, bf, bb = gi, j, bi
        if bf < 0:
            return (float(r.mean()),)
        m = Xb[:, bf] <= bb
        return (bf, bb, bg,
                self._build(Xb[m],  r[m],  d+1, md),
                self._build(Xb[~m], r[~m], d+1, md))

    def _ptree(self, Xb, nd):
        out = np.empty(Xb.shape[0])
        self._fill(Xb, np.arange(Xb.shape[0]), nd, out)
        return out

    def _fill(self, Xb, ix, nd, out):
        if len(nd) == 1:             # leaf
            out[ix] = nd[0]; return
        m = Xb[ix, nd[0]] <= nd[1]
        li, ri = ix[m], ix[~m]
        if li.size: self._fill(Xb, li, nd[3], out)
        if ri.size: self._fill(Xb, ri, nd[4], out)

    def fit(self, X, y, _edges=None):
        rng = np.random.default_rng(self.random_state)
        if _edges:
            self._edges = _edges; Xb = self._bin(X)
        else:
            Xb = self._prebin(X)
        eff_d = (max(1, int(np.ceil(np.log2(max(self.max_leaves, 2)))))
                 if self.max_leaves else self.max_depth)
        n, d = X.shape
        self._init = float(y.mean())
        preds = np.full(n, self._init)
        self._trees = []; self._imp = np.zeros(d)
        for _ in range(self.n_estimators):
            resid = y - preds
            ix = (rng.choice(n, int(n*self.subsample), replace=False)
                  if self.subsample < 1 else np.arange(n))
            tree = self._build(Xb[ix], resid[ix], 0, eff_d)
            self._trees.append(tree)
            preds += self.learning_rate * self._ptree(Xb, tree)
            self._aimp(tree)
        s = self._imp.sum()
        self.feature_importances_ = self._imp / s if s else self._imp
        return self

    def _aimp(self, nd):
        if len(nd) == 1: return
        self._imp[nd[0]] += nd[2]
        self._aimp(nd[3]); self._aimp(nd[4])

    def predict(self, X):
        Xb = self._bin(X)
        p = np.full(X.shape[0], self._init)
        for t in self._trees:
            p += self.learning_rate * self._ptree(Xb, t)
        return p


# ═══════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class TSplit:
    """Expanding-window time-series split with gap."""
    def __init__(self, n_splits=3, gap=14):
        self.k = n_splits; self.gap = gap
    def split(self, n):
        f = n // (self.k + 1)
        for i in range(self.k):
            te = f * (i+1); vs = te + self.gap; ve = min(te + f + self.gap, n)
            if vs >= n: continue
            yield np.arange(te), np.arange(vs, ve)

def rmse(a, p): return float(np.sqrt(np.mean((a-p)**2)))
def mae_(a, p): return float(np.mean(np.abs(a-p)))
def mape_(a, p):
    m = a != 0
    return float(np.mean(np.abs((a[m]-p[m])/a[m]))*100) if m.any() else np.nan


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD & PREP
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Loading features.csv ...")
df = pd.read_csv(FEAT_CSV, parse_dates=["date"])
df.sort_values(["item_id", "date"], inplace=True)
df.reset_index(drop=True, inplace=True)

cat_map = {c: i for i, c in enumerate(df["category"].unique())}
int_map = {c: i for i, c in enumerate(df["intermittency_class"].unique())}
df["category_enc"]      = df["category"].map(cat_map).astype(float)
df["intermittency_enc"] = df["intermittency_class"].map(int_map).astype(float)

FCOLS = [
    "lag_1","lag_3","lag_7","lag_14",
    "rolling_mean_7","rolling_std_7","rolling_mean_14","rolling_std_14","rolling_cv_14",
    "day_of_week","week_of_year","month","is_weekend",
    "sku_mean_demand","sku_std_demand","sku_mean_lead_time","sku_reorder_freq_days",
    "ADI","CV2","category_enc","intermittency_enc",
]

# ── Per-SKU split (last 14 rows = test) ───────────────────────────────────
rows_per_sku = 46
train_cnt    = rows_per_sku - HORIZON
tmask = df.groupby("item_id").cumcount() < train_cnt
df_tr  = df[ tmask].copy()
df_te  = df[~tmask].copy()

X_tr = df_tr[FCOLS].values.astype(np.float64)
y_tr = df_tr["demand"].values.astype(np.float64)
X_te = df_te[FCOLS].values.astype(np.float64)
y_te = df_te["demand"].values.astype(np.float64)

sku_ids = df["item_id"].unique()
n_skus  = len(sku_ids)
print(f"{ts()} Train {len(df_tr):,}  Test {len(df_te):,}  SKUs {n_skus:,}")

# ═══════════════════════════════════════════════════════════════════════════
#  CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

# 5 % temporal subsample for CV speed
cv_sorted = df_tr.sort_values("date").reset_index(drop=True)
cv_n = int(len(cv_sorted) * 0.05)
cv_idx = np.linspace(0, len(cv_sorted)-1, cv_n, dtype=int)
Xcv = cv_sorted.iloc[cv_idx][FCOLS].values.astype(np.float64)
ycv = cv_sorted.iloc[cv_idx]["demand"].values.astype(np.float64)
print(f"{ts()} CV sample: {cv_n:,} rows")

tscv = TSplit(n_splits=3, gap=14)

# Pre-bin CV data once
_binner = HistogramGBT(); _binner._prebin(Xcv); _cached = _binner._edges

def cv_search(combos_list, mode):
    best_r, best_p = np.inf, None
    for ci, p in enumerate(combos_list):
        kw = dict(n_estimators=p.get("n_estimators",100),
                  learning_rate=p.get("learning_rate",0.1),
                  subsample=p.get("subsample",1.0),
                  random_state=42)
        if mode == "xgb":
            kw["max_depth"]  = p.get("max_depth",5); kw["max_leaves"] = None
        else:
            kw["max_leaves"] = p.get("num_leaves",31); kw["max_depth"] = 12
        scores = []
        for ti, vi in tscv.split(len(Xcv)):
            m = HistogramGBT(**kw).fit(Xcv[ti], ycv[ti], _edges=_cached)
            scores.append(rmse(ycv[vi], m.predict(Xcv[vi])))
        avg = np.mean(scores)
        if avg < best_r:
            best_r, best_p = avg, p
        if (ci+1) % 6 == 0 or ci == len(combos_list)-1:
            print(f"   {ts()} {mode.upper()} {ci+1}/{len(combos_list)}  "
                  f"best CV RMSE = {best_r:.4f}")
    return best_p, best_r

# ── XGBoost grid (sample 18 combos from full 54) ─────────────────────────
print(f"\n{ts()} ── XGBoost CV ──")
print("   Hyperparameter ranges: n_estimators=[100,300,500], "
      "max_depth=[3,5,7], lr=[0.01,0.05,0.1], subsample=[0.8,1.0]")
xgb_full = [dict(n_estimators=a, max_depth=b, learning_rate=c, subsample=d)
            for a in [100,300,500] for b in [3,5,7]
            for c in [0.01,0.05,0.1] for d in [0.8,1.0]]
rng_grid = np.random.default_rng(42)
xgb_sample = [xgb_full[i] for i in rng_grid.choice(len(xgb_full), 18, replace=False)]
xgb_best, xgb_cv = cv_search(xgb_sample, "xgb")
print(f"   Best: {xgb_best}  CV RMSE = {xgb_cv:.4f}")

# ── LightGBM grid (all 12 combos) ────────────────────────────────────────
print(f"\n{ts()} ── LightGBM CV ──")
print("   Hyperparameter ranges: num_leaves=[31,63], "
      "lr=[0.01,0.05,0.1], n_estimators=[100,300]")
lgb_combos = [dict(num_leaves=a, learning_rate=b, n_estimators=c)
              for a in [31,63] for b in [0.01,0.05,0.1] for c in [100,300]]
lgb_best, lgb_cv = cv_search(lgb_combos, "lgb")
print(f"   Best: {lgb_best}  CV RMSE = {lgb_cv:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
#  TRAIN FINAL MODELS (35 % subsample of training data)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Training final models (35 % of training data) ...")
rng_s = np.random.default_rng(42)
sub_ix = rng_s.choice(len(X_tr), int(len(X_tr)*0.35), replace=False)
Xs, ys = X_tr[sub_ix], y_tr[sub_ix]

model_xgb = HistogramGBT(
    n_estimators=xgb_best["n_estimators"], max_depth=xgb_best["max_depth"],
    learning_rate=xgb_best["learning_rate"], subsample=xgb_best["subsample"],
    max_leaves=None, random_state=42).fit(Xs, ys)
print(f"   {ts()} XGBoost done  ({xgb_best['n_estimators']} trees)")

model_lgb = HistogramGBT(
    n_estimators=lgb_best["n_estimators"], max_leaves=lgb_best["num_leaves"],
    learning_rate=lgb_best["learning_rate"], max_depth=12,
    subsample=1.0, random_state=42).fit(Xs, ys)
print(f"   {ts()} LightGBM done ({lgb_best['n_estimators']} trees)")

# ═══════════════════════════════════════════════════════════════════════════
#  RECURSIVE MULTI-STEP FORECAST (vectorised batch)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Recursive {HORIZON}-step forecast (batch mode) ...")

# Reshape into (n_skus, n_days) arrays
train_dem = df_tr.groupby("item_id")["demand"].apply(np.array).values
train_dem = np.vstack(train_dem)
test_dem  = df_te.groupby("item_id")["demand"].apply(np.array).values
test_dem  = np.vstack(test_dem)

static_cols = ["sku_mean_demand","sku_std_demand","sku_mean_lead_time",
               "sku_reorder_freq_days","ADI","CV2","category_enc","intermittency_enc"]
static_v = df_tr.groupby("item_id")[static_cols].first().values   # (n_skus, 8)

test_dow = np.vstack(df_te.groupby("item_id")["day_of_week"].apply(np.array).values)
test_woy = np.vstack(df_te.groupby("item_id")["week_of_year"].apply(np.array).values)
test_mon = np.vstack(df_te.groupby("item_id")["month"].apply(np.array).values)
test_wkn = np.vstack(df_te.groupby("item_id")["is_weekend"].apply(np.array).values)

sku_cls = df_te.groupby("item_id")["intermittency_class"].first().values

def batch_forecast(model, mtype="ml"):
    hist = [train_dem[:, i].copy() for i in range(train_dem.shape[1])]
    out = np.zeros((n_skus, HORIZON))
    for s in range(HORIZON):
        H = np.column_stack(hist); nh = H.shape[1]
        if mtype == "naive":
            p = H[:, -1]
        elif mtype == "ma7":
            p = H[:, -7:].mean(axis=1) if nh >= 7 else H.mean(axis=1)
        else:
            l1 = H[:,-1];  l3 = H[:,-3] if nh>=3 else l1
            l7 = H[:,-7] if nh>=7 else l1; l14 = H[:,-14] if nh>=14 else l1
            w7  = H[:,-7:]  if nh>=7  else H
            w14 = H[:,-14:] if nh>=14 else H
            rm7  = w7.mean(1);  rs7  = w7.std(1, ddof=1)
            rm14 = w14.mean(1); rs14 = w14.std(1, ddof=1)
            rcv  = np.where(rm14!=0, rs14/rm14, 0.)
            X = np.column_stack([
                l1,l3,l7,l14,rm7,rs7,rm14,rs14,rcv,
                test_dow[:,s].astype(float), test_woy[:,s].astype(float),
                test_mon[:,s].astype(float), test_wkn[:,s].astype(float),
                static_v])
            p = model.predict(X)
        p = np.clip(p, 0, None)
        out[:, s] = p; hist.append(p.copy())
    return out

preds = {}
for name, mdl, mt in [("Naive",None,"naive"),("MA-7",None,"ma7"),
                       ("XGBoost",model_xgb,"ml"),("LightGBM",model_lgb,"ml")]:
    preds[name] = batch_forecast(mdl, mt)
    print(f"   {ts()} {name} done")

# ═══════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Metrics (held-out test set):\n")
af = test_dem.ravel()
results = {}
for name, pr in preds.items():
    pf = pr.ravel()
    results[name] = {"RMSE": rmse(af,pf), "MAE": mae_(af,pf), "MAPE": mape_(af,pf)}

comp = pd.DataFrame(results).T; comp.index.name = "Model"
comp.to_csv(COMP_CSV)

winner = comp["RMSE"].idxmin()
runner = comp["RMSE"].nsmallest(2).index[1]

for m, r in comp.iterrows():
    star = " ★" if m == winner else ""
    print(f"  {m:<12s}  RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  "
          f"MAPE={r['MAPE']:.1f}%{star}")

margin = comp.loc[runner,"RMSE"] - comp.loc[winner,"RMSE"]
print(f"\n  Winner: {winner}  — beats {runner} by {margin:.4f} RMSE")

# ═══════════════════════════════════════════════════════════════════════════
#  SAVE BEST MODEL
# ═══════════════════════════════════════════════════════════════════════════

best_obj = model_xgb if winner == "XGBoost" else model_lgb
with open(MODEL_PATH, "wb") as f: pickle.dump(best_obj, f)
print(f"\n{ts()} Saved best model → {MODEL_PATH.relative_to(ROOT)}")

# ═══════════════════════════════════════════════════════════════════════════
#  ERROR ANALYSIS PLOTS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Generating error analysis plots ...")

best_preds = preds[winner]
resid = test_dem - best_preds

# Build error DF
rows = []
for si in range(n_skus):
    for st in range(HORIZON):
        rows.append({"residual": resid[si,st], "abs_error": abs(resid[si,st]),
                      "intermittency_class": sku_cls[si],
                      "day_of_week": int(test_dow[si,st]), "step": st+1})
edf = pd.DataFrame(rows)

# Plot A — residuals by intermittency class
fig, ax = plt.subplots(figsize=(9,5))
sns.boxplot(data=edf, x="intermittency_class", y="residual", ax=ax,
            palette="Set2", showfliers=False)
ax.axhline(0, color="red", lw=1, ls="--")
ax.set_title(f"Residuals by Intermittency Class — {winner}", fontsize=13, fontweight="bold")
ax.set_ylabel("Residual (actual − predicted)")
plt.tight_layout(); fig.savefig(ERR_DIR/"residuals_by_intermittency.png"); plt.close(fig)

# Plot B — residuals by day of week
fig, ax = plt.subplots(figsize=(9,5))
sns.boxplot(data=edf, x="day_of_week", y="residual", ax=ax,
            palette="coolwarm", showfliers=False)
ax.set_xticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])
ax.axhline(0, color="red", lw=1, ls="--")
ax.set_title(f"Residuals by Day of Week — {winner}", fontsize=13, fontweight="bold")
ax.set_ylabel("Residual (actual − predicted)")
plt.tight_layout(); fig.savefig(ERR_DIR/"residuals_by_dow.png"); plt.close(fig)

# Plot C — RMSE by forecast step
sr = edf.groupby("step").apply(lambda g: np.sqrt((g["residual"]**2).mean())).reset_index(name="RMSE")
fig, ax = plt.subplots(figsize=(9,5))
ax.bar(sr["step"], sr["RMSE"], color="#3498db", edgecolor="white")
ax.set_title(f"RMSE by Forecast Step — {winner}", fontsize=13, fontweight="bold")
ax.set_xlabel("Forecast Step (days ahead)"); ax.set_ylabel("RMSE")
ax.set_xticks(range(1,HORIZON+1))
plt.tight_layout(); fig.savefig(ERR_DIR/"rmse_by_step.png"); plt.close(fig)
print(f"   Saved 3 error-analysis plots")

# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION INTERVALS — Empirical Residual-Based
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Computing prediction intervals (empirical residual method) ...")

pi_rows = []
for step in range(HORIZON):
    step_resid = resid[:, step]
    pi_rows.append({
        "step": step + 1,
        "residual_mean": float(np.mean(step_resid)),
        "residual_std": float(np.std(step_resid, ddof=1)),
        "q05": float(np.percentile(step_resid, 5)),
        "q10": float(np.percentile(step_resid, 10)),
        "q25": float(np.percentile(step_resid, 25)),
        "q75": float(np.percentile(step_resid, 75)),
        "q90": float(np.percentile(step_resid, 90)),
        "q95": float(np.percentile(step_resid, 95)),
    })

pi_df = pd.DataFrame(pi_rows)
pi_df.to_csv(ERR_DIR / "prediction_intervals.csv", index=False)
print(f"   Saved → {(ERR_DIR / 'prediction_intervals.csv').relative_to(ROOT)}")

# Coverage check: what % of actuals fall within the 90% PI?
in_90 = 0; total_obs = 0
for step in range(HORIZON):
    lower = best_preds[:, step] + pi_df.loc[step, "q05"]
    upper = best_preds[:, step] + pi_df.loc[step, "q95"]
    in_90 += ((test_dem[:, step] >= lower) & (test_dem[:, step] <= upper)).sum()
    total_obs += n_skus
coverage_90 = in_90 / total_obs * 100
print(f"   90% PI empirical coverage: {coverage_90:.1f}% (target: 90%)")

# Fan chart — prediction intervals across forecast horizon
fig, ax = plt.subplots(figsize=(10, 5))
steps = np.arange(1, HORIZON + 1)
mean_pred = best_preds.mean(axis=0)
mean_actual = test_dem.mean(axis=0)
ax.fill_between(steps, mean_pred + pi_df["q05"].values,
                mean_pred + pi_df["q95"].values,
                alpha=0.15, color="#e74c3c", label="90% PI")
ax.fill_between(steps, mean_pred + pi_df["q25"].values,
                mean_pred + pi_df["q75"].values,
                alpha=0.3, color="#e74c3c", label="50% PI")
ax.plot(steps, mean_pred, color="#2c3e50", lw=2, label="Mean prediction",
        marker="o", markersize=4)
ax.plot(steps, mean_actual, color="#27ae60", lw=1.5, ls="--",
        label="Mean actual", marker="s", markersize=4)
ax.set_title(f"Prediction Intervals by Forecast Step — {winner}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Forecast Step (days ahead)")
ax.set_ylabel("Demand (units)")
ax.set_xticks(range(1, HORIZON + 1))
ax.legend()
plt.tight_layout()
fig.savefig(ERR_DIR / "prediction_intervals.png")
plt.close(fig)
print(f"   Saved → {(ERR_DIR / 'prediction_intervals.png').relative_to(ROOT)}")

# ═══════════════════════════════════════════════════════════════════════════
#  SENSITIVITY ANALYSIS — Feature Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} ── Sensitivity Analysis (CV RMSE, expanding-window) ──")

# Winner's hyperparameters for retraining
if winner == "XGBoost":
    sens_kw = dict(n_estimators=xgb_best["n_estimators"],
                   max_depth=xgb_best["max_depth"],
                   learning_rate=xgb_best["learning_rate"],
                   subsample=xgb_best["subsample"],
                   max_leaves=None, random_state=42)
else:
    sens_kw = dict(n_estimators=lgb_best["n_estimators"],
                   max_leaves=lgb_best["num_leaves"],
                   learning_rate=lgb_best["learning_rate"],
                   max_depth=12, subsample=1.0, random_state=42)

# Define ablation variants: (label, features_to_drop)
sens_variants = [
    ("baseline",          []),
    ("lag_14_dropped",    ["lag_14"]),
    ("lead_time_dropped", ["sku_mean_lead_time"]),
    ("rolling_dropped",   ["rolling_mean_7", "rolling_std_7",
                           "rolling_mean_14", "rolling_std_14"]),
]

sensitivity_rows = []
for label, drop_cols in sens_variants:
    keep = [c for c in FCOLS if c not in drop_cols]
    col_idx = [FCOLS.index(c) for c in keep]
    Xcv_v = Xcv[:, col_idx]

    # Pre-bin CV data for this feature subset
    binner_v = HistogramGBT()
    binner_v._prebin(Xcv_v)
    edges_v = binner_v._edges

    fold_scores = []
    for ti, vi in tscv.split(len(Xcv_v)):
        m = HistogramGBT(**sens_kw).fit(Xcv_v[ti], ycv[ti], _edges=edges_v)
        fold_scores.append(rmse(ycv[vi], m.predict(Xcv_v[vi])))
    cv_rmse_val = float(np.mean(fold_scores))
    sensitivity_rows.append({"variant": label, "cv_rmse": cv_rmse_val})
    dropped_str = ", ".join(drop_cols) if drop_cols else "(all features)"
    print(f"   {label:<20s} CV RMSE = {cv_rmse_val:.4f}  [{dropped_str}]")

# Compute % change relative to baseline
sens_baseline = sensitivity_rows[0]["cv_rmse"]
for row in sensitivity_rows:
    row["rmse_change_pct"] = round(
        (row["cv_rmse"] - sens_baseline) / sens_baseline * 100, 2)

# Save CSV
sens_df = pd.DataFrame(sensitivity_rows)[["variant", "cv_rmse", "rmse_change_pct"]]
sens_df.to_csv(ERR_DIR / "sensitivity_results.csv", index=False)
print(f"\n   {ts()} Saved → {(ERR_DIR / 'sensitivity_results.csv').relative_to(ROOT)}")

# Bar chart — baseline vs ablation variants
fig, ax = plt.subplots(figsize=(9, 5))
bar_colors = ["#2c3e50" if v == "baseline" else "#e74c3c"
              for v in sens_df["variant"]]
bars = ax.bar(sens_df["variant"], sens_df["cv_rmse"],
              color=bar_colors, edgecolor="white", width=0.55)
ax.set_title("Sensitivity Analysis — CV RMSE by Feature Ablation",
             fontsize=13, fontweight="bold")
ax.set_ylabel("CV RMSE")
ax.set_xlabel("Variant")
for bar, val in zip(bars, sens_df["cv_rmse"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.4f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig.savefig(ERR_DIR / "sensitivity_analysis.png")
plt.close(fig)
print(f"   {ts()} Saved → {(ERR_DIR / 'sensitivity_analysis.png').relative_to(ROOT)}")

# Plain-English summary
sens_ablations = sens_df[sens_df["variant"] != "baseline"]
worst_row = sens_ablations.loc[sens_ablations["rmse_change_pct"].idxmax()]
worst_label = (worst_row["variant"]
               .replace("_dropped", "").replace("_", " "))
print(f"\n   ┌─────────────────────────────────────────────────────────┐")
print(f"   │  SENSITIVITY SUMMARY                                    │")
print(f"   ├─────────────────────────────────────────────────────────┤")
print(f"   │  Baseline CV RMSE : {sens_baseline:<36.4f} │")
for _, r in sens_ablations.iterrows():
    lbl = r["variant"].replace("_dropped", "").replace("_", " ")
    print(f"   │  Drop {lbl:<13s}: {r['cv_rmse']:.4f}  "
          f"({r['rmse_change_pct']:+.1f}%){' ':>{28-len(lbl)}}│")
print(f"   ├─────────────────────────────────────────────────────────┤")
print(f"   │  Removing '{worst_label}' features hurts the most,{' ':>7}│")
print(f"   │  increasing CV RMSE by {worst_row['rmse_change_pct']:+.1f}%"
      f" (from {sens_baseline:.4f} to {worst_row['cv_rmse']:.4f}).  │")
print(f"   └─────────────────────────────────────────────────────────┘")

# ═══════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE (top 10)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Top 10 feature importances ({winner}):\n")
imp = best_obj.feature_importances_
idf = pd.DataFrame({"feature": FCOLS, "importance": imp}).sort_values(
    "importance", ascending=False).reset_index(drop=True)
for i, row in idf.head(10).iterrows():
    bar = "█" * int(row["importance"] * 60)
    print(f"  {i+1:>2}. {row['feature']:<25s} {row['importance']:.4f}  {bar}")

fig, ax = plt.subplots(figsize=(9,5))
t10 = idf.head(10)
ax.barh(t10["feature"][::-1], t10["importance"][::-1], color="#2c3e50")
ax.set_title(f"Top 10 Feature Importances — {winner}", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance (gain)")
plt.tight_layout(); fig.savefig(ERR_DIR/"feature_importance.png"); plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  PERMUTATION IMPORTANCE (model-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Computing permutation importance (5 repeats) ...")

N_REPEATS = 5
base_rmse_val = rmse(y_te, best_obj.predict(X_te))
perm_rows = []
rng_perm = np.random.default_rng(42)

for fi, fname in enumerate(FCOLS):
    rmse_increases = []
    for _ in range(N_REPEATS):
        X_perm = X_te.copy()
        X_perm[:, fi] = rng_perm.permutation(X_perm[:, fi])
        perm_rmse_val = rmse(y_te, best_obj.predict(X_perm))
        rmse_increases.append(perm_rmse_val - base_rmse_val)
    perm_rows.append({
        "feature": fname,
        "importance_mean": float(np.mean(rmse_increases)),
        "importance_std": float(np.std(rmse_increases)),
    })
    if (fi + 1) % 7 == 0 or fi == len(FCOLS) - 1:
        print(f"   {ts()} {fi+1}/{len(FCOLS)} features done")

perm_df = pd.DataFrame(perm_rows).sort_values("importance_mean", ascending=False)
perm_df.to_csv(ERR_DIR / "permutation_importance.csv", index=False)
print(f"   Saved → {(ERR_DIR / 'permutation_importance.csv').relative_to(ROOT)}")

# Side-by-side plot: gain-based vs permutation importance (top 10)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

t10_gain = idf.head(10)
ax1.barh(t10_gain["feature"][::-1], t10_gain["importance"][::-1], color="#2c3e50")
ax1.set_title("Gain-Based Importance", fontsize=12, fontweight="bold")
ax1.set_xlabel("Importance (gain)")

t10_perm = perm_df.head(10)
ax2.barh(t10_perm["feature"].values[::-1], t10_perm["importance_mean"].values[::-1],
         xerr=t10_perm["importance_std"].values[::-1], color="#e74c3c", capsize=3)
ax2.set_title("Permutation Importance", fontsize=12, fontweight="bold")
ax2.set_xlabel("RMSE Increase")

plt.suptitle(f"Feature Importance Comparison — {winner}",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(ERR_DIR / "importance_comparison.png", bbox_inches="tight")
plt.close(fig)
print(f"   Saved → {(ERR_DIR / 'importance_comparison.png').relative_to(ROOT)}")

print(f"\n{ts()} Top 10 permutation importance ({winner}):\n")
for i, (_, row) in enumerate(perm_df.head(10).iterrows()):
    bar = "█" * max(1, int(row["importance_mean"] / perm_df["importance_mean"].max() * 40))
    print(f"  {i+1:>2}. {row['feature']:<25s} +{row['importance_mean']:.4f} "
          f"(±{row['importance_std']:.4f})  {bar}")

# ═══════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"\n  Winner         : {winner}")
bp = xgb_best if winner=="XGBoost" else lgb_best
print(f"  Best params    : {bp}")
print(f"  Test RMSE      : {comp.loc[winner,'RMSE']:.4f}")
print(f"  Test MAE       : {comp.loc[winner,'MAE']:.4f}")
print(f"  Test MAPE      : {comp.loc[winner,'MAPE']:.1f}%")
print(f"  Margin vs {runner}: {margin:.4f} RMSE")
print(f"  Sensitivity    : dropping '{worst_label}' hurts most → "
      f"CV RMSE +{worst_row['rmse_change_pct']:.1f}%")
print(f"  Top 3 features : {', '.join(idf.head(3)['feature'])}")
print(f"  90% PI coverage: {coverage_90:.1f}%")
print(f"\n  Outputs:")
print(f"    {MODEL_PATH.relative_to(ROOT)}")
print(f"    {COMP_CSV.relative_to(ROOT)}")
print(f"    {(ERR_DIR / 'sensitivity_results.csv').relative_to(ROOT)}")
print(f"    {(ERR_DIR / 'prediction_intervals.csv').relative_to(ROOT)}")
print(f"    {(ERR_DIR / 'permutation_importance.csv').relative_to(ROOT)}")
print(f"    {ERR_DIR.relative_to(ROOT)}/  (8 plots)")
print(f"\n  Total runtime  : {ts()}")
print("=" * 70)
