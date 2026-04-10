"""
05_optimisation.py
═══════════════════════════════════════════════════════════════════════
Dynamic inventory optimisation using the trained XGBoost forecast model.

Pipeline:
  1. Generate 14-day recursive demand forecasts for every SKU
  2. Derive demand & lead-time statistics from forecasts + raw data
  3. Compute dynamic safety stock, ROP, EOQ, and total annual cost
  4. Compare against a static baseline (historical avg + 7-day buffer)
  5. Save recommendations and print cost-reduction summary

Configurable cost parameters (passed as constants at the top):
  S_ORDER  — fixed ordering cost per order  (default $50)
  H_HOLD   — annual holding cost per unit   (default $2)
  PENALTY  — shortage penalty per unit       (default $10)
  Z_SL     — z-score for service level       (1.645 → 95 %)
  SIGMA_LT — assumed lead-time std (days)    (default 1.5)
═══════════════════════════════════════════════════════════════════════
"""

import warnings, pickle, time
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# ── configurable parameters ────────────────────────────────────────────────
S_ORDER   = 50.0      # ordering cost per order ($)
H_HOLD    = 2.0       # annual holding cost per unit ($ — default)
PENALTY   = 10.0      # shortage penalty per unit ($)
Z_SL      = 1.645     # z-score for 95 % service level
SIGMA_LT  = 1.5       # assumed lead-time std deviation (days)
HORIZON   = 14

ROOT       = Path(__file__).resolve().parent.parent
FEAT_CSV   = ROOT / "data" / "cleaned" / "features.csv"
CLEAN_CSV  = ROOT / "data" / "cleaned" / "daily_demand.csv"
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
REC_CSV    = ROOT / "outputs" / "recommendations.csv"

T0 = time.time()
def ts(): return f"[{time.time()-T0:5.0f}s]"

print("=" * 70)
print("  INVENTORY OPTIMISATION  —  05_optimisation.py")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
#  HistogramGBT class (must be in scope for pickle.load)
# ═══════════════════════════════════════════════════════════════════════════

N_BINS = 16; MIN_LEAF = 30

class HistogramGBT:
    def __init__(self, *, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=1.0, max_leaves=None, random_state=42):
        self.n_estimators=n_estimators; self.max_depth=max_depth
        self.learning_rate=learning_rate; self.subsample=subsample
        self.max_leaves=max_leaves; self.random_state=random_state
    def _prebin(self, X):
        self._edges={}; Xb=np.empty(X.shape,np.uint8)
        for j in range(X.shape[1]):
            e=np.unique(np.percentile(X[:,j],np.linspace(0,100,N_BINS+1)))
            Xb[:,j]=np.clip(np.searchsorted(e[1:-1],X[:,j],side="right"),0,N_BINS-1)
            self._edges[j]=e
        return Xb
    def _bin(self, X):
        Xb=np.empty(X.shape,np.uint8)
        for j in range(X.shape[1]):
            e=self._edges[j]
            Xb[:,j]=np.clip(np.searchsorted(e[1:-1],X[:,j],side="right"),0,N_BINS-1)
        return Xb
    def _build(self,Xb,r,d,md):
        n=len(r)
        if d>=md or n<2*MIN_LEAF: return(float(r.mean()),)
        tot=r.sum(); bg,bf,bb=0.,-1,-1
        for j in range(Xb.shape[1]):
            hs=np.bincount(Xb[:,j],weights=r,minlength=N_BINS)
            hc=np.bincount(Xb[:,j],minlength=N_BINS)
            cs=np.cumsum(hs);cc=np.cumsum(hc);rs=tot-cs;rc=n-cc
            ok=(cc>=MIN_LEAF)&(rc>=MIN_LEAF)
            if not ok.any(): continue
            g=np.full(N_BINS,-np.inf); g[ok]=cs[ok]**2/cc[ok]+rs[ok]**2/rc[ok]
            bi=int(np.argmax(g)); gi=g[bi]-tot**2/n
            if gi>bg: bg,bf,bb=gi,j,bi
        if bf<0: return(float(r.mean()),)
        m=Xb[:,bf]<=bb
        return(bf,bb,bg,self._build(Xb[m],r[m],d+1,md),self._build(Xb[~m],r[~m],d+1,md))
    def _ptree(self,Xb,nd):
        out=np.empty(Xb.shape[0]); self._fill(Xb,np.arange(Xb.shape[0]),nd,out); return out
    def _fill(self,Xb,ix,nd,out):
        if len(nd)==1: out[ix]=nd[0]; return
        m=Xb[ix,nd[0]]<=nd[1]; li,ri=ix[m],ix[~m]
        if li.size: self._fill(Xb,li,nd[3],out)
        if ri.size: self._fill(Xb,ri,nd[4],out)
    def fit(self,X,y,_edges=None):
        rng=np.random.default_rng(self.random_state)
        if _edges: self._edges=_edges; Xb=self._bin(X)
        else: Xb=self._prebin(X)
        eff_d=(max(1,int(np.ceil(np.log2(max(self.max_leaves,2)))))
               if self.max_leaves else self.max_depth)
        n,d=X.shape; self._init=float(y.mean()); preds=np.full(n,self._init)
        self._trees=[]; self._imp=np.zeros(d)
        for _ in range(self.n_estimators):
            resid=y-preds
            ix=(rng.choice(n,int(n*self.subsample),replace=False)
                if self.subsample<1 else np.arange(n))
            tree=self._build(Xb[ix],resid[ix],0,eff_d)
            self._trees.append(tree); preds+=self.learning_rate*self._ptree(Xb,tree)
            self._aimp(tree)
        s=self._imp.sum()
        self.feature_importances_=self._imp/s if s else self._imp; return self
    def _aimp(self,nd):
        if len(nd)==1: return
        self._imp[nd[0]]+=nd[2]; self._aimp(nd[3]); self._aimp(nd[4])
    def predict(self,X):
        Xb=self._bin(X); p=np.full(X.shape[0],self._init)
        for t in self._trees: p+=self.learning_rate*self._ptree(Xb,t)
        return p


# ═══════════════════════════════════════════════════════════════════════════
#  Normal CDF approximation (Abramowitz & Stegun)
# ═══════════════════════════════════════════════════════════════════════════

def norm_cdf(x):
    a1,a2,a3 = 0.254829592, -0.284496736, 1.421413741
    a4,a5,p  = -1.453152027, 1.061405429, 0.3275911
    s = np.where(x >= 0, 1.0, -1.0)
    x = np.abs(x) / np.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t * np.exp(-x*x)
    return 0.5 * (1 + s * y)


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Loading data and model ...")

df_feat  = pd.read_csv(FEAT_CSV,  parse_dates=["date"])
df_clean = pd.read_csv(CLEAN_CSV, parse_dates=["last_restock_date"])

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Encode categoricals (must match 04's encoding)
cat_map = {c: i for i, c in enumerate(df_feat["category"].unique())}
int_map = {c: i for i, c in enumerate(df_feat["intermittency_class"].unique())}
df_feat["category_enc"]      = df_feat["category"].map(cat_map).astype(float)
df_feat["intermittency_enc"] = df_feat["intermittency_class"].map(int_map).astype(float)

FCOLS = [
    "lag_1","lag_3","lag_7","lag_14",
    "rolling_mean_7","rolling_std_7","rolling_mean_14","rolling_std_14","rolling_cv_14",
    "day_of_week","week_of_year","month","is_weekend",
    "sku_mean_demand","sku_std_demand","sku_mean_lead_time","sku_std_lead_time",
    "sku_reorder_freq_days","ADI","CV2","category_enc","intermittency_enc",
]

sku_ids = df_feat["item_id"].unique()
n_skus  = len(sku_ids)
print(f"   SKUs: {n_skus:,}  |  Model: {len(model._trees)} trees")

# ═══════════════════════════════════════════════════════════════════════════
#  2. GENERATE 14-DAY FORECASTS (recursive batch)
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Generating {HORIZON}-day recursive forecasts ...")

df_feat.sort_values(["item_id", "date"], inplace=True)

# All 46 rows per SKU used as history (production scenario)
demands_all = np.vstack(
    df_feat.groupby("item_id")["demand"].apply(np.array).values)   # (n, 46)

static_cols = ["sku_mean_demand","sku_std_demand","sku_mean_lead_time",
               "sku_reorder_freq_days","ADI","CV2",
               "category_enc","intermittency_enc"]
static_v = df_feat.groupby("item_id")[static_cols].first().values  # (n, 9)

# Calendar features for forecast horizon (extend from last date per SKU)
last_dates = df_feat.groupby("item_id")["date"].last().values
fc_dates = np.array([[pd.Timestamp(d) + pd.Timedelta(days=s+1)
                       for s in range(HORIZON)] for d in last_dates])  # (n, 14)

fc_dow  = np.vectorize(lambda d: d.dayofweek)(fc_dates).astype(float)
fc_woy  = np.vectorize(lambda d: d.isocalendar()[1])(fc_dates).astype(float)
fc_mon  = np.vectorize(lambda d: d.month)(fc_dates).astype(float)
fc_wknd = (fc_dow >= 5).astype(float)

# Recursive batch forecast
hist = [demands_all[:, i].copy() for i in range(demands_all.shape[1])]
forecasts = np.zeros((n_skus, HORIZON))

for s in range(HORIZON):
    H  = np.column_stack(hist); nh = H.shape[1]
    l1 = H[:,-1]; l3 = H[:,-3] if nh>=3 else l1
    l7 = H[:,-7] if nh>=7 else l1; l14 = H[:,-14] if nh>=14 else l1
    w7  = H[:,-7:]  if nh>=7  else H
    w14 = H[:,-14:] if nh>=14 else H
    rm7=w7.mean(1);  rs7=w7.std(1,ddof=1)
    rm14=w14.mean(1); rs14=w14.std(1,ddof=1)
    rcv=np.where(rm14!=0, rs14/rm14, 0.)
    X = np.column_stack([
        l1,l3,l7,l14,rm7,rs7,rm14,rs14,rcv,
        fc_dow[:,s], fc_woy[:,s], fc_mon[:,s], fc_wknd[:,s],
        static_v])
    p = np.clip(model.predict(X), 0, None)
    forecasts[:, s] = p
    hist.append(p.copy())

print(f"   Forecast matrix: {forecasts.shape}")
print(f"   Mean daily forecast: {forecasts.mean():.2f} units/day")

# ═══════════════════════════════════════════════════════════════════════════
#  3. DEMAND & LEAD-TIME STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Computing demand and lead-time statistics ...")

mu_demand    = forecasts.mean(axis=1)          # mean daily demand from forecast
sigma_demand = forecasts.std(axis=1, ddof=1)   # std of daily demand from forecast
forecast_14d = forecasts.sum(axis=1)           # total 14-day forecast

# Lead-time from cleaned dataset (one value per SKU, same order as sku_ids)
sku_order = df_feat.groupby("item_id").ngroup()
clean_indexed = df_clean.set_index("item_id").loc[sku_ids]

mu_lead    = clean_indexed["lead_time_days"].values.astype(float)
sigma_lead = np.full(n_skus, SIGMA_LT)  # assumed constant (configurable)

# Per-SKU holding cost from dataset (per DAY → convert to per YEAR)
h_actual = clean_indexed["holding_cost_per_unit_day"].values * 365.0
stock_level = clean_indexed["stock_level"].values.astype(float)
unit_price  = clean_indexed["unit_price"].values.astype(float)
categories  = clean_indexed["category"].values
stockouts_lm = clean_indexed["stockout_count_last_month"].values.astype(float)

print(f"   mu_demand  range: [{mu_demand.min():.2f}, {mu_demand.max():.2f}]")
print(f"   mu_lead    range: [{mu_lead.min():.0f}, {mu_lead.max():.0f}] days")

# ═══════════════════════════════════════════════════════════════════════════
#  4. DYNAMIC INVENTORY POLICY
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Computing dynamic inventory policy (z = {Z_SL}, SL = 95%) ...")

# Safety stock = z * sqrt(mu_lead * sigma_demand² + mu_demand² * sigma_lead²)
safety_stock = Z_SL * np.sqrt(mu_lead * sigma_demand**2 + mu_demand**2 * sigma_lead**2)

# Reorder point
rop = mu_demand * mu_lead + safety_stock

# EOQ = sqrt(2*D*S / H)
D_annual = mu_demand * 365.0
H_sku    = np.where(h_actual > 0, h_actual, H_HOLD)  # use dataset value, fallback to default
eoq      = np.sqrt(2 * D_annual * S_ORDER / H_sku)
eoq      = np.clip(eoq, 1, None)  # at least 1 unit

# Total annual cost (dynamic)
ordering_cost_dyn = (D_annual / eoq) * S_ORDER
holding_cost_dyn  = (eoq / 2 + safety_stock) * H_sku
# Shortage cost: P(stockout) × D × penalty
p_shortage_dyn    = 1 - 0.95  # by design of z = 1.645
shortage_cost_dyn = p_shortage_dyn * D_annual * PENALTY
total_cost_dyn    = ordering_cost_dyn + holding_cost_dyn + shortage_cost_dyn

print(f"   Safety stock  mean: {safety_stock.mean():.1f}  median: {np.median(safety_stock):.1f}")
print(f"   ROP           mean: {rop.mean():.1f}  median: {np.median(rop):.1f}")
print(f"   EOQ           mean: {eoq.mean():.1f}  median: {np.median(eoq):.1f}")

# ═══════════════════════════════════════════════════════════════════════════
#  5. STATIC BASELINE
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Computing static baseline (historical avg + 7-day buffer) ...")

# Static uses overall historical average demand (from cleaned data)
hist_mean = clean_indexed["daily_demand"].values.astype(float)
hist_std  = clean_indexed["demand_std_dev"].values.astype(float)
avg_lead  = mu_lead.mean()  # single global average lead time

# Static safety stock = historical avg demand × 7 days
safety_stock_static = hist_mean * 7.0
rop_static          = hist_mean * avg_lead + safety_stock_static

# Static EOQ (same formula, using historical demand)
D_annual_static = hist_mean * 365.0
eoq_static      = np.sqrt(2 * D_annual_static * S_ORDER / H_sku)
eoq_static      = np.clip(eoq_static, 1, None)

# Static total cost
ordering_static  = (D_annual_static / eoq_static) * S_ORDER
holding_static   = (eoq_static / 2 + safety_stock_static) * H_sku

# Effective static service level: z_static = ss_static / sigma_total
sigma_total  = np.sqrt(mu_lead * hist_std**2 + hist_mean**2 * sigma_lead**2)
z_static     = np.where(sigma_total > 0, safety_stock_static / sigma_total, 3.0)
p_shortage_s = 1 - norm_cdf(z_static)
shortage_static = p_shortage_s * D_annual_static * PENALTY
total_cost_sta  = ordering_static + holding_static + shortage_static

print(f"   Static SS     mean: {safety_stock_static.mean():.1f}")
print(f"   Static ROP    mean: {rop_static.mean():.1f}")
print(f"   Avg effective SL  : {norm_cdf(z_static).mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
#  6. COMPARISON & RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Building recommendations table ...")

cost_saving     = total_cost_sta - total_cost_dyn
cost_saving_pct = np.where(total_cost_sta > 0,
                           cost_saving / total_cost_sta * 100, 0.0)

# Reorder action: "Reorder Now" if current stock < dynamic ROP
action = np.where(stock_level < rop, "Reorder Now", "Hold")

rec = pd.DataFrame({
    "SKU":                   sku_ids,
    "category":              categories,
    "forecast_14d_total":    np.round(forecast_14d, 2),
    "mu_demand":             np.round(mu_demand, 3),
    "sigma_demand":          np.round(sigma_demand, 3),
    "mu_lead":               mu_lead,
    "sigma_lead":            sigma_lead,
    "safety_stock":          np.round(safety_stock, 2),
    "ROP":                   np.round(rop, 2),
    "EOQ":                   np.round(eoq, 2),
    "current_stock":         stock_level.astype(int),
    "annual_cost_dynamic":   np.round(total_cost_dyn, 2),
    "annual_cost_static":    np.round(total_cost_sta, 2),
    "cost_saving_pct":       np.round(cost_saving_pct, 2),
    "action":                action,
})

rec.to_csv(REC_CSV, index=False)
print(f"   Saved → {REC_CSV.relative_to(ROOT)}")
print(f"   Shape: {rec.shape}")

# ═══════════════════════════════════════════════════════════════════════════
#  7. SUMMARY & TOP 10
# ═══════════════════════════════════════════════════════════════════════════

n_better   = (cost_saving_pct > 0).sum()
n_worse    = (cost_saving_pct < 0).sum()
avg_saving = cost_saving_pct.mean()
total_dyn  = total_cost_dyn.sum()
total_sta  = total_cost_sta.sum()
agg_saving = (total_sta - total_dyn) / total_sta * 100

reorder_now = (action == "Reorder Now").sum()

print("\n" + "=" * 70)
print("  OPTIMISATION SUMMARY")
print("=" * 70)

print(f"""
  Policy parameters:
    Service level target  : 95% (z = {Z_SL})
    Ordering cost (S)     : ${S_ORDER:.0f}
    Holding cost (H)      : per-SKU from dataset (mean ${H_sku.mean():.0f}/yr)
    Shortage penalty      : ${PENALTY:.0f}/unit
    Lead-time std (σ_LT)  : {SIGMA_LT} days

  Aggregate results:
    Total annual cost (dynamic) : ${total_dyn:>14,.0f}
    Total annual cost (static)  : ${total_sta:>14,.0f}
    Aggregate cost reduction    : {agg_saving:>8.1f}%

  Per-SKU breakdown:
    SKUs where dynamic is better : {n_better:>5,} / {n_skus:,}  ({n_better/n_skus*100:.1f}%)
    SKUs where static is better  : {n_worse:>5,} / {n_skus:,}  ({n_worse/n_skus*100:.1f}%)
    Average cost saving (per SKU): {avg_saving:>8.1f}%

  Action summary:
    Reorder Now : {reorder_now:>5,} SKUs  ({reorder_now/n_skus*100:.1f}%)
    Hold        : {n_skus - reorder_now:>5,} SKUs  ({(n_skus-reorder_now)/n_skus*100:.1f}%)
""")

# ── Top 10 SKUs by cost saving ────────────────────────────────────────────
print("  TOP 10 SKUs BY COST SAVING")
print("  " + "-" * 66)
top10 = rec.nlargest(10, "cost_saving_pct")
for _, r in top10.iterrows():
    print(f"  {r['SKU']:<10s} {r['category']:<12s}  "
          f"dyn=${r['annual_cost_dynamic']:>9,.0f}  "
          f"sta=${r['annual_cost_static']:>9,.0f}  "
          f"saving={r['cost_saving_pct']:>6.1f}%  "
          f"action={r['action']}")

print("\n  " + "-" * 66)

# ── Bottom 5 (where static may be better) ────────────────────────────────
print("\n  BOTTOM 5 SKUs (static may outperform)")
print("  " + "-" * 66)
bot5 = rec.nsmallest(5, "cost_saving_pct")
for _, r in bot5.iterrows():
    print(f"  {r['SKU']:<10s} {r['category']:<12s}  "
          f"dyn=${r['annual_cost_dynamic']:>9,.0f}  "
          f"sta=${r['annual_cost_static']:>9,.0f}  "
          f"saving={r['cost_saving_pct']:>6.1f}%  "
          f"action={r['action']}")

# ═══════════════════════════════════════════════════════════════════════════
#  8. FORECAST ERROR → COST PROPAGATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n{ts()} Forecast error → cost propagation analysis ...")

ERR_DIR = ROOT / "outputs" / "error_analysis"
ERR_DIR.mkdir(parents=True, exist_ok=True)

# Simulate: scale sigma_demand by multipliers to show impact on costs
error_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
prop_rows = []

for mult in error_multipliers:
    sd_sim = sigma_demand * mult
    ss_sim = Z_SL * np.sqrt(mu_lead * sd_sim**2 + mu_demand**2 * sigma_lead**2)
    rop_sim = mu_demand * mu_lead + ss_sim

    holding_sim  = (eoq / 2 + ss_sim) * H_sku
    ordering_sim = ordering_cost_dyn
    shortage_sim = shortage_cost_dyn
    total_sim    = ordering_sim + holding_sim + shortage_sim

    reorder_sim = int((stock_level < rop_sim).sum())

    prop_rows.append({
        "error_multiplier": mult,
        "mean_safety_stock": float(ss_sim.mean()),
        "mean_rop": float(rop_sim.mean()),
        "total_annual_cost": float(total_sim.sum()),
        "reorder_now_count": reorder_sim,
        "cost_vs_baseline_pct": float(
            (total_sim.sum() - total_cost_dyn.sum()) / total_cost_dyn.sum() * 100
        ),
    })

prop_df = pd.DataFrame(prop_rows)
prop_df.to_csv(ERR_DIR / "error_cost_propagation.csv", index=False)
print(f"   Saved → outputs/error_analysis/error_cost_propagation.csv")

print(f"\n   {'Multiplier':>12s}  {'Avg SS':>8s}  {'Avg ROP':>8s}  "
      f"{'Total Cost':>14s}  {'Reorder':>8s}  {'Cost Δ':>8s}")
print(f"   {'-'*12}  {'-'*8}  {'-'*8}  {'-'*14}  {'-'*8}  {'-'*8}")
for _, r in prop_df.iterrows():
    print(f"   {r['error_multiplier']:>10.2f}×  {r['mean_safety_stock']:>8.1f}  "
          f"{r['mean_rop']:>8.1f}  ${r['total_annual_cost']:>13,.0f}  "
          f"{r['reorder_now_count']:>7,}  {r['cost_vs_baseline_pct']:>+7.1f}%")

print("\n" + "=" * 70)
print(f"  Runtime: {ts()}")
print("=" * 70)
