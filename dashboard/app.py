"""
Streamlit Dashboard — MSE 433 Demand Forecasting & Inventory Optimisation
=========================================================================

Launch:  streamlit run dashboard/app.py

Tabs
----
  1. Demand Forecast   — 14-day forecast chart per SKU with ±1 std band
  2. Inventory Policy   — ROP, safety stock, EOQ, action for all SKUs
  3. Cost Comparison    — Dynamic vs static policy for top 20 SKUs
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
FEAT_CSV   = ROOT / "data" / "cleaned" / "features.csv"
CLEAN_CSV  = ROOT / "data" / "cleaned" / "daily_demand.csv"
REC_CSV    = ROOT / "outputs" / "recommendations.csv"
MODEL_PATH = ROOT / "outputs" / "best_model.joblib"
COMP_CSV   = ROOT / "outputs" / "model_comparison.csv"
INTERM_CSV = ROOT / "outputs" / "sku_intermittency.csv"

HORIZON = 14
PENALTY = 10.0   # shortage penalty per unit ($)

# ── HistogramGBT class (needed for pickle) ────────────────────────────────
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


# ── Normal distribution helpers ───────────────────────────────────────────

def norm_cdf(x):
    """Abramowitz & Stegun approximation of the normal CDF."""
    a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
    a4, a5, p  = -1.453152027, 1.061405429, 0.3275911
    s = np.where(x >= 0, 1.0, -1.0)
    x = np.abs(x) / np.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1 - (((((a5*t+a4)*t)+a3)*t+a2)*t+a1)*t * np.exp(-x*x)
    return 0.5 * (1 + s * y)


def norm_ppf(p):
    """Abramowitz & Stegun rational approximation of the inverse normal CDF."""
    a0, a1, a2 = 2.515517, 0.802853, 0.010328
    b1, b2, b3 = 1.432788, 0.189269, 0.001308
    if p < 0.5:
        return -norm_ppf(1.0 - p)
    t = np.sqrt(-2.0 * np.log(1.0 - p))
    return t - (a0 + a1 * t + a2 * t**2) / (1.0 + b1 * t + b2 * t**2 + b3 * t**3)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_recommendations():
    return pd.read_csv(REC_CSV)

@st.cache_data
def load_features():
    return pd.read_csv(FEAT_CSV, parse_dates=["date"])

@st.cache_data
def load_clean():
    return pd.read_csv(CLEAN_CSV, parse_dates=["last_restock_date"])

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_model_comparison():
    return pd.read_csv(COMP_CSV, index_col=0)

@st.cache_data
def load_intermittency():
    return pd.read_csv(INTERM_CSV)

@st.cache_data
def load_prediction_intervals():
    pi_path = ROOT / "outputs" / "error_analysis" / "prediction_intervals.csv"
    if pi_path.exists():
        return pd.read_csv(pi_path)
    return None

@st.cache_data
def load_permutation_importance():
    perm_path = ROOT / "outputs" / "error_analysis" / "permutation_importance.csv"
    if perm_path.exists():
        return pd.read_csv(perm_path)
    return None

@st.cache_data
def load_gain_importance():
    feat_csv = ROOT / "data" / "cleaned" / "features.csv"
    model = load_model()
    fcols = [
        "lag_1","lag_3","lag_7","lag_14",
        "rolling_mean_7","rolling_std_7","rolling_mean_14","rolling_std_14","rolling_cv_14",
        "day_of_week","week_of_year","month","is_weekend",
        "sku_mean_demand","sku_std_demand","sku_mean_lead_time","sku_reorder_freq_days",
        "ADI","CV2",
    ]
    imp = model.feature_importances_
    # feature_importances_ length may differ from fcols if encoding cols included
    # use min length for safety
    n = min(len(fcols), len(imp))
    return pd.DataFrame({"feature": fcols[:n], "importance": imp[:n]}).sort_values(
        "importance", ascending=False)

@st.cache_data
def load_error_propagation():
    ep_path = ROOT / "outputs" / "error_analysis" / "error_cost_propagation.csv"
    if ep_path.exists():
        return pd.read_csv(ep_path)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  REAL-TIME POLICY RECALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def recalculate_policies(rec_base, clean_df, s_order, h_hold, service_pct):
    """Recalculate dynamic & static inventory policies from slider values."""
    z_sl = norm_ppf(service_pct / 100.0)

    mu_demand    = rec_base["mu_demand"].values
    sigma_demand = rec_base["sigma_demand"].values
    mu_lead      = rec_base["mu_lead"].values
    sigma_lead   = rec_base["sigma_lead"].values
    current_stock = rec_base["current_stock"].values

    H_sku = h_hold  # uniform holding cost from slider

    # ── Dynamic policy ────────────────────────────────────────────────────
    safety_stock = z_sl * np.sqrt(
        mu_lead * sigma_demand**2 + mu_demand**2 * sigma_lead**2
    )
    rop       = mu_demand * mu_lead + safety_stock
    D_annual  = mu_demand * 365.0
    eoq       = np.clip(np.sqrt(2 * D_annual * s_order / H_sku), 1, None)

    ordering_dyn = (D_annual / eoq) * s_order
    holding_dyn  = (eoq / 2 + safety_stock) * H_sku
    shortage_dyn = (1.0 - service_pct / 100.0) * D_annual * PENALTY
    total_dyn    = ordering_dyn + holding_dyn + shortage_dyn

    # ── Static baseline ───────────────────────────────────────────────────
    clean_indexed = clean_df.set_index("item_id")
    hist_mean = clean_indexed.reindex(rec_base["SKU"])["daily_demand"].values.astype(float)
    hist_std  = clean_indexed.reindex(rec_base["SKU"])["demand_std_dev"].values.astype(float)

    ss_static       = hist_mean * 7.0
    D_annual_static = hist_mean * 365.0
    eoq_static      = np.clip(np.sqrt(2 * D_annual_static * s_order / H_sku), 1, None)

    ordering_sta = (D_annual_static / eoq_static) * s_order
    holding_sta  = (eoq_static / 2 + ss_static) * H_sku

    sigma_total  = np.sqrt(mu_lead * hist_std**2 + hist_mean**2 * sigma_lead**2)
    z_static     = np.where(sigma_total > 0, ss_static / sigma_total, 3.0)
    shortage_sta = (1 - norm_cdf(z_static)) * D_annual_static * PENALTY
    total_sta    = ordering_sta + holding_sta + shortage_sta

    # ── Assemble result ───────────────────────────────────────────────────
    cost_saving_pct = np.where(
        total_sta > 0, (total_sta - total_dyn) / total_sta * 100, 0.0
    )
    action = np.where(current_stock < rop, "Reorder Now", "Hold")

    result = rec_base.copy()
    result["safety_stock"]        = np.round(safety_stock, 2)
    result["ROP"]                 = np.round(rop, 2)
    result["EOQ"]                 = np.round(eoq, 2)
    result["annual_cost_dynamic"] = np.round(total_dyn, 2)
    result["annual_cost_static"]  = np.round(total_sta, 2)
    result["cost_saving_pct"]     = np.round(cost_saving_pct, 2)
    result["action"]              = action
    return result


@st.cache_data
def generate_sku_forecast(sku_id: str):
    """Generate 14-day recursive forecast for a single SKU."""
    df = load_features()
    model = load_model()

    sku_data = df[df["item_id"] == sku_id].sort_values("date").copy()
    if sku_data.empty:
        return None, None, None

    cat_map = {c: i for i, c in enumerate(df["category"].unique())}
    int_map = {c: i for i, c in enumerate(df["intermittency_class"].unique())}

    hist_demand = sku_data["demand"].values.tolist()
    last_date = sku_data["date"].max()

    row0 = sku_data.iloc[0]
    static = np.array([
        row0["sku_mean_demand"], row0["sku_std_demand"],
        row0["sku_mean_lead_time"], row0["sku_reorder_freq_days"],
        row0["ADI"], row0["CV2"],
        cat_map.get(row0["category"], 0),
        int_map.get(row0["intermittency_class"], 0),
    ])

    forecasts = []
    for step in range(HORIZON):
        fc_date = last_date + pd.Timedelta(days=step + 1)
        H = np.array(hist_demand)
        nh = len(H)
        l1 = H[-1]; l3 = H[-3] if nh >= 3 else l1
        l7 = H[-7] if nh >= 7 else l1; l14 = H[-14] if nh >= 14 else l1
        w7 = H[-7:] if nh >= 7 else H
        w14 = H[-14:] if nh >= 14 else H
        rm7 = w7.mean(); rs7 = w7.std(ddof=1) if len(w7) > 1 else 0.0
        rm14 = w14.mean(); rs14 = w14.std(ddof=1) if len(w14) > 1 else 0.0
        rcv = rs14 / rm14 if rm14 != 0 else 0.0

        X = np.array([[
            l1, l3, l7, l14, rm7, rs7, rm14, rs14, rcv,
            float(fc_date.dayofweek),
            float(fc_date.isocalendar()[1]),
            float(fc_date.month),
            float(1 if fc_date.dayofweek >= 5 else 0),
            *static
        ]])
        pred = max(0.0, float(model.predict(X)[0]))
        forecasts.append(pred)
        hist_demand.append(pred)

    fc_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(HORIZON)]
    hist_dates = sku_data["date"].tolist()
    hist_values = sku_data["demand"].tolist()

    return (hist_dates, hist_values), (fc_dates, forecasts), float(row0["sku_std_demand"])


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MSE 433 — Inventory Optimisation Dashboard",
    page_icon="📦",
    layout="wide",
)

st.title("📦 MSE 433 — Demand Forecast & Inventory Optimisation")

rec_base  = load_recommendations()
clean_df  = load_clean()
interm_df = load_intermittency()
sku_list  = sorted(rec_base["SKU"].unique())

# Join intermittency class onto recommendations
interm_map = interm_df.set_index("item_id")["intermittency_class"]
rec_base["intermittency_class"] = rec_base["SKU"].map(interm_map).fillna("Unknown")

# Most recent date in dataset (for stock label)
feat_df = load_features()
dataset_last_date = feat_df["date"].max().strftime("%Y-%m-%d")

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

selected_sku = st.sidebar.selectbox(
    "Select SKU",
    sku_list,
    index=0,
    help="Choose a SKU to view its 14-day demand forecast",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Date Range")

# Date range picker for historical demand display
import datetime as _dt
feat_dates = feat_df["date"].dropna()
_min_date = feat_dates.min().date()
_max_date = feat_dates.max().date()
date_range = st.sidebar.date_input(
    "Historical window",
    value=(_min_date, _max_date),
    min_value=_min_date,
    max_value=_max_date,
    help="Filter the historical demand window shown in the forecast chart",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Cost Parameters")

ordering_cost = st.sidebar.slider(
    "Ordering cost ($/order)", min_value=10, max_value=200, value=50, step=5
)
holding_cost = st.sidebar.slider(
    "Holding cost ($/unit/yr)", min_value=0.5, max_value=20.0, value=2.0, step=0.5
)
service_level = st.sidebar.slider(
    "Service level (%)", min_value=80, max_value=99, value=95, step=1
)

# ── Model Performance ────────────────────────────────────────────────────
st.sidebar.markdown("---")
with st.sidebar.expander("Model Performance"):
    comp = load_model_comparison()
    best_model = comp["RMSE"].idxmin()
    comp = comp.rename(columns={"RMSE": "CV RMSE"})
    styled_comp = comp.style.format({
        "CV RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE": "{:.1f}"
    }).apply(
        lambda row: [
            "background-color: #d4edda; font-weight: bold" if row.name == best_model
            else "" for _ in row
        ],
        axis=1,
    )
    st.dataframe(styled_comp, use_container_width=True)
    st.caption(f"Best model: **{best_model}** (lowest CV RMSE)")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model**: from-scratch Histogram GBT (XGBoost-style)  \n"
    "**Horizon**: 14-day recursive forecast  \n"
    f"**SKUs**: {len(sku_list):,}"
)

# ── Info note ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.info(
    "**Data updates:** export CSV from WMS → place in `/data/raw/` → "
    "run `python run_pipeline.py` → restart dashboard"
)


# ═══════════════════════════════════════════════════════════════════════════
#  RECALCULATE POLICIES FROM SLIDER VALUES
# ═══════════════════════════════════════════════════════════════════════════

rec = recalculate_policies(rec_base, clean_df, ordering_cost, holding_cost, service_level)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Demand Forecast",
    "📋 Inventory Policy",
    "💰 Cost Comparison",
    "🔬 Model Diagnostics",
])


# ── Tab 1: Demand Forecast ────────────────────────────────────────────────
with tab1:
    st.subheader(f"14-Day Demand Forecast — {selected_sku}")

    result = generate_sku_forecast(selected_sku)
    if result[0] is None:
        st.warning(f"No data found for SKU {selected_sku}")
    else:
        (hist_dates, hist_vals), (fc_dates, fc_vals), std_dev = result

        hist_df = pd.DataFrame({
            "date": hist_dates, "demand": hist_vals, "type": "Historical"
        })
        # Apply date range filter from sidebar
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            dr_start, dr_end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            hist_df = hist_df[(hist_df["date"] >= dr_start) & (hist_df["date"] <= dr_end)]
        fc_arr = np.array(fc_vals)

        # Use step-specific empirical prediction intervals if available
        pi_data = load_prediction_intervals()

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_df["date"], hist_df["demand"], color="#2c3e50", lw=1.5,
                label="Historical demand", alpha=0.8)
        ax.plot(fc_dates, fc_arr, color="#e74c3c", lw=2,
                label="Forecast", marker="o", markersize=4)

        if pi_data is not None and len(pi_data) == HORIZON:
            upper_90 = fc_arr + pi_data["q95"].values
            lower_90 = np.clip(fc_arr + pi_data["q05"].values, 0, None)
            upper_50 = fc_arr + pi_data["q75"].values
            lower_50 = np.clip(fc_arr + pi_data["q25"].values, 0, None)
            ax.fill_between(fc_dates, lower_90, upper_90,
                            alpha=0.12, color="#e74c3c", label="90% PI")
            ax.fill_between(fc_dates, lower_50, upper_50,
                            alpha=0.25, color="#e74c3c", label="50% PI")
        else:
            upper = fc_arr + std_dev
            lower = np.clip(fc_arr - std_dev, 0, None)
            ax.fill_between(fc_dates, lower, upper,
                            alpha=0.2, color="#e74c3c", label="±1 std band")

        ax.axvline(hist_df["date"].iloc[-1], color="gray", ls="--", alpha=0.5,
                   label="Forecast start")
        ax.set_ylabel("Daily Demand (units)")
        ax.set_title(f"SKU {selected_sku} — Historical + 14-Day Forecast",
                     fontweight="bold")
        ax.legend(loc="upper left")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        sku_rec = rec[rec["SKU"] == selected_sku].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("14-Day Total", f"{sku_rec['forecast_14d_total']:.0f} units")
        col2.metric("Avg Daily", f"{sku_rec['mu_demand']:.1f} units")
        col3.metric(
            f"Last recorded stock (as of {dataset_last_date})",
            f"{sku_rec['current_stock']:,}",
        )
        col4.metric("Action", sku_rec["action"])


# ── Tab 2: Inventory Policy ──────────────────────────────────────────────
with tab2:
    st.subheader("Inventory Policy — All SKUs")

    col1, col2, col3 = st.columns(3)
    cat_filter = col1.multiselect(
        "Filter by category",
        options=sorted(rec["category"].unique()),
        default=sorted(rec["category"].unique()),
    )
    action_filter = col2.multiselect(
        "Filter by action",
        options=sorted(rec["action"].unique()),
        default=sorted(rec["action"].unique()),
    )
    interm_filter = col3.multiselect(
        "Filter by intermittency class",
        options=sorted(rec["intermittency_class"].unique()),
        default=sorted(rec["intermittency_class"].unique()),
    )

    filtered = rec[
        (rec["category"].isin(cat_filter))
        & (rec["action"].isin(action_filter))
        & (rec["intermittency_class"].isin(interm_filter))
    ].copy()

    display_cols = [
        "SKU", "category", "intermittency_class",
        "safety_stock", "ROP", "EOQ",
        "current_stock", "mu_demand", "mu_lead",
        "annual_cost_dynamic", "annual_cost_static",
        "cost_saving_pct", "action",
    ]
    filtered_display = filtered[display_cols].sort_values(
        "cost_saving_pct", ascending=False
    )

    st.dataframe(
        filtered_display.style.format({
            "safety_stock": "{:.1f}",
            "ROP": "{:.1f}",
            "EOQ": "{:.1f}",
            "mu_demand": "{:.2f}",
            "mu_lead": "{:.0f}",
            "annual_cost_dynamic": "${:,.0f}",
            "annual_cost_static": "${:,.0f}",
            "cost_saving_pct": "{:.1f}%",
        }).map(
            lambda v: "background-color: #d4edda" if v == "Hold"
            else "background-color: #f8d7da",
            subset=["action"],
        ),
        use_container_width=True,
        height=500,
    )

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total SKUs", f"{len(filtered):,}")
    col2.metric("Reorder Now", f"{(filtered['action'] == 'Reorder Now').sum():,}")
    col3.metric("Avg Cost Saving", f"{filtered['cost_saving_pct'].mean():.1f}%")
    col4.metric("Total Dynamic Cost",
                f"${filtered['annual_cost_dynamic'].sum():,.0f}")


# ── Tab 3: Cost Comparison ───────────────────────────────────────────────
with tab3:
    st.subheader("Cost Comparison — Top 20 SKUs by Saving")

    top20 = rec.nlargest(20, "cost_saving_pct")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(top20))
    width = 0.35

    bars1 = ax.bar(x - width/2, top20["annual_cost_static"].values, width,
                   label="Static Policy", color="#e74c3c", alpha=0.8)
    bars2 = ax.bar(x + width/2, top20["annual_cost_dynamic"].values, width,
                   label="Dynamic Policy", color="#27ae60", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(top20["SKU"].values, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Annual Cost ($)")
    ax.set_title("Annual Inventory Cost: Dynamic vs Static Policy — Top 20 SKUs",
                 fontweight="bold")
    ax.legend()

    for i, (_, row) in enumerate(top20.iterrows()):
        ax.annotate(
            f"-{row['cost_saving_pct']:.0f}%",
            xy=(i, max(row["annual_cost_static"], row["annual_cost_dynamic"])),
            ha="center", va="bottom", fontsize=7, color="#2c3e50", fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    total_dyn = rec["annual_cost_dynamic"].sum()
    total_sta = rec["annual_cost_static"].sum()
    agg_saving = (total_sta - total_dyn) / total_sta * 100 if total_sta > 0 else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Static Cost", f"${total_sta:,.0f}")
    col2.metric("Total Dynamic Cost", f"${total_dyn:,.0f}")
    col3.metric("Aggregate Saving", f"{agg_saving:.1f}%",
                delta=f"-${total_sta - total_dyn:,.0f}")


# ── Tab 4: Model Diagnostics ───────────────────────────────────────────────
with tab4:
    st.subheader("Model Diagnostics")

    import matplotlib.pyplot as plt

    diag1, diag2, diag3 = st.tabs([
        "Feature Importance",
        "Prediction Intervals",
        "Error → Cost Propagation",
    ])

    # ── Diagnostics sub-tab 1: Feature Importance Comparison ────────────
    with diag1:
        st.markdown("#### Gain-Based vs Permutation Importance")
        st.caption(
            "**Gain-based** measures how much each feature reduces loss in tree "
            "splits. **Permutation** measures how much RMSE increases when a "
            "feature is randomly shuffled — a model-agnostic validation."
        )

        perm_data = load_permutation_importance()

        if perm_data is not None:
            col_a, col_b = st.columns(2)

            with col_a:
                # Gain-based importance (from model)
                model = load_model()
                fcols = [
                    "lag_1","lag_3","lag_7","lag_14",
                    "rolling_mean_7","rolling_std_7","rolling_mean_14",
                    "rolling_std_14","rolling_cv_14",
                    "day_of_week","week_of_year","month","is_weekend",
                    "sku_mean_demand","sku_std_demand","sku_mean_lead_time",
                    "sku_reorder_freq_days","ADI","CV2",
                    "category_enc","intermittency_enc",
                ]
                imp = model.feature_importances_
                n = min(len(fcols), len(imp))
                gain_df = pd.DataFrame({
                    "feature": fcols[:n], "importance": imp[:n]
                }).sort_values("importance", ascending=False)

                fig, ax = plt.subplots(figsize=(7, 5))
                t10 = gain_df.head(10)
                ax.barh(t10["feature"].values[::-1],
                        t10["importance"].values[::-1], color="#2c3e50")
                ax.set_title("Gain-Based (Top 10)", fontweight="bold")
                ax.set_xlabel("Importance (gain)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                fig, ax = plt.subplots(figsize=(7, 5))
                t10p = perm_data.head(10)
                ax.barh(t10p["feature"].values[::-1],
                        t10p["importance_mean"].values[::-1],
                        xerr=t10p["importance_std"].values[::-1],
                        color="#e74c3c", capsize=3)
                ax.set_title("Permutation (Top 10)", fontweight="bold")
                ax.set_xlabel("RMSE Increase")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            st.markdown("##### Full Permutation Importance Table")
            st.dataframe(
                perm_data.style.format({
                    "importance_mean": "{:.4f}",
                    "importance_std": "{:.4f}",
                }).bar(subset=["importance_mean"], color="#e74c3c80"),
                use_container_width=True,
            )
        else:
            st.info("Run the pipeline to generate permutation importance data.")

    # ── Diagnostics sub-tab 2: Prediction Intervals ────────────────────
    with diag2:
        st.markdown("#### Prediction Interval Analysis")
        st.caption(
            "Empirical prediction intervals derived from held-out test residuals "
            "at each forecast step. The fan chart shows 50% and 90% intervals."
        )

        pi_data = load_prediction_intervals()

        if pi_data is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            steps = pi_data["step"].values
            # Left: interval width growth
            ax1.plot(steps, pi_data["residual_std"].values, color="#2c3e50",
                     lw=2, marker="o")
            ax1.set_title("Residual Std Dev by Step", fontweight="bold")
            ax1.set_xlabel("Forecast Step (days ahead)")
            ax1.set_ylabel("Residual Std Dev")
            ax1.set_xticks(range(1, HORIZON + 1))

            # Right: quantile bands
            ax2.fill_between(steps, pi_data["q05"].values, pi_data["q95"].values,
                             alpha=0.15, color="#e74c3c", label="90% PI")
            ax2.fill_between(steps, pi_data["q25"].values, pi_data["q75"].values,
                             alpha=0.3, color="#e74c3c", label="50% PI")
            ax2.axhline(0, color="black", lw=0.8, ls="--")
            ax2.plot(steps, pi_data["residual_mean"].values, color="#2c3e50",
                     lw=2, marker="o", label="Mean residual")
            ax2.set_title("Residual Distribution by Step", fontweight="bold")
            ax2.set_xlabel("Forecast Step (days ahead)")
            ax2.set_ylabel("Residual (actual − predicted)")
            ax2.set_xticks(range(1, HORIZON + 1))
            ax2.legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            st.markdown("##### Per-Step Interval Table")
            st.dataframe(
                pi_data.style.format({
                    "residual_mean": "{:.2f}", "residual_std": "{:.2f}",
                    "q05": "{:.2f}", "q10": "{:.2f}", "q25": "{:.2f}",
                    "q75": "{:.2f}", "q90": "{:.2f}", "q95": "{:.2f}",
                }),
                use_container_width=True,
            )
        else:
            st.info("Run the pipeline to generate prediction interval data.")

    # ── Diagnostics sub-tab 3: Error → Cost Propagation ────────────────
    with diag3:
        st.markdown("#### How Forecast Error Impacts Inventory Costs")
        st.caption(
            "Simulates the effect of scaling forecast uncertainty (σ_demand) on "
            "safety stock, reorder decisions, and total annual cost. A multiplier "
            "of 1.0× is the current model; higher values simulate worse forecasts."
        )

        ep_data = load_error_propagation()

        if ep_data is not None:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            mults = ep_data["error_multiplier"].values

            # Left: safety stock
            axes[0].plot(mults, ep_data["mean_safety_stock"].values,
                         color="#2c3e50", lw=2, marker="o")
            axes[0].axvline(1.0, color="gray", ls="--", alpha=0.5)
            axes[0].set_title("Avg Safety Stock", fontweight="bold")
            axes[0].set_xlabel("Error Multiplier")
            axes[0].set_ylabel("Units")

            # Centre: total cost
            costs = ep_data["total_annual_cost"].values
            colours = ["#27ae60" if m <= 1.0 else "#e74c3c" for m in mults]
            axes[1].bar(mults.astype(str), costs, color=colours, edgecolor="white")
            axes[1].set_title("Total Annual Cost", fontweight="bold")
            axes[1].set_xlabel("Error Multiplier")
            axes[1].set_ylabel("Cost ($)")
            axes[1].tick_params(axis="x", rotation=30)

            # Right: reorder count
            axes[2].plot(mults, ep_data["reorder_now_count"].values,
                         color="#8e44ad", lw=2, marker="s")
            axes[2].axvline(1.0, color="gray", ls="--", alpha=0.5)
            axes[2].set_title("Reorder Now Count", fontweight="bold")
            axes[2].set_xlabel("Error Multiplier")
            axes[2].set_ylabel("SKUs")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            # Key insight callout
            baseline_cost = ep_data.loc[
                ep_data["error_multiplier"] == 1.0, "total_annual_cost"
            ].values[0]
            worst_cost = ep_data["total_annual_cost"].max()
            worst_mult = ep_data.loc[
                ep_data["total_annual_cost"].idxmax(), "error_multiplier"
            ]
            pct_increase = (worst_cost - baseline_cost) / baseline_cost * 100

            st.warning(
                f"**Key insight:** If forecast error triples ({worst_mult:.0f}× "
                f"current σ), total annual cost increases by "
                f"**{pct_increase:.1f}%** — from "
                f"${baseline_cost:,.0f} to ${worst_cost:,.0f}. "
                f"This underscores the value of maintaining forecast accuracy."
            )

            st.markdown("##### Full Propagation Table")
            st.dataframe(
                ep_data.style.format({
                    "error_multiplier": "{:.2f}×",
                    "mean_safety_stock": "{:.1f}",
                    "mean_rop": "{:.1f}",
                    "total_annual_cost": "${:,.0f}",
                    "reorder_now_count": "{:,}",
                    "cost_vs_baseline_pct": "{:+.1f}%",
                }),
                use_container_width=True,
            )
        else:
            st.info("Run the pipeline to generate error-cost propagation data.")
