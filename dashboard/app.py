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

HORIZON = 14

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
def generate_sku_forecast(sku_id: str):
    """Generate 14-day recursive forecast for a single SKU."""
    df = load_features()
    model = load_model()

    sku_data = df[df["item_id"] == sku_id].sort_values("date").copy()
    if sku_data.empty:
        return None, None, None

    # Encode categoricals
    cat_map = {c: i for i, c in enumerate(df["category"].unique())}
    int_map = {c: i for i, c in enumerate(df["intermittency_class"].unique())}

    # Historical demand
    hist_demand = sku_data["demand"].values.tolist()
    last_date = sku_data["date"].max()

    # Static features
    row0 = sku_data.iloc[0]
    static = np.array([
        row0["sku_mean_demand"], row0["sku_std_demand"],
        row0["sku_mean_lead_time"], row0["sku_reorder_freq_days"],
        row0["ADI"], row0["CV2"],
        cat_map.get(row0["category"], 0),
        int_map.get(row0["intermittency_class"], 0),
    ])

    # Recursive forecast
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

rec = load_recommendations()
sku_list = sorted(rec["SKU"].unique())

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Controls")

selected_sku = st.sidebar.selectbox(
    "Select SKU",
    sku_list,
    index=0,
    help="Choose a SKU to view its 14-day demand forecast",
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

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model**: from-scratch Histogram GBT (XGBoost-style)  \n"
    "**Horizon**: 14-day recursive forecast  \n"
    f"**SKUs**: {len(sku_list):,}"
)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📈 Demand Forecast",
    "📋 Inventory Policy",
    "💰 Cost Comparison",
])


# ── Tab 1: Demand Forecast ────────────────────────────────────────────────
with tab1:
    st.subheader(f"14-Day Demand Forecast — {selected_sku}")

    result = generate_sku_forecast(selected_sku)
    if result[0] is None:
        st.warning(f"No data found for SKU {selected_sku}")
    else:
        (hist_dates, hist_vals), (fc_dates, fc_vals), std_dev = result

        # Build chart dataframe
        hist_df = pd.DataFrame({
            "date": hist_dates, "demand": hist_vals, "type": "Historical"
        })
        fc_arr = np.array(fc_vals)
        fc_df = pd.DataFrame({
            "date": fc_dates,
            "demand": fc_arr,
            "upper": fc_arr + std_dev,
            "lower": np.clip(fc_arr - std_dev, 0, None),
            "type": "Forecast"
        })

        # Use matplotlib for better control
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hist_df["date"], hist_df["demand"], color="#2c3e50", lw=1.5,
                label="Historical demand", alpha=0.8)
        ax.plot(fc_df["date"], fc_df["demand"], color="#e74c3c", lw=2,
                label="Forecast", marker="o", markersize=4)
        ax.fill_between(fc_df["date"], fc_df["lower"], fc_df["upper"],
                        alpha=0.2, color="#e74c3c", label="±1 std confidence band")
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

        # Summary metrics
        sku_rec = rec[rec["SKU"] == selected_sku].iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("14-Day Total", f"{sku_rec['forecast_14d_total']:.0f} units")
        col2.metric("Avg Daily", f"{sku_rec['mu_demand']:.1f} units")
        col3.metric("Current Stock", f"{sku_rec['current_stock']:,}")
        col4.metric("Action", sku_rec["action"])


# ── Tab 2: Inventory Policy ──────────────────────────────────────────────
with tab2:
    st.subheader("Inventory Policy — All SKUs")

    # Filter options
    col1, col2 = st.columns(2)
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

    filtered = rec[
        (rec["category"].isin(cat_filter)) & (rec["action"].isin(action_filter))
    ].copy()

    # Display columns
    display_cols = [
        "SKU", "category", "safety_stock", "ROP", "EOQ",
        "current_stock", "mu_demand", "mu_lead",
        "annual_cost_dynamic", "annual_cost_static",
        "cost_saving_pct", "action"
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
        }).applymap(
            lambda v: "background-color: #d4edda" if v == "Hold" else "background-color: #f8d7da",
            subset=["action"]
        ),
        use_container_width=True,
        height=500,
    )

    # Summary stats
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

    # Add saving % labels
    for i, (_, row) in enumerate(top20.iterrows()):
        ax.annotate(
            f"-{row['cost_saving_pct']:.0f}%",
            xy=(i, max(row["annual_cost_static"], row["annual_cost_dynamic"])),
            ha="center", va="bottom", fontsize=7, color="#2c3e50", fontweight="bold"
        )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Aggregate summary
    st.markdown("---")
    total_dyn = rec["annual_cost_dynamic"].sum()
    total_sta = rec["annual_cost_static"].sum()
    agg_saving = (total_sta - total_dyn) / total_sta * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Static Cost", f"${total_sta:,.0f}")
    col2.metric("Total Dynamic Cost", f"${total_dyn:,.0f}")
    col3.metric("Aggregate Saving", f"{agg_saving:.1f}%",
                delta=f"-${total_sta - total_dyn:,.0f}")
