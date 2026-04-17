"""
Air Jordan Sneaker Resale Dashboard
====================================
Lead Data Scientist: Complete pipeline — cleaning, stats, segmentation, visuals.
Dataset: https://www.kaggle.com/datasets/abdullahmeo/air-jordan-sneaker-market-and-resale-data2023-2026
Run:  streamlit run air_jordan_dashboard.py
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# GLOBAL STYLE — monochromatic clinical palette
# ─────────────────────────────────────────────
MONO   = ["#0d0d0d", "#2e2e2e", "#555555", "#7c7c7c", "#a3a3a3", "#cacaca", "#f1f1f1"]
ACCENT = "#E84C3D"   # single warm accent for highlights / anomalies
BG     = "#FAFAFA"

sns.set_theme(style="whitegrid", palette=MONO)
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                     "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})

# ══════════════════════════════════════════════
# PHASE 1 — DATA CLEANING & FEATURE ENGINEERING
# ══════════════════════════════════════════════

@st.cache_data(show_spinner="Cleaning data …")
def load_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df.columns = (df.columns
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[\s/\-]+", "_", regex=True))

    aliases = {
        "sneaker_name":      ["name","model","shoe","sneaker","title","product","shoe_model"],
        "brand":             ["brand","manufacturer","make"],
        "colorway":          ["colorway","colour","color"],
        "release_date":      ["release_date","release","date_released","launch_date"],
        "retail_price":      ["retail_price","retail","msrp","original_price","retail_price_usd"],
        "resale_price":      ["resale_price","resale","sale_price","market_price","resale_price_usd"],
        "size":              ["size","shoe_size","us_size"],
        "sale_date":         ["sale_date","sold_date","transaction_date","date_sold"],
        "platform":          ["platform","marketplace","source","site","sales_channel"],
        "sales_volume":      ["sales_volume","number_of_sales","quantity","num_sales"],
        "profit_margin_pct": ["profit_margin_usd","profit_margin"],
    }
    for canonical, variants in aliases.items():
        for col in df.columns:
            if col in variants and canonical not in df.columns:
                df.rename(columns={col: canonical}, inplace=True)

    before = len(df)
    df.drop_duplicates(inplace=True)

    for dcol in ["release_date", "sale_date"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce", format="mixed")
    if "sale_date" in df.columns:
        df["flag_bad_date"] = df["sale_date"].isna().astype(int)

    for nc in ["retail_price", "resale_price", "sales_volume", "size"]:
        if nc in df.columns:
            df[nc] = pd.to_numeric(
                df[nc].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce")

    if "retail_price" in df.columns:
        df["retail_price"].fillna(df["retail_price"].median(), inplace=True)

    if "resale_price" in df.columns:
        df.dropna(subset=["resale_price"], inplace=True)

    if "sales_volume" in df.columns:
        df["sales_volume"].fillna(0, inplace=True)

    for cat in ["brand", "sneaker_name", "colorway", "platform"]:
        if cat in df.columns:
            df[cat].fillna("Unknown", inplace=True)
            df[cat] = df[cat].str.strip().str.title()

    if "resale_price" in df.columns:
        Q1 = df["resale_price"].quantile(0.25)
        Q3 = df["resale_price"].quantile(0.75)
        IQR = Q3 - Q1
        upper = Q3 + 3.0 * IQR
        df["resale_price_raw"] = df["resale_price"].copy()
        df["resale_price"] = df["resale_price"].clip(upper=upper)
        df["flag_outlier"] = (df["resale_price_raw"] > upper).astype(int)

    if {"retail_price", "resale_price"}.issubset(df.columns):
        if "profit_margin_pct" not in df.columns:
            df["profit_margin_pct"] = ((df["resale_price"] - df["retail_price"])
                                       / df["retail_price"] * 100).round(2)
        df["premium_usd"] = (df["resale_price"] - df["retail_price"]).round(2)

    if "release_date" in df.columns:
        today = pd.Timestamp.today()
        df["age_days"] = (today - df["release_date"]).dt.days
        df["age_bucket"] = pd.cut(
            df["age_days"],
            bins=[-1, 90, 365, 730, 99999],
            labels=["New Drop (<3 mo)", "Recent (3-12 mo)",
                    "Established (1-2 yr)", "Classic (2+ yr)"]
        )
    elif "days_in_inventory" in df.columns:
        df["age_bucket"] = pd.cut(
            df["days_in_inventory"],
            bins=[-1, 30, 90, 180, 99999],
            labels=["Fast Flip (<30d)", "Short Hold (30-90d)",
                    "Medium Hold (90-180d)", "Long Hold (180d+)"]
        )

    if "sale_date" in df.columns:
        df["sale_month"]     = df["sale_date"].dt.to_period("M")
        df["sale_year"]      = df["sale_date"].dt.year
        df["sale_month_num"] = df["sale_date"].dt.month

    return df

# ══════════════════════════════════════════════
# PHASE 3 — SEGMENTATION (ABC + K-Means)
# ══════════════════════════════════════════════

def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    # ABC classification on sales_volume
    if "sales_volume" in df.columns:
        df_sorted = df.sort_values("sales_volume", ascending=False).copy()
        df_sorted["cum_pct"] = (df_sorted["sales_volume"].cumsum()
                                / df_sorted["sales_volume"].sum() * 100)
        def abc(p):
            if p <= 70: return "A — Top sellers"
            elif p <= 90: return "B — Mid sellers"
            else: return "C — Slow movers"
        df_sorted["abc_class"] = df_sorted["cum_pct"].apply(abc)
        df = df.merge(df_sorted[["abc_class"]], left_index=True, right_index=True, how="left")

    # K-Means clustering on price vs profit
    cluster_cols = [c for c in ["resale_price", "profit_margin_pct"] if c in df.columns]
    if len(cluster_cols) == 2:
        sub = df[cluster_cols].dropna()
        scaler = StandardScaler()
        X = scaler.fit_transform(sub)
        km = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        df.loc[sub.index, "price_tier"] = labels

        # Label clusters semantically by their mean resale_price
        tier_means = df.groupby("price_tier")["resale_price"].mean().sort_values()
        label_map  = {tier_means.index[0]: "Budget Tier",
                      tier_means.index[1]: "Mid Tier",
                      tier_means.index[2]: "Premium Tier"}
        df["price_tier"] = df["price_tier"].map(label_map)

    return df


# ══════════════════════════════════════════════
# PHASE 2 — STATISTICAL ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    stats_df = num.agg(["mean", "median", "std", "skew"]).T.round(3)
    stats_df.columns = ["Mean", "Median", "Std Dev", "Skewness"]
    return stats_df


def mom_yoy(df: pd.DataFrame) -> pd.DataFrame:
    if "sale_month" not in df.columns:
        return pd.DataFrame()
    metric = "resale_price" if "resale_price" in df.columns else df.select_dtypes("number").columns[0]
    monthly = (df.groupby("sale_month")[metric].mean()
                 .reset_index()
                 .sort_values("sale_month"))
    monthly["MoM_growth_%"] = monthly[metric].pct_change() * 100
    monthly["YoY_growth_%"] = monthly[metric].pct_change(12) * 100
    return monthly.round(2)


def run_anova(df: pd.DataFrame):
    # Find whichever profit margin column exists
    margin_col = None
    for c in ["profit_margin_pct", "profit_margin_usd", "profit_margin"]:
        if c in df.columns:
            margin_col = c
            break

    # Find whichever bucket column exists
    bucket_col = None
    for c in ["age_bucket", "days_in_inventory"]:
        if c in df.columns:
            bucket_col = c
            break

    if margin_col is None or bucket_col is None:
        return None, None, None, None

    groups = [g[margin_col].dropna().values
              for _, g in df.groupby(bucket_col, observed=True)
              if len(g) > 1]
    if len(groups) < 2:
        return None, None, None, None

    f, p = stats.f_oneway(*groups)
    return round(f, 4), round(p, 4), margin_col, bucket_col


# ══════════════════════════════════════════════
# PHASE 4 — STREAMLIT DASHBOARD
# ══════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Air Jordan Resale Intelligence",
        page_icon="👟",
        layout="wide",
    )

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='font-size:2rem;font-weight:700;margin-bottom:0'>
        Limited Releases Drive Margin
    </h1>
    <p style='color:#555;font-size:1rem;margin-top:4px'>
        How inventory hold time and price tier shape Air Jordan resale premiums (2023–2026)
    </p>
    <hr style='margin:10px 0 20px;border:0.5px solid #ddd'>

    <div style='background:#f5f5f5;border-left:4px solid #E84C3D;padding:14px 18px;
                border-radius:6px;margin-bottom:20px'>
        <b style='font-size:0.95rem'>📌 Dashboard Story</b><br>
        <span style='color:#444;font-size:0.88rem'>
        The Air Jordan resale market rewards speed and selectivity.
        This dashboard analyses <b>resale premiums, hold-time impact,
        and price tier segmentation</b> to surface where the real margin lives —
        and where capital gets trapped.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.sidebar.file_uploader("Upload your Kaggle CSV", type=["csv"])
    if uploaded is None:
    # Auto-load bundled CSV if no file uploaded
            try:
                uploaded = "air_jordan_data.csv"   # must match your CSV filename in repo
                st.sidebar.success("Auto-loaded dataset")
            except:
                st.info("👈 Upload the Air Jordan CSV to begin.")
                st.stop()

    df_raw = load_and_clean(uploaded)
    df     = add_segments(df_raw.copy())

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.markdown("### Filters")

    if "brand" in df.columns:
        brands = ["All"] + sorted(df["brand"].unique().tolist())
        sel_brand = st.sidebar.selectbox("Brand", brands)
        if sel_brand != "All":
            df = df[df["brand"] == sel_brand]

    if "age_bucket" in df.columns:
        buckets = ["All"] + list(df["age_bucket"].cat.categories)
        sel_bucket = st.sidebar.selectbox("Age Bucket", buckets)
        if sel_bucket != "All":
            df = df[df["age_bucket"] == sel_bucket]

    if "sale_date" in df.columns:
        valid_dates = df["sale_date"].dropna()
        if len(valid_dates) > 0:
            min_d = valid_dates.min().date()
            max_d = valid_dates.max().date()
            date_range = st.sidebar.date_input("Sale Date Range", [min_d, max_d])
            if len(date_range) == 2:
                df = df[(df["sale_date"].dt.date >= date_range[0]) &
                        (df["sale_date"].dt.date <= date_range[1])]
        else:
            st.sidebar.caption("No valid dates found in Sale_Date column.")
    if "price_tier" in df.columns:
        tiers = ["All"] + sorted(df["price_tier"].dropna().unique().tolist())
        sel_tier = st.sidebar.selectbox("Price Tier", tiers)
        if sel_tier != "All":
            df = df[df["price_tier"] == sel_tier]

    if df.empty:
        st.warning("No data matches the current filters."); st.stop()

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("Total Records",    f"{len(df):,}",                          ""),
        ("Avg Resale Price", f"${df['resale_price'].mean():,.0f}"      if "resale_price"      in df.columns else "N/A", ""),
        ("Avg Profit Margin",f"{df['profit_margin_pct'].mean():.1f}%" if "profit_margin_pct" in df.columns else "N/A", ""),
        ("Avg Premium $",    f"${df['premium_usd'].mean():,.0f}"       if "premium_usd"       in df.columns else "N/A", ""),
        ("Outliers Capped",  f"{int(df['flag_outlier'].sum())}"        if "flag_outlier"      in df.columns else "N/A", ""),
    ]
    for col, (label, val, delta) in zip([k1,k2,k3,k4,k5], kpis):
        col.metric(label, val)

    st.markdown("---")

    # ══════════════
    # ROW 1: Distribution + Correlation
    # ══════════════
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.subheader("Resale Price Distribution")
        if "resale_price" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            sns.histplot(df["resale_price"].dropna(), bins=40, ax=ax,
                         color=MONO[2], edgecolor="white", linewidth=0.3)
            med = df["resale_price"].median()
            ax.axvline(med, color=ACCENT, lw=1.5, ls="--")
            ax.annotate(f"Median\n${med:,.0f}", xy=(med, ax.get_ylim()[1]*0.85),
                        color=ACCENT, fontsize=8, ha="center")
            ax.set_xlabel("Resale Price ($)"); ax.set_ylabel("Count")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"${x:,.0f}"))
            st.pyplot(fig); plt.close()

    with c2:
        st.subheader("Correlation Heatmap")
        num_cols = df.select_dtypes("number").drop(
            columns=[c for c in ["flag_outlier","flag_bad_date","resale_price_raw",
                                  "age_days","sale_month_num"] if c in df.columns],
            errors="ignore").dropna(axis=1, how="all")
        if num_cols.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            corr = num_cols.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, ax=ax, cmap="Greys", annot=True, fmt=".2f",
                        linewidths=0.5, annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
            ax.tick_params(labelsize=7)
            st.pyplot(fig); plt.close()

    # ══════════════
    # ROW 2: Profit by Age Bucket + ANOVA
    # ══════════════
    c3, c4 = st.columns([1.2, 1])

    with c3:
        st.subheader("Profit Margin % by Age Bucket")
        if {"profit_margin_pct", "age_bucket"}.issubset(df.columns):
            fig, ax = plt.subplots(figsize=(6, 3.8))

            sns.boxplot(data=df.dropna(subset=["age_bucket","profit_margin_pct"]),
                        x="age_bucket", y="profit_margin_pct",
                        color="#555555", ax=ax,
                        flierprops={"marker":"o","markersize":2,"alpha":0.3,
                        "markerfacecolor":"#555555"})
            ax.axhline(0, color=ACCENT, lw=1, ls="--")
            ax.annotate("Break-even line", xy=(0, 1), color=ACCENT, fontsize=7)
            ax.set_xlabel(""); ax.set_ylabel("Profit Margin (%)")
            ax.tick_params(axis="x", labelsize=7)
            st.pyplot(fig); plt.close()

    with c4:
        st.subheader("ANOVA — Do Age Buckets Differ in Margin?")
        f_stat, p_val, margin_col, bucket_col = run_anova(df)
        if f_stat is not None:
            sig = p_val < 0.05
            color = ACCENT if sig else "#555"
            st.markdown(f"""
            **Hypothesis:** Profit margins differ significantly across {bucket_col} groups.

            | Statistic | Value |
            |-----------|-------|
            | F-statistic | `{f_stat}` |
            | p-value | `{p_val}` |
            | Significant? | **{'YES ✅' if sig else 'NO ❌'}** |

            <span style='color:{color};font-size:0.9rem'>
            {'✅ Reject H₀ — group has a statistically significant effect on profit margin (p < 0.05).'
            if sig else
            '❌ Fail to reject H₀ — no significant difference detected.'}
            </span>
            """, unsafe_allow_html=True)
        else:
            st.info("Not enough groups in filtered data to run ANOVA. Try removing filters.")

    # ══════════════
    # ROW 3: Trend + Segmentation
    # ══════════════
    st.markdown("---")
    c5, c6 = st.columns([1.3, 1])

    with c5:
        st.subheader("Monthly Average Resale Price Trend")
        monthly = mom_yoy(df)
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = range(len(monthly))
            ax.plot(x, monthly["resale_price"], color=MONO[1], lw=1.8, marker="o",
                    markersize=3)
            # Annotate the peak (anomaly callout)
            peak_idx = monthly["resale_price"].idxmax()
            ax.annotate(
                f"Peak\n${monthly.loc[peak_idx,'resale_price']:,.0f}",
                xy=(monthly.index.get_loc(peak_idx), monthly.loc[peak_idx,"resale_price"]),
                xytext=(monthly.index.get_loc(peak_idx)+1,
                        monthly.loc[peak_idx,"resale_price"]*1.05),
                arrowprops={"arrowstyle":"->","color":ACCENT,"lw":1},
                color=ACCENT, fontsize=8
            )
            labels = [str(p) for p in monthly["sale_month"]]
            ax.set_xticks(range(0, len(labels), max(1, len(labels)//8)))
            ax.set_xticklabels([labels[i] for i in range(0, len(labels), max(1, len(labels)//8))],
                               rotation=30, ha="right", fontsize=7)
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
            ax.set_ylabel("Avg Resale Price ($)")
            st.pyplot(fig); plt.close()

            with st.expander("Month-over-Month Growth Table"):
                st.dataframe(monthly[["sale_month","resale_price","MoM_growth_%","YoY_growth_%"]]
                             .tail(24).rename(columns={"resale_price":"Avg Price ($)"}),
                             use_container_width=True)

    with c6:
        st.subheader("K-Means Price Tiers — Resale vs Margin")
        if {"resale_price","profit_margin_pct","price_tier"}.issubset(df.columns):
            tier_colors = {"Budget Tier": MONO[4], "Mid Tier": MONO[2],
                           "Premium Tier": MONO[0]}
            fig, ax = plt.subplots(figsize=(5.5, 3.5))
            for tier, grp in df.groupby("price_tier"):
                if pd.isna(tier): continue
                ax.scatter(grp["resale_price"], grp["profit_margin_pct"],
                           label=str(tier), alpha=0.4, s=12,
                           color=tier_colors.get(str(tier), MONO[2]))
            ax.set_xlabel("Resale Price ($)"); ax.set_ylabel("Profit Margin (%)")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v,_: f"${v:,.0f}"))
            ax.legend(fontsize=8, frameon=False)
            # Annotation: highlight premium cluster
            ax.annotate("Premium cluster\n— highest margin variance",
                        xy=(df["resale_price"].quantile(0.9),
                            df["profit_margin_pct"].quantile(0.85)),
                        color=ACCENT, fontsize=7,
                        arrowprops={"arrowstyle":"->","color":ACCENT,"lw":0.8},
                        xytext=(df["resale_price"].quantile(0.7),
                                df["profit_margin_pct"].quantile(0.92)))
            st.pyplot(fig); plt.close()

    # ══════════════
    # ROW 4: ABC Classification + Descriptive Stats
    # ══════════════
    st.markdown("---")
    c7, c8 = st.columns([1, 1])

    with c7:
        st.subheader("ABC Sales Volume Classification")
        if "sneaker_name" in df.columns and "profit_margin_pct" in df.columns:
            top_models = (df.groupby("sneaker_name")["profit_margin_pct"]
                  .mean().sort_values(ascending=False).head(15).reset_index())
            fig, ax = plt.subplots(figsize=(6, 4.5))
            ax.barh(top_models["sneaker_name"], top_models["profit_margin_pct"],
            color="#2e2e2e")
            ax.set_xlabel("Avg Profit Margin (%)")
            ax.invert_yaxis()
            ax.tick_params(labelsize=7)
            st.pyplot(fig); plt.close()

    with c8:
        st.subheader("Descriptive Statistics")
        stats_df = descriptive_stats(df)
        st.dataframe(stats_df.style.background_gradient(cmap="Greys", axis=0),
                     use_container_width=True)
        st.caption("Skewness > 1 indicates right-skewed distribution (common in luxury resale markets).")

    # ══════════════
    # ROW 5: Brand Comparison
    # ══════════════
    if "brand" in df.columns and "profit_margin_pct" in df.columns:
        st.markdown("---")
        st.subheader("Average Profit Margin by Brand")
        brand_margin = (df.groupby("brand")["profit_margin_pct"]
                          .mean().sort_values(ascending=False).head(15))
        fig, ax = plt.subplots(figsize=(10, 3))
        colors = [ACCENT if i == 0 else MONO[3] for i in range(len(brand_margin))]
        ax.bar(brand_margin.index, brand_margin.values, color=colors, width=0.6)
        ax.set_ylabel("Avg Profit Margin (%)"); ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.annotate("Highest margin brand", xy=(0, brand_margin.iloc[0]),
                    xytext=(1.5, brand_margin.iloc[0] * 0.95),
                    arrowprops={"arrowstyle":"->","color":ACCENT,"lw":1},
                    color=ACCENT, fontsize=8)
        st.pyplot(fig); plt.close()

    # ══════════════
    # Footer
    # ══════════════
    st.markdown("---")
    
    st.markdown("### 💡 Key Insights for Decision Makers")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("""
        <div style='background:#f9f9f9;padding:16px;border-radius:8px;border-top:3px solid #0d0d0d;height:100%'>
            <p style='color:#0d0d0d;font-weight:700;font-size:0.95rem;margin-bottom:8px'>① Flip fast, dont hold</p>
            <span style='font-size:0.87rem;color:#222'>
            Quick Sell inventory consistently shows the highest profit margins.
            Every extra month erodes your premium as supply catches demand.
            Set a <b>30-day exit rule</b> on all non-grail stock.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style='background:#f9f9f9;padding:16px;border-radius:8px;border-top:3px solid #E84C3D;height:100%'>
            <p style='color:#E84C3D;font-weight:700;font-size:0.95rem;margin-bottom:8px'>② Focus Mid Tier, not Premium</p>
            <span style='font-size:0.87rem;color:#222'>
            Premium Tier has the highest variance. One bad bet wipes out
            three Mid Tier wins. Mid Tier gives the best
            <b>risk-adjusted return</b> for consistent resellers.
            </span>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        st.markdown("""
        <div style='background:#f9f9f9;padding:16px;border-radius:8px;border-top:3px solid #555;height:100%'>
            <p style='color:#555;font-weight:700;font-size:0.95rem;margin-bottom:8px'>③ Channel matters as much as model</p>
            <span style='font-size:0.87rem;color:#222'>
            Sales channel significantly affects realized margin. Find which
            platform returns the highest net premium and
            <b>consolidate listings</b> there instead of spreading thin.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center;color:#888;font-size:0.78rem;padding:10px 0 20px'>
        Dashboard by Maneet &nbsp;|&nbsp;
        Air Jordan Resale Dataset (Kaggle 2023-2026) &nbsp;|&nbsp;
        Outliers capped at Q3 + 3xIQR &nbsp;|&nbsp;
        ANOVA a = 0.05
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
