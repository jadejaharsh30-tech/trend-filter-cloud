import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


# ==============================
# CONFIG
# ==============================

PARQUET_FILE = "data/weekly_trend_factor.parquet"
TOP_N = 20

st.set_page_config(
    page_title="Trend Factor Dashboard",
    layout="wide"
)

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    df = pd.read_parquet(PARQUET_FILE)
    # Safety check
    if "rebalance_week" not in df.columns:
        df["rebalance_week"] = df["date"].dt.to_period("W-FRI")
    df["rebalance_week"] = df["rebalance_week"].astype(str)
    return df

df = load_data()

def run_weekly_backtest(df, top_pct=0.2):
    """
    Weekly long-only backtest using factor ranking.
    """
    df = df.sort_values(["rebalance_week", "ticker"])

    portfolio_returns = []

    weeks = sorted(df["rebalance_week"].unique())

    for i in range(len(weeks) - 1):
        this_week = weeks[i]
        next_week = weeks[i + 1]

        # Snapshot at rebalance
        snap = df[df["rebalance_week"] == this_week].copy()

        # Select top percentile
        cutoff = snap["percentile"].quantile(1 - top_pct)
        longs = snap[snap["percentile"] >= cutoff]

        if longs.empty:
            portfolio_returns.append(0.0)
            continue

        # Next week prices
        next_prices = df[
            (df["rebalance_week"] == next_week) &
            (df["ticker"].isin(longs["ticker"]))
        ][["ticker", "adj_close"]]

        current_prices = snap[
            snap["ticker"].isin(longs["ticker"])
        ][["ticker", "adj_close"]]

        merged = current_prices.merge(
            next_prices,
            on="ticker",
            suffixes=("_cur", "_next")
        )

        merged["ret"] = (
            merged["adj_close_next"] /
            merged["adj_close_cur"] - 1
        )

        portfolio_returns.append(merged["ret"].mean())

    returns = pd.Series(portfolio_returns, index=weeks[:-1])
    return returns

def performance_metrics(returns, freq=52):
    ann_return = (1 + returns).prod() ** (freq / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd = drawdown.min()

    return ann_return, ann_vol, sharpe, max_dd, cum

# ==============================
# SIDEBAR CONTROLS
# ==============================

st.sidebar.title("Controls")

weeks = sorted(df["rebalance_week"].unique())
selected_week = st.sidebar.selectbox(
    "Rebalance Week",
    weeks,
    index=len(weeks) - 1
)

subset = df[df["rebalance_week"] == selected_week].copy()

subset = subset.sort_values("factor", ascending=True)

# ==============================
# HEADER
# ==============================

st.title("📈 Exponential Trend Factor Dashboard")
st.caption(
    "Factor = Annualized Exponential Regression Slope × R² | Weekly Rebalance"
)

# ==============================
# KPI ROW
# ==============================

col1, col2, col3, col4 = st.columns(4)

col1.metric("Universe Size", subset["ticker"].nunique())
col2.metric("Top Factor", round(subset["factor"].max(), 2))
col3.metric("Median Factor", round(subset["factor"].median(), 2))
col4.metric("Bottom Factor", round(subset["factor"].min(), 2))

# ==============================
# TOP / BOTTOM TABLES
# ==============================

st.subheader("🏆 Top Trend Leaders")

st.dataframe(
    subset.head(TOP_N)[
        ["ticker", "factor", "percentile", "annualized_slope", "r2"]
    ],
    width='stretch'
)

st.subheader("🔻 Bottom Trend Laggards")

st.dataframe(
    subset.tail(TOP_N)[
        ["ticker", "factor", "percentile", "annualized_slope", "r2"]
    ],
    width='stretch'
)

# ==============================
# FACTOR DISTRIBUTION
# ==============================

st.subheader("📊 Factor Distribution")

fig_dist = px.histogram(
    subset,
    x="factor",
    nbins=50,
    title="Cross-Sectional Distribution of Trend Factor"
)

st.plotly_chart(fig_dist, width='stretch')

# ==============================
# STOCK DEEP DIVE
# ==============================

st.subheader("🔍 Stock Deep-Dive")

selected_stock = st.selectbox(
    "Select Stock",
    sorted(df["ticker"].unique())
)

stock_df = df[df["ticker"] == selected_stock].sort_values("date")

fig_ts = px.line(
    stock_df,
    x="date",
    y="factor",
    title=f"Trend Factor Over Time – {selected_stock}"
)

st.plotly_chart(fig_ts, width='stretch')

# ==============================
# COMPONENT BREAKDOWN
# ==============================

st.subheader("🧩 Factor Components")

fig_components = px.line(
    stock_df,
    x="date",
    y=["annualized_slope", "r2"],
    title=f"Slope and R² Over Time – {selected_stock}"
)

st.plotly_chart(fig_components, width='stretch')

# ==============================
# FOOTER
# ==============================

st.caption(
    "Built for systematic trend and momentum research | Weekly cross-sectional factor"
)


# ==============================
# PORTFOLIO BACKTEST
# ==============================

st.subheader("📉 Portfolio Backtest")

top_pct = st.slider(
    "Top percentile to hold (long-only)",
    min_value=0.05,
    max_value=0.5,
    value=0.2,
    step=0.05
)

@st.cache_data
def run_backtest_cached(df, top_pct):
    return run_weekly_backtest(df, top_pct)

with st.spinner("Running backtest..."):
    returns = run_backtest_cached(df, top_pct)

ann_ret, ann_vol, sharpe, max_dd, cum_curve = performance_metrics(returns)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annualized Return", f"{ann_ret:.2%}")
col2.metric("Annualized Volatility", f"{ann_vol:.2%}")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}")
col4.metric("Max Drawdown", f"{max_dd:.2%}")

fig_pnl = px.line(
    cum_curve,
    title="Cumulative Portfolio Value (Weekly Rebalanced)",
    labels={"value": "Portfolio Value", "index": "Week"}
)

st.plotly_chart(fig_pnl, width='stretch')
