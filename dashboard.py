from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


# ==============================
# CONFIG
# ==============================

PARQUET_FILE = "data/weekly_trend_factor.parquet"
TOP_N = 20
REQUIRED_COLUMNS = {
    "date", "ticker", "adj_close", "factor", "percentile", "annualized_slope", "r2", "rebalance_week"
}

st.set_page_config(
    page_title="Trend Factor Dashboard",
    layout="wide"
)


# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    parquet_path = Path(PARQUET_FILE)

    if not parquet_path.exists():
        st.error(
            "Data file not found. Generate it first with: `python trend_factor.py` "
            f"(expected at `{PARQUET_FILE}`)."
        )
        st.stop()

    df = pd.read_parquet(parquet_path)

    if "rebalance_week" not in df.columns and "date" in df.columns:
        df["rebalance_week"] = df["date"].dt.to_period("W-FRI")

    missing_cols = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_cols:
        st.error(
            "Input data is missing required columns: "
            + ", ".join(missing_cols)
            + ". Rebuild the factor dataset with `python trend_factor.py`."
        )
        st.stop()

    df["rebalance_week"] = df["rebalance_week"].astype(str)
    return df


df = load_data()


def run_weekly_backtest(df, top_pct=0.2, transaction_cost_bps=10, missing_next_week="drop"):
    """
    Weekly long-only backtest using factor ranking.

    Args:
        top_pct: Top percentile to hold.
        transaction_cost_bps: Round-trip cost estimate in basis points.
        missing_next_week: How to handle constituents missing next-week prices.
            - "drop": ignore missing holdings in weekly return average
            - "cash": include missing holdings at 0% return
    """
    df = df.sort_values(["rebalance_week", "ticker"])

    portfolio_returns = []
    turnover_series = []
    prev_holdings = set()

    weeks = sorted(df["rebalance_week"].unique())

    for i in range(len(weeks) - 1):
        this_week = weeks[i]
        next_week = weeks[i + 1]

        # Snapshot at rebalance
        snap = df[df["rebalance_week"] == this_week].copy()

        # Select top percentile
        cutoff = snap["percentile"].quantile(1 - top_pct)
        longs = snap[snap["percentile"] >= cutoff]

        current_holdings = set(longs["ticker"].tolist())

        if not current_holdings:
            portfolio_returns.append(0.0)
            turnover_series.append(0.0)
            prev_holdings = current_holdings
            continue

        buys = len(current_holdings - prev_holdings)
        sells = len(prev_holdings - current_holdings)
        turnover = (buys + sells) / max(len(current_holdings), 1)
        turnover_series.append(turnover)

        # Next week prices
        next_prices = df[
            (df["rebalance_week"] == next_week) &
            (df["ticker"].isin(current_holdings))
        ][["ticker", "adj_close"]]

        current_prices = snap[
            snap["ticker"].isin(current_holdings)
        ][["ticker", "adj_close"]]

        merged = current_prices.merge(
            next_prices,
            on="ticker",
            suffixes=("_cur", "_next"),
            how="left"
        )

        merged["ret"] = (
            merged["adj_close_next"] /
            merged["adj_close_cur"] - 1
        )

        if missing_next_week == "cash":
            merged["ret"] = merged["ret"].fillna(0.0)
        else:
            merged = merged.dropna(subset=["ret"])

        gross_ret = 0.0 if merged.empty else merged["ret"].mean()
        cost = turnover * (transaction_cost_bps / 10000)
        net_ret = gross_ret - cost

        portfolio_returns.append(net_ret)
        prev_holdings = current_holdings

    returns = pd.Series(portfolio_returns, index=weeks[:-1])
    turnover = pd.Series(turnover_series, index=weeks[:-1])
    return returns, turnover


def performance_metrics(returns, freq=52):
    if len(returns) == 0:
        return np.nan, np.nan, np.nan, np.nan, pd.Series(dtype=float)

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

subset_desc = subset.sort_values("factor", ascending=False)
subset_asc = subset.sort_values("factor", ascending=True)


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
    subset_desc.head(TOP_N)[
        ["ticker", "factor", "percentile", "annualized_slope", "r2"]
    ],
    width='stretch'
)

st.subheader("🔻 Bottom Trend Laggards")

st.dataframe(
    subset_asc.head(TOP_N)[
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

col_a, col_b, col_c = st.columns(3)

top_pct = col_a.slider(
    "Top percentile to hold (long-only)",
    min_value=0.05,
    max_value=0.5,
    value=0.2,
    step=0.05
)

transaction_cost_bps = col_b.slider(
    "Transaction cost (bps, round-trip)",
    min_value=0,
    max_value=100,
    value=10,
    step=1
)

missing_next_week = col_c.selectbox(
    "Missing next-week prices",
    options=["drop", "cash"],
    index=0
)


@st.cache_data
def run_backtest_cached(df, top_pct, transaction_cost_bps, missing_next_week):
    return run_weekly_backtest(df, top_pct, transaction_cost_bps, missing_next_week)


with st.spinner("Running backtest..."):
    returns, turnover = run_backtest_cached(df, top_pct, transaction_cost_bps, missing_next_week)

ann_ret, ann_vol, sharpe, max_dd, cum_curve = performance_metrics(returns)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Annualized Return", f"{ann_ret:.2%}" if pd.notna(ann_ret) else "N/A")
col2.metric("Annualized Volatility", f"{ann_vol:.2%}" if pd.notna(ann_vol) else "N/A")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
col4.metric("Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "N/A")
col5.metric("Avg Weekly Turnover", f"{turnover.mean():.2f}" if len(turnover) else "N/A")

fig_pnl = px.line(
    cum_curve,
    title="Cumulative Portfolio Value (Weekly Rebalanced, Net of Costs)",
    labels={"value": "Portfolio Value", "index": "Week"}
)

st.plotly_chart(fig_pnl, width='stretch')
