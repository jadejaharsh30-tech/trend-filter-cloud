from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from time import perf_counter


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
    ordered = df.sort_values(["rebalance_week", "ticker"])

    week_prices = {}
    week_holdings = {}

    for week, snap in ordered.groupby("rebalance_week", sort=True):
        cutoff = snap["percentile"].quantile(1 - top_pct)
        holdings = set(snap.loc[snap["percentile"] >= cutoff, "ticker"].tolist())
        week_holdings[week] = holdings
        week_prices[week] = snap.set_index("ticker")["adj_close"]

    weeks = sorted(week_holdings.keys())

    portfolio_returns = []
    turnover_series = []
    prev_holdings = set()

    for i in range(len(weeks) - 1):
        this_week = weeks[i]
        next_week = weeks[i + 1]

        current_holdings = week_holdings[this_week]

        if not current_holdings:
            portfolio_returns.append(0.0)
            turnover_series.append(0.0)
            prev_holdings = current_holdings
            continue

        buys = len(current_holdings - prev_holdings)
        sells = len(prev_holdings - current_holdings)
        turnover = (buys + sells) / max(len(current_holdings), 1)
        turnover_series.append(turnover)

        tickers = sorted(current_holdings)
        current_prices = week_prices[this_week].reindex(tickers)
        next_prices = week_prices[next_week].reindex(tickers)

        returns_vec = (next_prices / current_prices) - 1

        if missing_next_week == "cash":
            returns_vec = returns_vec.fillna(0.0)
        else:
            returns_vec = returns_vec.dropna()

        gross_ret = 0.0 if returns_vec.empty else returns_vec.mean()
        cost = turnover * (transaction_cost_bps / 10000)
        net_ret = gross_ret - cost

        portfolio_returns.append(net_ret)
        prev_holdings = current_holdings

    returns = pd.Series(portfolio_returns, index=weeks[:-1])
    turnover = pd.Series(turnover_series, index=weeks[:-1])
    return returns, turnover


def build_holdings_and_trade_log(df, top_pct=0.2, as_of_date=None):
    """Build weekly holdings, trade events, and holding periods per ticker."""
    if as_of_date is None:
        as_of_date = pd.Timestamp.today().normalize()

    ordered = df.sort_values(["rebalance_week", "ticker", "date"])
    weeks = sorted(ordered["rebalance_week"].unique())

    week_dates = (
        ordered.groupby("rebalance_week", as_index=False)["date"]
        .max()
        .set_index("rebalance_week")["date"]
        .to_dict()
    )

    week_holdings = {}
    for week in weeks:
        snap = ordered[ordered["rebalance_week"] == week].copy()
        cutoff = snap["percentile"].quantile(1 - top_pct)
        longs = snap[snap["percentile"] >= cutoff]
        week_holdings[week] = set(longs["ticker"].tolist())

    trade_events = []
    prev_holdings = set()
    for week in weeks:
        current_holdings = week_holdings[week]
        buys = current_holdings - prev_holdings
        sells = prev_holdings - current_holdings
        trade_date = week_dates[week]

        for ticker in sorted(buys):
            trade_events.append(
                {"date": trade_date, "rebalance_week": week, "ticker": ticker, "action": "BUY"}
            )
        for ticker in sorted(sells):
            trade_events.append(
                {"date": trade_date, "rebalance_week": week, "ticker": ticker, "action": "SELL"}
            )

        prev_holdings = current_holdings

    holding_periods = []
    all_tickers = sorted(set(ordered["ticker"].unique()))

    for ticker in all_tickers:
        in_position = False
        start_week = None

        for idx, week in enumerate(weeks):
            is_held = ticker in week_holdings[week]
            prev_week = weeks[idx - 1] if idx > 0 else None

            if is_held and not in_position:
                in_position = True
                start_week = week

            if not is_held and in_position:
                in_position = False
                end_week = prev_week
                start_date = week_dates[start_week]
                end_date = week_dates[end_week]
                holding_periods.append(
                    {
                        "ticker": ticker,
                        "start_rebalance_week": start_week,
                        "end_rebalance_week": end_week,
                        "start_date": start_date,
                        "end_date": end_date,
                        "days_held": (end_date - start_date).days + 1,
                        "status": "closed",
                    }
                )

        if in_position:
            start_date = week_dates[start_week]
            holding_periods.append(
                {
                    "ticker": ticker,
                    "start_rebalance_week": start_week,
                    "end_rebalance_week": "OPEN",
                    "start_date": start_date,
                    "end_date": as_of_date,
                    "days_held": (as_of_date - start_date).days + 1,
                    "status": "open",
                }
            )

    trades_df = pd.DataFrame(trade_events).sort_values(["date", "ticker", "action"]).reset_index(drop=True)
    periods_df = pd.DataFrame(holding_periods).sort_values(["ticker", "start_date"]).reset_index(drop=True)

    if periods_df.empty:
        summary_df = pd.DataFrame(columns=["ticker", "total_days_held", "holding_period_count"])
    else:
        summary_df = (
            periods_df.groupby("ticker", as_index=False)
            .agg(total_days_held=("days_held", "sum"), holding_period_count=("ticker", "size"))
            .sort_values("total_days_held", ascending=False)
            .reset_index(drop=True)
        )

    return trades_df, periods_df, summary_df


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


@st.cache_data
def build_trade_log_cached(df, top_pct, as_of_date):
    return build_holdings_and_trade_log(df, top_pct=top_pct, as_of_date=as_of_date)


with st.spinner("Running backtest..."):
    backtest_start = perf_counter()
    returns, turnover = run_backtest_cached(df, top_pct, transaction_cost_bps, missing_next_week)
    trades_df, periods_df, summary_df = build_holdings_and_trade_log(df, top_pct=top_pct)

ann_ret, ann_vol, sharpe, max_dd, cum_curve = performance_metrics(returns)

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Annualized Return", f"{ann_ret:.2%}" if pd.notna(ann_ret) else "N/A")
col2.metric("Annualized Volatility", f"{ann_vol:.2%}" if pd.notna(ann_vol) else "N/A")
col3.metric("Sharpe Ratio", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
col4.metric("Max Drawdown", f"{max_dd:.2%}" if pd.notna(max_dd) else "N/A")
col5.metric("Avg Weekly Turnover", f"{turnover.mean():.2f}" if len(turnover) else "N/A")
st.caption(f"Backtest computation time: {backtest_runtime_s:.3f}s")

fig_pnl = px.line(
    cum_curve,
    title="Cumulative Portfolio Value (Weekly Rebalanced, Net of Costs)",
    labels={"value": "Portfolio Value", "index": "Week"}
)

st.plotly_chart(fig_pnl, width='stretch')

st.subheader("📒 Trade Log & Holding Durations")

st.caption(
    "Trade log includes weekly BUY/SELL actions. Holding periods show each continuous time "
    "a ticker stayed in the portfolio, including open positions through today."
)

col_t1, col_t2 = st.columns(2)
col_t1.metric("Trade Events", len(trades_df))
col_t2.metric("Unique Tickers Held", summary_df["ticker"].nunique() if not summary_df.empty else 0)

st.markdown("**Per-Ticker Total Holding Days**")
st.dataframe(summary_df, width='stretch')

st.markdown("**Ticker Holding Periods (Start/End Dates)**")
st.dataframe(periods_df, width='stretch')

st.markdown("**Weekly Trade Log**")
st.dataframe(trades_df, width='stretch')

st.download_button(
    "Download trade log CSV",
    data=trades_df.to_csv(index=False),
    file_name="trade_log.csv",
    mime="text/csv",
)

st.download_button(
    "Download holding periods CSV",
    data=periods_df.to_csv(index=False),
    file_name="holding_periods.csv",
    mime="text/csv",
)
