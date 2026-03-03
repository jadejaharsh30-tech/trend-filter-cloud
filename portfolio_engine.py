import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from risk_overlay import generate_exposure_series

# ============================================================
# CONFIG
# ============================================================

DB_PATH = "market_data_yfinance.db"
TABLE_NAME = "ohlcv"
PARQUET_FILE = "data/weekly_trend_factor.parquet"

RISK_FREE_RATE = 0.07
MAX_PERCENTILE = 0.25
RESULTS_FILE = "portfolio_backtest_results_with_regime.csv"


# ============================================================
# LOAD DATA
# ============================================================


def _get_price_column(conn: sqlite3.Connection, table_name: str) -> str:
    cols = pd.read_sql(f"PRAGMA table_info({table_name})", conn)["name"].tolist()
    if "adj_close" in cols:
        return "adj_close"
    if "close" in cols:
        return "close"
    raise ValueError(f"Neither 'adj_close' nor 'close' found in table {table_name}")


def load_factor_data(parquet_file: str = PARQUET_FILE) -> pd.DataFrame:
    df = pd.read_parquet(parquet_file)
    df["date"] = pd.to_datetime(df["date"])

    if "percentile_rank" not in df.columns and "percentile" in df.columns:
        df["percentile_rank"] = df["percentile"]

    df["rebalance_week"] = pd.to_datetime(df["rebalance_week"]).dt.tz_localize(None)
    return df


def load_price_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    price_col = _get_price_column(conn, table_name)
    df = pd.read_sql(
        f"SELECT date, ticker, {price_col} AS close FROM {table_name}",
        conn,
        parse_dates=["date"],
    )
    conn.close()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ============================================================
# PREP RETURNS
# ============================================================


def prepare_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    px = price_df.copy()
    px["return"] = px.groupby("ticker")["close"].pct_change()
    return px.pivot(index="date", columns="ticker", values="return")


# ============================================================
# DROP-OUT ENGINE WITH EXPOSURE
# ============================================================


def run_single_bucket(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    percentile_cutoff: float,
    price_df: pd.DataFrame,
):
    threshold = 1 - percentile_cutoff
    holdings = set()
    portfolio_returns = []

    px = price_df.copy()
    px["dma_100"] = px.groupby("ticker")["close"].transform(lambda x: x.rolling(100).mean())
    px["rebalance_week"] = px["date"].dt.to_period("W-FRI").dt.end_time

    weekly_price = (
        px.sort_values("date")
        .groupby(["ticker", "rebalance_week"], as_index=False)
        .last()
    )
    weekly_lookup = weekly_price.set_index(["rebalance_week", "ticker"])

    rebalance_dates = sorted(pd.to_datetime(factor_df["rebalance_week"]).unique())
    exposure_map = dict(zip(exposure_df["rebalance_week"], exposure_df["exposure"]))

    for i in range(len(rebalance_dates) - 1):
        current_week = rebalance_dates[i]
        next_week = rebalance_dates[i + 1]

        week_slice = factor_df[factor_df["rebalance_week"] == current_week]

        eligible = set()
        for _, row in week_slice.iterrows():
            ticker = row["ticker"]
            if row["percentile_rank"] < threshold:
                continue

            try:
                price_row = weekly_lookup.loc[(current_week, ticker)]
            except KeyError:
                continue

            close = price_row["close"]
            dma_100 = price_row["dma_100"]

            if pd.isna(dma_100):
                continue

            if close > dma_100:
                eligible.add(ticker)

        holdings = (holdings & eligible) | (eligible - holdings)

        if not holdings:
            continue

        mask = (returns_df.index > current_week) & (returns_df.index <= next_week)
        period_returns = returns_df.loc[mask, sorted(holdings)]

        if period_returns.empty:
            continue

        daily_portfolio_ret = period_returns.mean(axis=1)
        exposure = exposure_map.get(current_week, 1.0)
        adjusted_returns = daily_portfolio_ret * exposure

        portfolio_returns.append(adjusted_returns)

    if not portfolio_returns:
        return None

    result = pd.concat(portfolio_returns).sort_index()
    return result


# ============================================================
# PERFORMANCE METRICS
# ============================================================


def compute_metrics(daily_returns: pd.Series) -> dict:
    daily_returns = daily_returns.dropna()
    if daily_returns.empty:
        return {"CAGR": np.nan, "AnnualVol": np.nan, "Sharpe": np.nan, "MaxDrawdown": np.nan}

    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    years = len(daily_returns) / 252
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

    annual_vol = daily_returns.std() * np.sqrt(252)
    excess_return = cagr - RISK_FREE_RATE if pd.notna(cagr) else np.nan
    sharpe = excess_return / annual_vol if annual_vol and annual_vol != 0 else np.nan

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_dd = drawdown.min()

    return {
        "CAGR": cagr,
        "AnnualVol": annual_vol,
        "Sharpe": sharpe,
        "MaxDrawdown": max_dd,
    }


# ============================================================
# FULL SWEEP
# ============================================================


def run_full_backtest(
    db_path: str = DB_PATH,
    table_name: str = TABLE_NAME,
    parquet_file: str = PARQUET_FILE,
    max_percentile: float = MAX_PERCENTILE,
) -> pd.DataFrame:
    print("Loading factor data...")
    factor_df = load_factor_data(parquet_file=parquet_file)

    print("Loading price data...")
    price_df = load_price_data(db_path=db_path, table_name=table_name)

    print("Preparing daily returns...")
    returns_df = prepare_returns(price_df)

    print("Generating risk overlay exposure...")
    rebalance_weeks = factor_df["rebalance_week"].unique()
    exposure_df = generate_exposure_series(rebalance_weeks)

    results = []
    percentiles = np.arange(0.01, max_percentile + 0.01, 0.01)

    for pct in percentiles:
        print(f"Running bucket {int(pct * 100)}%")
        daily_returns = run_single_bucket(
            factor_df=factor_df,
            returns_df=returns_df,
            exposure_df=exposure_df,
            percentile_cutoff=pct,
            price_df=price_df,
        )
        if daily_returns is None:
            continue

        metrics = compute_metrics(daily_returns)
        metrics["Percentile"] = round(float(pct), 4)
        results.append(metrics)

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Run percentile sweep portfolio backtest with risk overlay.")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--table-name", default=TABLE_NAME)
    parser.add_argument("--factor-file", default=PARQUET_FILE)
    parser.add_argument("--max-percentile", type=float, default=MAX_PERCENTILE)
    parser.add_argument("--output", default=RESULTS_FILE)
    return parser.parse_args()


def main():
    args = parse_args()
    results = run_full_backtest(
        db_path=args.db_path,
        table_name=args.table_name,
        parquet_file=args.factor_file,
        max_percentile=args.max_percentile,
    )

    if results.empty:
        print("No valid results generated.")
        return

    print("\n================ BACKTEST RESULTS WITH REGIME FILTER ================")
    print(results.sort_values("Sharpe", ascending=False))

    output_path = Path(args.output)
    results.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
