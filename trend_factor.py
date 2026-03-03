import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = "market_data_yfinance.db"
TABLE_NAME = "ohlcv"

LOOKBACKS = [15, 30, 45, 60, 90]
SLOPE_WEIGHTS = {
    15: 0.25,
    30: 0.25,
    45: 0.20,
    60: 0.15,
    90: 0.15,
}

DAYS_IN_YEAR = 252
LAMBDA_ACCEL = 0.5
MIN_OBS = max(LOOKBACKS)

OUTPUT_FILE = "data/weekly_trend_factor.parquet"


# ============================================================
# DATA LOADING
# ============================================================


def _get_price_column(conn: sqlite3.Connection, table_name: str) -> str:
    cols = pd.read_sql(f"PRAGMA table_info({table_name})", conn)["name"].tolist()
    if "adj_close" in cols:
        return "adj_close"
    if "close" in cols:
        return "close"
    raise ValueError(f"Neither 'adj_close' nor 'close' found in table {table_name}")


def load_data(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    price_col = _get_price_column(conn, table_name)
    df = pd.read_sql(
        f"""
        SELECT date, ticker, {price_col} AS close
        FROM {table_name}
        ORDER BY ticker, date
        """,
        conn,
        parse_dates=["date"],
    )
    conn.close()

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


# ============================================================
# FAST ROLLING REGRESSION
# ============================================================


def rolling_regression_log_price(log_price: pd.Series, window: int):
    """Rolling annualized slope + R² for log-prices using rolling apply."""
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    cov_xy = log_price.rolling(window).apply(
        lambda y: np.dot(x - x_mean, y - y.mean()), raw=True
    )

    slope = cov_xy / x_var
    annualized_slope = (np.exp(slope) ** DAYS_IN_YEAR) - 1

    y_var = log_price.rolling(window).var() * (window - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = (cov_xy**2) / (x_var * y_var)

    return annualized_slope, r_squared.clip(lower=0.0, upper=1.0)


# ============================================================
# PER-TICKER FACTOR
# ============================================================


def compute_daily_factor(group: pd.DataFrame, lambda_accel: float) -> pd.DataFrame:
    group = group.sort_values("date").copy()

    prices = group["close"].where(group["close"] > 0)
    log_price = np.log(prices)

    if len(group) < MIN_OBS:
        group["weighted_slope"] = np.nan
        group["r2_avg"] = np.nan
        group["acceleration"] = np.nan
        group["base_score"] = np.nan
        group["trend_factor"] = np.nan
        return group

    for lb in LOOKBACKS:
        slope, r2 = rolling_regression_log_price(log_price, lb)
        group[f"slope_{lb}"] = slope
        group[f"r2_{lb}"] = r2

    group["weighted_slope"] = sum(
        SLOPE_WEIGHTS[lb] * group[f"slope_{lb}"] for lb in LOOKBACKS
    )

    group["r2_avg"] = group[[f"r2_{lb}" for lb in LOOKBACKS]].mean(axis=1)

    accel_1 = group["r2_15"] - group["r2_30"]
    accel_2 = group["r2_30"] - group["r2_45"]
    group["acceleration"] = (accel_1 + accel_2) / 2

    group["base_score"] = group["weighted_slope"] * group["r2_avg"]
    group["trend_factor"] = group["base_score"] * (1 + lambda_accel * group["acceleration"])

    return group


# ============================================================
# WEEKLY RANKING
# ============================================================


def weekly_rebalance_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["trend_factor"]).copy()
    df["rebalance_week"] = df["date"].dt.to_period("W-FRI").dt.end_time

    weekly = (
        df.sort_values("date")
        .groupby(["ticker", "rebalance_week"], as_index=False)
        .last()
    )

    weekly["percentile_rank"] = (
        weekly.groupby("rebalance_week")["trend_factor"].rank(pct=True)
    )
    weekly["rank"] = (
        weekly.groupby("rebalance_week")["trend_factor"].rank(ascending=False, method="first")
    )

    # Backward-compatible aliases for dashboard/backtest modules
    weekly["adj_close"] = weekly["close"]
    weekly["factor"] = weekly["trend_factor"]
    weekly["percentile"] = weekly["percentile_rank"]
    weekly["annualized_slope"] = weekly["weighted_slope"]
    weekly["r2"] = weekly["r2_avg"]

    return weekly.reset_index(drop=True)


# ============================================================
# MAIN ENGINE
# ============================================================


def run_factor_engine(db_path: str = DB_PATH, table_name: str = TABLE_NAME, lambda_accel: float = LAMBDA_ACCEL) -> pd.DataFrame:
    print("Loading data...")
    df = load_data(db_path=db_path, table_name=table_name)

    print("Computing daily factor metrics...")
    df = (
        df.groupby("ticker", group_keys=False)
        .apply(lambda g: compute_daily_factor(g, lambda_accel))
        .reset_index(drop=True)
    )

    print("Applying weekly rebalance and ranking...")
    weekly_ranked = weekly_rebalance_and_rank(df)
    return weekly_ranked


def parse_args():
    parser = argparse.ArgumentParser(description="Build weekly multi-horizon trend factor dataset.")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--table-name", default=TABLE_NAME)
    parser.add_argument("--lambda-accel", type=float, default=LAMBDA_ACCEL)
    parser.add_argument("--output", default=OUTPUT_FILE)
    return parser.parse_args()


def main():
    args = parse_args()
    weekly_ranked = run_factor_engine(
        db_path=args.db_path,
        table_name=args.table_name,
        lambda_accel=args.lambda_accel,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving output to {args.output} ...")
    weekly_ranked.to_parquet(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
