import argparse
import sqlite3
import pandas as pd
import numpy as np
from numpy.linalg import lstsq

# ==============================
# CONFIGURATION
# ==============================

DB_PATH = "market_data_yfinance.db"
TABLE_NAME = "ohlcv"

PRICE_COL = "adj_close"
DATE_COL = "date"
TICKER_COL = "ticker"

LOOKBACK = 90           # regression window (bars)
BARS_IN_YEAR = 252      # annualization
MIN_OBS = 60            # minimum data points for IPO safety
REBALANCE_FREQ = "W-FRI"

OUTPUT_FILE = "data/weekly_trend_factor.parquet"


# ==============================
# DATA LOADING
# ==============================

def load_data(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        f"""
        SELECT {DATE_COL}, {TICKER_COL}, {PRICE_COL}
        FROM {TABLE_NAME}
        ORDER BY {TICKER_COL}, {DATE_COL}
        """,
        conn,
        parse_dates=[DATE_COL]
    )
    conn.close()
    return df


# ==============================
# REGRESSION CORE
# ==============================

def exp_regression_slope_r2(log_prices: np.ndarray):
    """
    Exponential regression slope on log-prices.
    Returns:
        slope_per_bar, r_squared
    """
    y = log_prices
    x = np.arange(len(y))

    X = np.column_stack([x, np.ones(len(x))])
    slope, intercept = lstsq(X, y, rcond=None)[0]

    y_hat = slope * x + intercept

    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)

    r2 = 0.0 if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return slope, r2


# ==============================
# FACTOR COMPUTATION PER STOCK
# ==============================

def compute_factor_for_stock(
    stock_df: pd.DataFrame,
    lookback: int = LOOKBACK,
    min_obs: int = MIN_OBS
) -> pd.DataFrame:
    prices = stock_df[PRICE_COL]

    if len(prices) < min_obs:
        stock_df["annualized_slope"] = np.nan
        stock_df["r2"] = np.nan
        stock_df["factor"] = np.nan
        return stock_df

    # Guard against invalid log inputs.
    valid_prices = prices.where(prices > 0)
    log_prices = np.log(valid_prices)

    annualized_slopes = np.full(len(stock_df), np.nan)
    r2_values = np.full(len(stock_df), np.nan)

    for i in range(lookback, len(stock_df)):
        window = log_prices.iloc[i - lookback: i].values

        if np.isnan(window).any():
            continue

        slope, r2 = exp_regression_slope_r2(window)
        annualized = (np.exp(slope) ** BARS_IN_YEAR) - 1.0

        annualized_slopes[i] = annualized
        r2_values[i] = r2

    stock_df["annualized_slope"] = annualized_slopes
    stock_df["r2"] = r2_values
    stock_df["factor"] = stock_df["annualized_slope"] * stock_df["r2"]

    return stock_df


# ==============================
# UNIVERSE-WIDE FACTOR BUILD
# ==============================

def build_factor_universe(
    df: pd.DataFrame,
    lookback: int = LOOKBACK,
    min_obs: int = MIN_OBS
) -> pd.DataFrame:
    results = []

    for _, stock_df in df.groupby(TICKER_COL):
        stock_df = stock_df.sort_values(DATE_COL).reset_index(drop=True)
        stock_df = compute_factor_for_stock(stock_df, lookback=lookback, min_obs=min_obs)
        results.append(stock_df)

    return pd.concat(results, ignore_index=True)


# ==============================
# WEEKLY REBALANCING & RANKING
# ==============================

def weekly_rebalance_and_rank(df: pd.DataFrame, rebalance_freq: str = REBALANCE_FREQ) -> pd.DataFrame:
    df = df.dropna(subset=["factor"]).copy()

    # Define rebalance week
    df["rebalance_week"] = df[DATE_COL].dt.to_period(rebalance_freq)

    # Keep last observation per stock per week (CRITICAL)
    weekly_snapshot = (
        df.sort_values(DATE_COL)
          .groupby([TICKER_COL, "rebalance_week"], as_index=False)
          .last()
    )

    # Cross-sectional ranking
    ranked = (
        weekly_snapshot
        .groupby("rebalance_week", group_keys=False)
        .apply(
            lambda x: x.assign(
                rank=x["factor"].rank(ascending=False, method="first"),
                percentile=x["factor"].rank(pct=True, ascending=True)
            ),
            include_groups=False
        )
    )

    return ranked.reset_index(drop=True)


# ==============================
# MAIN EXECUTION
# ==============================

def parse_args():
    parser = argparse.ArgumentParser(description="Build weekly trend factor dataset.")
    parser.add_argument("--db-path", default=DB_PATH)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--min-obs", type=int, default=MIN_OBS)
    parser.add_argument("--rebalance-freq", default=REBALANCE_FREQ)
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading data...")
    raw_data = load_data(db_path=args.db_path)

    print("Building factor universe...")
    factor_data = build_factor_universe(raw_data, lookback=args.lookback, min_obs=args.min_obs)

    print("Applying weekly rebalancing and ranking...")
    weekly_ranked = weekly_rebalance_and_rank(factor_data, rebalance_freq=args.rebalance_freq)

    print(f"Saving output to {args.output} ...")
    weekly_ranked.to_parquet(args.output)

    print("Done.")


if __name__ == "__main__":
    main()
