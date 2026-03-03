import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# CONFIG
# ============================================================

NIFTY_TICKER = "^NSEI"
START_DATE = "2010-01-01"

# DMAs above count -> exposure
EXPOSURE_MAP = {
    3: 1.00,
    2: 0.90,
    1: 0.70,
    0: 0.45,
}


# ============================================================
# DOWNLOAD NIFTY DATA
# ============================================================


def load_nifty_data(start_date: str = START_DATE, ticker: str = NIFTY_TICKER) -> pd.DataFrame:
    nifty = yf.download(
        ticker,
        start=start_date,
        progress=False,
        auto_adjust=False,
    )

    if nifty.empty:
        raise ValueError("No Nifty data downloaded from yfinance.")

    nifty = nifty.reset_index()

    # yfinance may return MultiIndex columns in some versions
    if isinstance(nifty.columns, pd.MultiIndex):
        nifty.columns = [col[0] if isinstance(col, tuple) else col for col in nifty.columns]

    date_col = "Date" if "Date" in nifty.columns else "date"
    close_col = "Close" if "Close" in nifty.columns else "close"

    nifty = nifty[[date_col, close_col]].copy()
    nifty.columns = ["date", "close"]
    nifty["date"] = pd.to_datetime(nifty["date"])

    return nifty


# ============================================================
# COMPUTE DMAs
# ============================================================


def compute_dmas(nifty_df: pd.DataFrame) -> pd.DataFrame:
    df = nifty_df.copy()
    df["dma_50"] = df["close"].rolling(50).mean()
    df["dma_100"] = df["close"].rolling(100).mean()
    df["dma_200"] = df["close"].rolling(200).mean()
    return df


# ============================================================
# WEEKLY EXPOSURE SERIES
# ============================================================


def compute_weekly_exposure(nifty_df: pd.DataFrame, rebalance_weeks) -> pd.DataFrame:
    df = nifty_df.copy()
    df["rebalance_week"] = df["date"].dt.to_period("W-FRI").dt.end_time

    weekly = (
        df.sort_values("date")
        .groupby("rebalance_week", as_index=False)
        .last()
    )

    dma_count = (
        (weekly["close"] > weekly["dma_50"]).astype(int)
        + (weekly["close"] > weekly["dma_100"]).astype(int)
        + (weekly["close"] > weekly["dma_200"]).astype(int)
    )
    weekly["exposure"] = dma_count.map(EXPOSURE_MAP).fillna(0.25)

    rb_index = pd.to_datetime(pd.Series(rebalance_weeks)).dt.tz_localize(None)
    exposure_df = weekly[["rebalance_week", "exposure"]].copy()
    exposure_df["rebalance_week"] = pd.to_datetime(exposure_df["rebalance_week"]).dt.tz_localize(None)
    exposure_df = exposure_df[exposure_df["rebalance_week"].isin(rb_index)]

    return exposure_df.sort_values("rebalance_week").reset_index(drop=True)


# ============================================================
# MAIN FUNCTION
# ============================================================


def generate_exposure_series(rebalance_weeks) -> pd.DataFrame:
    print("Downloading Nifty data...")
    nifty = load_nifty_data()

    print("Computing DMAs...")
    nifty = compute_dmas(nifty)

    print("Computing weekly exposure levels...")
    exposure_df = compute_weekly_exposure(nifty, rebalance_weeks)

    print("Risk overlay ready.")
    return exposure_df
