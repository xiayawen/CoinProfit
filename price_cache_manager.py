# ============================================================
# 0. Imports & Config
# ============================================================

import pandas as pd
import yfinance as yf
import time
import random
from pathlib import Path

DATA_DIR = Path("price_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Download OHLCV
# ============================================================

def download_ohlcv(ticker, start=None, end=None):
    """
    Download OHLCV data from Yahoo Finance
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        print(f"No data for {ticker}")
        return pd.DataFrame()

    df = df.reset_index()

    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"])

    return df


# ============================================================
# 2. Cache paths
# ============================================================

def get_price_cache_path(ticker, data_dir=DATA_DIR):
    ticker = ticker.upper().strip()
    return data_dir / f"{ticker}_ohlcv.csv"


# ============================================================
# 3. Read cache
# ============================================================

def read_cached_price(ticker, data_dir=DATA_DIR):
    path = get_price_cache_path(ticker, data_dir)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


# ============================================================
# 4. Save cache
# ============================================================

def save_cached_price(ticker, df, data_dir=DATA_DIR):
    path = get_price_cache_path(ticker, data_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"Saved cache for {ticker} → {path}")


# ============================================================
# 5. Core: load or update
# ============================================================

def load_or_update_price(ticker, data_dir=DATA_DIR):
    """
    Main cache logic:

    If no cache:
        download full history

    If cache exists:
        only fetch new data
    """
    ticker = ticker.upper().strip()
    path = get_price_cache_path(ticker, data_dir)

    # =========================
    # Case 1: No cache
    # =========================
    if not path.exists():
        print(f"{ticker}: no cache, downloading full history...")

        df = download_ohlcv(ticker)

        if not df.empty:
            df = df.sort_values("date")
            save_cached_price(ticker, df, data_dir)

        return df

    # =========================
    # Case 2: Cache exists
    # =========================
    print(f"{ticker}: cache found, checking updates...")

    df_cached = read_cached_price(ticker, data_dir)

    if df_cached.empty or "date" not in df_cached.columns:
        print(f"{ticker}: cache broken, re-downloading...")

        df = download_ohlcv(ticker)

        if not df.empty:
            df = df.sort_values("date")
            save_cached_price(ticker, df, data_dir)

        return df

    last_date = df_cached["date"].max()

    print(f"{ticker}: last cached date = {last_date}")

    # 防止重复，+1天
    start_new = last_date + pd.Timedelta(days=1)

    df_new = download_ohlcv(ticker, start=start_new)

    if df_new.empty:
        print(f"{ticker}: already up to date.")
        return df_cached

    print(f"{ticker}: found {len(df_new)} new rows")

    # =========================
    # Merge + deduplicate
    # =========================
    df_all = pd.concat([df_cached, df_new], ignore_index=True)

    df_all = df_all.drop_duplicates(subset=["date"])
    df_all = df_all.sort_values("date").reset_index(drop=True)

    save_cached_price(ticker, df_all, data_dir)

    return df_all


# ============================================================
# 6. Multi-ticker
# ============================================================

def load_or_update_prices(tickers, data_dir=DATA_DIR):
    all_data = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")

        try:
            df = load_or_update_price(ticker, data_dir)

            if not df.empty:
                all_data.append(df)

            # 模拟你SEC那种防限速节奏（好习惯）
            sleep_time = 1 + random.uniform(0, 1)
            time.sleep(sleep_time)

        except Exception as e:
            print(f"Failed for {ticker}: {e}")

    if not all_data:
        return pd.DataFrame()

    df_all = pd.concat(all_data, ignore_index=True)

    print("\nFinal shape:", df_all.shape)

    return df_all


# ============================================================
# 7. Example run
# ============================================================

if __name__ == "__main__":
    tickers = ["AAPL", "NVDA", "MSFT", "AMD"]

    df_all = load_or_update_prices(tickers)

    print(df_all.head())