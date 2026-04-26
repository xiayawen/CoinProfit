import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("price_cache")
DATA_DIR.mkdir(exist_ok=True)


# =========================
# 1. 单 ticker cache
# =========================
def _load_or_update_single_ticker(ticker, start="2000-01-01"):
    path = DATA_DIR / f"{ticker}.csv"

    # ---------- no cache ----------
    if not path.exists():
        df = yf.download(ticker, start=start, progress=False)
        df = _clean_columns(df)
        df.to_csv(path)

    # ---------- load ----------
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # ---------- update ----------
    if not df.empty:
        last_date = df.index.max()
        start_update = last_date - pd.Timedelta(days=5)

        new_df = yf.download(ticker, start=start_update, progress=False)
        new_df = _clean_columns(new_df)

        if not new_df.empty:
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            df.to_csv(path)

    return df


def _clean_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# =========================
# 2. 转 long format
# =========================
def _to_long_format(df, ticker):
    df = df.copy()

    df["date"] = df.index
    df["ticker"] = ticker

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    })

    # 🔥 关键：如果没有 adj_close，就用 close 代替
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]

    cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

    return df[cols]


# =========================
# 3. 多 ticker 主函数
# =========================
def get_clean_price_df(tickers):
    all_df = []

    for ticker in tickers:
        print(f"Processing {ticker}...")

        df = _load_or_update_single_ticker(ticker)
        df_long = _to_long_format(df, ticker)

        all_df.append(df_long)

    df_all = pd.concat(all_df, ignore_index=True)

    # 🔥 非常关键：排序
    df_all = df_all.sort_values(["ticker", "date"]).reset_index(drop=True)

    return df_all