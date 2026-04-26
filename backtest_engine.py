import numpy as np
import pandas as pd


# ============================================================
# 1. Merge factor + price
# ============================================================

def build_factor_panel(factor_df, price_df):

    price_df = price_df.copy()

    # ========================================================
    # ⭐ NEW：自动处理 MultiIndex price_df
    # ========================================================
    if isinstance(price_df.columns, pd.MultiIndex):

        # 判断是否是 ('close', ticker) 结构
        if "close" in price_df.columns.get_level_values(0):

            # 取 close 层
            close_df = price_df["close"].copy()

            # wide → long
            price_df = (
                close_df.stack()
                .reset_index()
            )

            price_df.columns = ["trade_date", "ticker", "close"]

        else:
            raise ValueError("MultiIndex price_df must contain 'close' level")

    # ========================================================
    # 标准处理（你原来的逻辑）
    # ========================================================
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

    price_df = price_df.sort_values(["ticker", "trade_date"])

    price_df["fwd_return"] = (
        price_df.groupby("ticker")["close"]
        .pct_change()
        .shift(-1)
    )

    trading_days = price_df[["trade_date", "ticker"]].drop_duplicates()

    df = trading_days.merge(
        factor_df,
        on=["trade_date", "ticker"],
        how="left"
    )

    # forward fill factor
    df = (
        df.sort_values(["ticker", "trade_date"])
        .groupby("ticker", group_keys=False)
        .apply(lambda g: g.ffill())
    )

    df = df.merge(
        price_df[["trade_date", "ticker", "fwd_return"]],
        on=["trade_date", "ticker"],
        how="inner"
    )

    df = df.dropna()

    return df


# ============================================================
# 2. IC
# ============================================================

def calc_ic_series(data, factor_col, ret_col="fwd_return"):
    tmp = data[["trade_date", factor_col, ret_col]].dropna()

    ic = tmp.groupby("trade_date").apply(
        lambda x: x[factor_col].corr(x[ret_col], method="spearman")
    )

    return ic


# ============================================================
# 3. Quantile backtest
# ============================================================

def build_quantile_returns(data, factor_col, n_quantiles=3):

    tmp = data.copy()

    tmp["rank"] = tmp.groupby("trade_date")[factor_col].rank(pct=True)

    tmp["quantile"] = (
        np.ceil(tmp["rank"] * n_quantiles)
        .clip(1, n_quantiles)
        .astype(int)
    )

    qret = tmp.groupby(["trade_date", "quantile"])["fwd_return"].mean().unstack()

    long_short = qret[n_quantiles] - qret[1]

    return qret, long_short


# ============================================================
# 4. Metrics
# ============================================================

def sharpe_ratio(ret):
    return ret.mean() / ret.std() * np.sqrt(252)


def summarize_ic(ic):
    return {
        "mean": ic.mean(),
        "std": ic.std(),
        "ir": ic.mean() / ic.std() * np.sqrt(252)
    }


def summarize_ls(ret):
    return {
        "mean": ret.mean(),
        "sharpe": sharpe_ratio(ret)
    }


# ============================================================
# 5. Main entry（最重要！）
# ============================================================

def run_backtest(factor_df, price_df, factor_col):

    panel = build_factor_panel(factor_df, price_df)

    ic = calc_ic_series(panel, factor_col)
    qret, ls = build_quantile_returns(panel, factor_col)

    results = {
        "ic": ic,
        "ic_summary": summarize_ic(ic),
        "long_short": ls,
        "ls_summary": summarize_ls(ls),
        "panel": panel
    }

    return results