import numpy as np
import pandas as pd


# ============================================================
# utils
# ============================================================

def sharpe_ratio(ret):
    if ret.std() == 0:
        return np.nan
    return ret.mean() / ret.std() * np.sqrt(252)


def max_drawdown(ret):
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return dd.min()


# ============================================================
# panel
# ============================================================

def build_factor_panel(factor_df, price_df):

    factor_df = factor_df.copy()
    price_df = price_df.copy()

    # 去重列（防止你之前那个 bug）
    price_df = price_df.loc[:, ~price_df.columns.duplicated()]

    # rename
    factor_df = factor_df.rename(columns={"transaction_date": "trade_date"})

    factor_df["trade_date"] = pd.to_datetime(factor_df["trade_date"])
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])

    price_df = price_df.sort_values(["ticker", "trade_date"])

    price_df["fwd_return"] = (
        price_df.groupby("ticker")["close"]
        .pct_change()
        .shift(-1)
    )

    df = price_df.merge(
        factor_df,
        on=["trade_date", "ticker"],
        how="left"
    )

    df = (
        df.sort_values(["ticker", "trade_date"])
        .groupby("ticker", group_keys=False)
        .apply(lambda g: g.ffill())
    )

    df = df.dropna(subset=["fwd_return"])

    return df


# ============================================================
# IC
# ============================================================

def calc_ic_series(data, factor_col):
    tmp = data[["trade_date", factor_col, "fwd_return"]].dropna()

    ic = tmp.groupby("trade_date").apply(
        lambda x: x[factor_col].corr(x["fwd_return"], method="spearman")
    )

    return ic


def summarize_ic(ic):
    return {
        "IC Mean": ic.mean(),
        "IC Std": ic.std(),
        "ICIR": ic.mean() / ic.std() * np.sqrt(252)
    }


# ============================================================
# quantile
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
# 单因子评估
# ============================================================

def evaluate_factor(panel, factor_col):

    ic = calc_ic_series(panel, factor_col)
    ic_summary = summarize_ic(ic)

    qret, ls = build_quantile_returns(panel, factor_col)

    ls_mean = ls.mean()
    ls_sharpe = sharpe_ratio(ls)
    ls_mdd = max_drawdown(ls)

    return {
        "factor": factor_col,
        "IC Mean": ic_summary["IC Mean"],
        "ICIR": ic_summary["ICIR"],
        "LS Mean": ls_mean,
        "Sharpe": ls_sharpe,
        "MaxDD": ls_mdd
    }


# ============================================================
# 多因子分析
# ============================================================

def analyze_factors(factor_df, price_df):

    print("\n" + "="*70)
    print("🧠 Running Factor Analysis (Auto Ranking)")
    print("="*70)

    panel = build_factor_panel(factor_df, price_df)

    # 自动识别因子列
    factor_cols = [
        col for col in factor_df.columns
        if col not in ["ticker", "transaction_date", "trade_date"]
    ]

    results = []

    for col in factor_cols:
        print(f"\n🔍 Evaluating: {col}")
        try:
            res = evaluate_factor(panel, col)
            results.append(res)
        except Exception as e:
            print(f"❌ Failed: {col} | {e}")

    results_df = pd.DataFrame(results)

    # ========================================================
    # 排序逻辑
    # ========================================================
    results_df = results_df.sort_values(
        by=["ICIR", "Sharpe"],
        ascending=False
    ).reset_index(drop=True)

    # ========================================================
    # PRINT
    # ========================================================
    print("\n📊 Factor Ranking")
    print("-"*70)
    print(results_df.round(4))

    # ========================================================
    # best factor
    # ========================================================
    if not results_df.empty:
        best = results_df.iloc[0]

        print("\n🏆 BEST FACTOR")
        print("-"*40)
        for k, v in best.items():
            if isinstance(v, float):
                print(f"{k:<12}: {v:.6f}")
            else:
                print(f"{k:<12}: {v}")

    print("\n✅ Analysis Done\n")

    return results_df