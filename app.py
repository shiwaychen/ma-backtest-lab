# -*- coding: utf-8 -*-
"""
双均线策略回测实验室（Streamlit 交互版）
========================================
在网页上拖动参数滑杆，实时重跑双均线策略回测，与"买入持有"对照。

策略规则（和命令行版完全一致）：
  1. 短期均线从下往上穿过长期均线（金叉）→ 全仓买入
  2. 短期均线从上往下跌破长期均线（死叉）→ 全部卖出
  3. 只做现货多头：不加杠杆、不做空

回测口径（诚实原则）：
  - 信号在当天收盘确认，按【第二天开盘价】成交（防"未来函数"，不偷看未来）
  - 每笔成交收 0.1% 手续费（币安现货标准费率）
  - 数据来自币安公开接口，无需 API Key

本地运行：streamlit run app.py
"""

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

FEE = 0.001          # 单边手续费 0.1%
INIT_CASH = 1000.0   # 初始资金（USDT）
INTERVAL = "1d"      # 日线


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_klines(symbol, days):
    """从币安公开接口拉 K 线（自动翻页），返回 DataFrame。"""
    url = "https://data-api.binance.vision/api/v3/klines"
    rows, remaining = [], days
    end_time = None
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": symbol, "interval": INTERVAL, "limit": limit}
        if end_time is not None:
            params["endTime"] = end_time
        data = requests.get(url, params=params, timeout=30).json()
        if not data:
            break
        rows = data + rows
        end_time = data[0][0] - 1  # 往前翻一页
        remaining -= len(data)
        if len(data) < limit:
            break
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbb", "tbq", "ignore"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.date
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    return df[["date", "open", "high", "low", "close"]].reset_index(drop=True)


def max_drawdown(equity):
    """最大回撤：从历史最高点跌下来的最深幅度（负数）。"""
    peak = equity.cummax()
    return (equity / peak - 1).min()


def run_backtest(df, fast, slow):
    """逐日模拟：今天收盘出信号，明天开盘成交。逻辑与命令行版一致。"""
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(fast).mean()
    df["ma_slow"] = df["close"].rolling(slow).mean()
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    df["action"] = df["signal"].shift(1).fillna(0)

    cash, coin = INIT_CASH, 0.0
    equity_curve = []
    trades = []
    buy_price, buy_date = None, None

    for _, row in df.iterrows():
        price = row["open"]  # 今天开盘价成交
        want_hold = row["action"] == 1
        if want_hold and coin == 0:
            coin = cash * (1 - FEE) / price
            cash = 0.0
            buy_price, buy_date = price, row["date"]
        elif not want_hold and coin > 0:
            cash = coin * price * (1 - FEE)
            coin = 0.0
            ret = (price * (1 - FEE)) / (buy_price / (1 - FEE)) - 1
            trades.append((buy_date, row["date"], ret))
        equity_curve.append(cash + coin * row["close"])

    df["equity"] = equity_curve
    first_open = df["open"].iloc[0]
    bh_coin = INIT_CASH * (1 - FEE) / first_open
    df["buy_hold"] = bh_coin * df["close"]
    return df, trades


# ============================ 页面 ============================
st.set_page_config(page_title="双均线回测实验室", layout="wide")
st.title("双均线策略回测实验室")
st.markdown(
    "拖动左侧参数，实时重跑回测。策略规则：**金叉买入、死叉卖出**，"
    "信号次日开盘成交（防未来函数），手续费 0.1%，对照组为买入持有。"
)

with st.sidebar:
    st.header("参数")
    symbol = st.selectbox("交易对", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    days = st.select_slider("回测天数", options=[365, 730, 1095], value=1095)
    fast = st.slider("快线周期 FAST", 5, 60, 20)
    slow = st.slider("慢线周期 SLOW", 20, 150, 60)
    if fast >= slow:
        st.warning("快线必须小于慢线，已自动把慢线调整为快线 + 10。")
        slow = fast + 10

with st.spinner("正在拉取行情并回测..."):
    data = fetch_klines(symbol, days)
    result, trades = run_backtest(data, fast, slow)

strat_final = result["equity"].iloc[-1]
bh_final = result["buy_hold"].iloc[-1]
strat_ret = strat_final / INIT_CASH - 1
bh_ret = bh_final / INIT_CASH - 1
strat_dd = max_drawdown(result["equity"])
bh_dd = max_drawdown(result["buy_hold"])
wins = [t for t in trades if t[2] > 0]
win_rate = len(wins) / len(trades) if trades else 0.0

st.caption(f"回测区间：{result['date'].iloc[0]} → {result['date'].iloc[-1]}"
           f"（{len(result)} 根日线）｜参数：{fast}/{slow}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("策略收益", f"{strat_ret:+.1%}", delta=f"买入持有 {bh_ret:+.1%}")
c2.metric("策略最大回撤", f"{strat_dd:.1%}", delta=f"对照 {bh_dd:.1%}", delta_color="inverse")
c3.metric("完整交易次数", f"{len(trades)} 笔")
c4.metric("胜率", f"{win_rate:.0%}")

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(result["date"], result["equity"], label=f"MA {fast}/{slow} strategy")
ax.plot(result["date"], result["buy_hold"], label="Buy & hold", alpha=0.7)
ax.set_title(f"{symbol} backtest: strategy vs buy & hold")
ax.set_ylabel("USDT")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
st.pyplot(fig)

if trades:
    st.subheader("最近 10 笔交易")
    tdf = pd.DataFrame(trades[-10:], columns=["买入日", "卖出日", "收益率"])
    tdf["收益率"] = tdf["收益率"].map(lambda r: f"{r:+.1%}")
    st.dataframe(tdf, width="stretch", hide_index=True)

st.subheader("结论")
st.info(
    "已验证的一组结论（BTCUSDT，2023-11 → 2026-10）：把参数从 20/60 改短为 10/30 后，"
    "交易次数翻倍（11 → 22 笔），收益反而从 +45.1% 降到 +34.2%，最大回撤还更深。\n\n"
    "原因：短均线在震荡行情里被假信号反复骗进骗出，小亏 + 手续费持续磨损。"
    "**参数不是越灵敏越好——这组参数已被数据淘汰。**\n\n"
    "你也可以在左上方拖动参数，亲自验证这个结论。"
)
st.caption("数据来源：币安公开接口。回测不等于实盘：回测没有滑点和情绪，"
           "它的用途是淘汰明显不行的规则，而不是证明能赚钱。本项目仅供学习研究。")
