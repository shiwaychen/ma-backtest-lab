# -*- coding: utf-8 -*-
"""
双均线策略回测实验室（Streamlit 交互版）
========================================
在网页上拖动参数滑杆，实时重跑双均线策略回测，与"买入持有"对照。

内置策略（同一回测口径，可切换对比）：
  1. 双均线：短期均线从下往上穿过长期均线（金叉）→ 全仓买入；死叉 → 全部卖出
  2. MACD：DIF（快慢 EMA 之差）在其信号线 DEA 上方 → 持有；下方 → 空仓
  3. 唐奇安通道：收盘突破前 20 日最高价 → 买入；跌破前 10 日最低价 → 卖出
  4. 只做现货多头：不加杠杆、不做空

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


def ma_hold(df, fast, slow):
    """双均线信号：快线在慢线上方 → 持有。返回逐日的持仓意愿（收盘确认）。"""
    return df["close"].rolling(fast).mean() > df["close"].rolling(slow).mean()


def macd_hold(df, fast, slow, signal):
    """MACD 信号：DIF = 快EMA − 慢EMA；DEA = DIF 的信号EMA。DIF 在 DEA 上方 → 持有。"""
    dif = (df["close"].ewm(span=fast, adjust=False).mean()
           - df["close"].ewm(span=slow, adjust=False).mean())
    dea = dif.ewm(span=signal, adjust=False).mean()
    return dif > dea


def donchian_hold(df, entry, exit_):
    """唐奇安通道（海龟规则）：收盘突破前 entry 日最高价 → 买入持有；
    跌破前 exit_ 日最低价 → 离场。带状态的信号，逐日推进。"""
    hi = df["close"].shift(1).rolling(entry).max()
    lo = df["close"].shift(1).rolling(exit_).min()
    state = False
    out = []
    for cl, h, l in zip(df["close"], hi, lo):
        if pd.notna(h) and pd.notna(l):
            if not state and cl > h:
                state = True
            elif state and cl < l:
                state = False
        out.append(state)
    return pd.Series(out, index=df.index)


def run_backtest(df, hold):
    """逐日模拟：信号（持仓意愿）在今天收盘确认，明天开盘成交。口径与命令行版一致。"""
    df = df.copy()
    df["action"] = hold.shift(1).fillna(False)

    cash, coin = INIT_CASH, 0.0
    equity_curve = []
    trades = []
    buy_price, buy_date = None, None

    for _, row in df.iterrows():
        price = row["open"]  # 今天开盘价成交
        want_hold = bool(row["action"])
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
    "在左侧切换标的与策略、拖动参数，实时重跑回测。所有策略同一口径："
    "信号次日开盘成交（防未来函数），手续费 0.1%，对照组为买入持有。"
)

# 下拉框显示名 → 币安现货真实代码。
# 代币股（bStocks）在币安的代码是「代码 + B + USDT」，如 NVDA → NVDABUSDT，
# 且 2026 年才上线，历史最多约 144 天，样本小、结论仅供练手。
SYMBOLS = {
    # 加密货币（历史长，3 年日线齐全）
    "BTC 比特币": "BTCUSDT",
    "ETH 以太坊": "ETHUSDT",
    "SOL 索拉纳": "SOLUSDT",
    "BNB 币安币": "BNBUSDT",
    "XRP 瑞波": "XRPUSDT",
    "DOGE 狗狗币": "DOGEUSDT",
    "ADA 艾达币": "ADAUSDT",
    "LINK 预言机": "LINKUSDT",
    "AVAX 雪崩": "AVAXUSDT",
    "LTC 莱特币": "LTCUSDT",
    # 代币股 bStocks（历史约 96–144 天，样本小）
    "NVDA 英伟达·代币股": "NVDABUSDT",
    "TSLA 特斯拉·代币股": "TSLABUSDT",
    "SNDK 闪迪·代币股": "SNDKBUSDT",
    "MU 美光·代币股": "MUBUSDT",
    "SPCX SpaceX·代币股": "SPCXBUSDT",
    "AMD·代币股": "AMDBUSDT",
    "INTC 英特尔·代币股": "INTCBUSDT",
    "QQQ 纳指ETF·代币股": "QQQBUSDT",
    "META·代币股": "METABUSDT",
    "MSFT 微软·代币股": "MSFTBUSDT",
    "GOOGL 谷歌·代币股": "GOOGLBUSDT",
    "SPY 标普ETF·代币股": "SPYBUSDT",
    "SKHY SK海力士·代币股": "SKHYBUSDT",
    "AMZN 亚马逊·代币股": "AMZNBUSDT",
    "AAPL 苹果·代币股": "AAPLBUSDT",
}

with st.sidebar:
    st.header("参数")
    symbol_label = st.selectbox("交易对 / 代币股", list(SYMBOLS.keys()))
    symbol = SYMBOLS[symbol_label]
    days = st.select_slider("回测天数", options=[365, 730, 1095], value=1095)
    strategy = st.selectbox("策略", ["双均线（金叉 / 死叉）", "MACD", "唐奇安通道（突破）"])
    sig = entry = exit_ = None
    if strategy.startswith("双均线"):
        fast = st.slider("快线周期 FAST", 5, 60, 20)
        slow = st.slider("慢线周期 SLOW", 20, 150, 60)
        if fast >= slow:
            st.warning("快线必须小于慢线，已自动把慢线调整为快线 + 10。")
            slow = fast + 10
        strat_name = f"双均线 {fast}/{slow}"
        plot_name = f"MA {fast}/{slow}"
    elif strategy.startswith("MACD"):
        fast = st.slider("MACD 快线 EMA", 5, 30, 12)
        slow = st.slider("MACD 慢线 EMA", 20, 60, 26)
        sig = st.slider("信号线 DEA", 3, 20, 9)
        if fast >= slow:
            st.warning("快线必须小于慢线，已自动把慢线调整为快线 + 10。")
            slow = fast + 10
        strat_name = f"MACD {fast}/{slow}/{sig}"
        plot_name = strat_name
    else:
        entry = st.slider("入场周期（突破前 N 日最高）", 10, 40, 20)
        exit_ = st.slider("出场周期（跌破前 N 日最低）", 5, 30, 10)
        if exit_ >= entry:
            st.warning("出场周期必须短于入场周期（海龟原版如此），已自动把出场调整为入场的一半。")
            exit_ = max(5, entry // 2)
        slow = entry  # 数据长度守卫按最长窗口（入场周期）判断
        strat_name = f"唐奇安 {entry}/{exit_}"
        plot_name = f"Donchian {entry}/{exit_}"

with st.spinner("正在拉取行情并回测..."):
    data = fetch_klines(symbol, days)
    if len(data) <= slow:
        st.warning(
            f"{symbol_label} 目前只有 {len(data)} 天历史，比慢线周期 {slow} 还短，"
            f"算不出信号。把慢线调到 {max(20, len(data) // 2)} 以下再试"
            "（代币股历史短，建议参数用更短的周期）。")
        st.stop()
    if strategy.startswith("双均线"):
        hold = ma_hold(data, fast, slow)
    elif strategy.startswith("MACD"):
        hold = macd_hold(data, fast, slow, sig)
    else:
        hold = donchian_hold(data, entry, exit_)
    result, trades = run_backtest(data, hold)

strat_final = result["equity"].iloc[-1]
bh_final = result["buy_hold"].iloc[-1]
strat_ret = strat_final / INIT_CASH - 1
bh_ret = bh_final / INIT_CASH - 1
strat_dd = max_drawdown(result["equity"])
bh_dd = max_drawdown(result["buy_hold"])
wins = [t for t in trades if t[2] > 0]
win_rate = len(wins) / len(trades) if trades else 0.0

st.caption(f"回测区间：{result['date'].iloc[0]} → {result['date'].iloc[-1]}"
           f"（{len(result)} 根日线）｜策略：{strat_name}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("策略收益", f"{strat_ret:+.1%}", delta=f"买入持有 {bh_ret:+.1%}")
c2.metric("策略最大回撤", f"{strat_dd:.1%}", delta=f"对照 {bh_dd:.1%}", delta_color="inverse")
c3.metric("完整交易次数", f"{len(trades)} 笔")
c4.metric("胜率", f"{win_rate:.0%}")

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(result["date"], result["equity"], label=f"{plot_name} strategy")
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
st.info(
    "MACD（12/26/9）同窗口实测（BTCUSDT，2023-11 → 2026-11）：收益 +69.4%、"
    "最大回撤 −34.6%、43 笔交易、胜率 40%——比双均线更灵敏，本窗口收益更高。\n\n"
    "但这仍是单市场、单窗口的样本内结果，只能说明“值得继续对比”，"
    "不能下结论说 MACD 强于双均线。换标的、换时段，排名可能反转。"
)
st.info(
    "唐奇安通道（20/10，海龟规则）同窗口实测：收益 +102.8%、最大回撤 −27.1%、"
    "22 笔交易、胜率 36%——本窗口最接近买入持有的收益，回撤却只有约一半。\n\n"
    "注意：胜率依然只有 36%，靠的还是少数大趋势单。它在本窗口占优很可能"
    "因为这段行情趋势性强；横盘市里假突破会连续小亏（与双均线同病）。"
)
st.caption("数据来源：币安公开接口。回测不等于实盘：回测没有滑点和情绪，"
           "它的用途是淘汰明显不行的规则，而不是证明能赚钱。本项目仅供学习研究。")
