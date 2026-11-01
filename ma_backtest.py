# -*- coding: utf-8 -*-
"""
双均线策略回测脚本（教学版）
============================
策略规则（写成代码，就这么三条）：
  1. 短期均线从下往上穿过长期均线（金叉）→ 全仓买入
  2. 短期均线从上往下跌破长期均线（死叉）→ 全部卖出
  3. 只做现货多头：不加杠杆、不做空、不扛单，信号反转就执行

回测口径：
  - 信号在当天收盘确认，按【第二天开盘价】成交（避免偷看未来）
  - 每笔成交收 0.1% 手续费（币安现货标准费率）
  - 数据来自币安公开接口，不需要任何 API Key

怎么改着玩（改文件顶部“配置区”的数字就行）：
  - FAST / SLOW：均线周期，比如改成 10 / 30，交易会更频繁
  - SYMBOL：换交易对，比如 'ETHUSDT'
  - DAYS：回测多少天的数据

注意：回测赚钱 != 实盘能赚钱。回测没有滑点、没有情绪，
它的用途是先淘汰明显不行的规则，而不是证明能赚钱。
"""

import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= 配置区（先从这里开始改） =================
SYMBOL = "BTCUSDT"     # 交易对
INTERVAL = "1d"        # K线周期：1d = 日线
DAYS = 1095            # 回测天数：约 3 年
FAST = 20              # 短期均线周期
SLOW = 60              # 长期均线周期
FEE = 0.001            # 单边手续费 0.1%
INIT_CASH = 1000.0     # 初始资金（USDT）
# ==========================================================


def fetch_klines(symbol, interval, days):
    """从币安公开接口拉 K 线（自动翻页），返回 DataFrame。"""
    url = "https://data-api.binance.vision/api/v3/klines"
    rows, remaining = [], days
    end_time = None
    while remaining > 0:
        limit = min(1000, remaining)
        params = {"symbol": symbol, "interval": interval, "limit": limit}
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


def run_backtest(df):
    """逐日模拟：今天收盘出信号，明天开盘成交。"""
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(FAST).mean()
    df["ma_slow"] = df["close"].rolling(SLOW).mean()
    # 信号：1 = 该持有，0 = 该空仓（金叉/死叉体现在信号翻转上）
    df["signal"] = (df["ma_fast"] > df["ma_slow"]).astype(int)
    # 昨天收盘的信号，决定今天开盘的动作
    df["action"] = df["signal"].shift(1).fillna(0)

    cash, coin = INIT_CASH, 0.0
    equity_curve = []
    trades = []          # 每笔完整交易：(买入日, 卖出日, 收益率)
    buy_price, buy_date = None, None

    for _, row in df.iterrows():
        price = row["open"]  # 今天开盘价成交
        want_hold = row["action"] == 1
        if want_hold and coin == 0:            # 该买，还没买 → 买
            coin = cash * (1 - FEE) / price
            cash = 0.0
            buy_price, buy_date = price, row["date"]
        elif not want_hold and coin > 0:       # 该卖，还持有 → 卖
            cash = coin * price * (1 - FEE)
            coin = 0.0
            # 本笔收益率 = 卖出到手价 / 买入实际成本价 - 1（都含手续费）
            ret = (price * (1 - FEE)) / (buy_price / (1 - FEE)) - 1
            trades.append((buy_date, row["date"], ret))
        equity_curve.append(cash + coin * row["close"])

    df["equity"] = equity_curve
    # 对照组：买入持有（第一天全仓买入后不动）
    first_open = df["open"].iloc[0]
    bh_coin = INIT_CASH * (1 - FEE) / first_open
    df["buy_hold"] = bh_coin * df["close"]
    return df, trades


def report(df, trades):
    """把结果打印成人能看懂的账。"""
    start, end = df["date"].iloc[0], df["date"].iloc[-1]
    strat_final = df["equity"].iloc[-1]
    bh_final = df["buy_hold"].iloc[-1]
    print(f"回测区间：{start} -> {end}（{len(df)} 根日线）")
    print(f"策略：{FAST}/{SLOW} 双均线，初始资金 {INIT_CASH:.0f} USDT")
    print("-" * 52)
    print(f"策略最终资金：{strat_final:,.2f} USDT（{strat_final / INIT_CASH - 1:+.1%}）")
    print(f"买入持有对照：{bh_final:,.2f} USDT（{bh_final / INIT_CASH - 1:+.1%}）")
    print(f"策略最大回撤：{max_drawdown(df['equity']):.1%}")
    print(f"对照最大回撤：{max_drawdown(df['buy_hold']):.1%}")
    if trades:
        wins = [t for t in trades if t[2] > 0]
        print(f"完整交易次数：{len(trades)} 笔，盈利 {len(wins)} 笔，胜率 {len(wins) / len(trades):.0%}")
        print("-" * 52)
        print("最近 5 笔交易（买入日 -> 卖出日：收益率）：")
        for b, s, r in trades[-5:]:
            print(f"  {b} -> {s}：{r:+.1%}")
    else:
        print("无完整交易")


def save_chart(df):
    plt.figure(figsize=(11, 5))
    plt.plot(df["date"], df["equity"], label=f"MA {FAST}/{SLOW} strategy")
    plt.plot(df["date"], df["buy_hold"], label="Buy & hold", alpha=0.7)
    plt.title(f"{SYMBOL} backtest: strategy vs buy & hold")
    plt.ylabel("USDT")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("equity_curve.png", dpi=130)
    print("收益曲线图已保存：equity_curve.png")


if __name__ == "__main__":
    print("正在拉取行情数据（公开接口，无需 Key）...")
    data = fetch_klines(SYMBOL, INTERVAL, DAYS)
    result, trades = run_backtest(data)
    report(result, trades)
    save_chart(result)
