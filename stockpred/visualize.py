"""可视化：预测图 + K 线形态图 + 回测对比图。"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # 无界面环境
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.patches import Rectangle

# 配置中文字体，避免图表中文乱码；按平台依次尝试常见中文字体
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/simhei.ttf",          # Windows 黑体
    "C:/Windows/Fonts/msyh.ttc",            # Windows 微软雅黑
    "/System/Library/Fonts/PingFang.ttc",   # macOS
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux 文泉驿
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
for _FONT_PATH in _FONT_CANDIDATES:
    if os.path.exists(_FONT_PATH):
        try:
            fm.fontManager.addfont(_FONT_PATH)
            plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
            break
        except Exception:  # 字体损坏/格式不支持时继续尝试下一个
            continue
plt.rcParams["axes.unicode_minus"] = False  # 正常显示负号

# A 股配色：涨红跌绿（与国际惯例相反，这里按国内习惯）
UP_COLOR = "#d62728"
DOWN_COLOR = "#2ca02c"


# 各周期预测线颜色
_HORIZON_COLORS = {1: "#9467bd", 5: "#ff7f0e", 7: "#2ca02c", 30: "#d62728"}


def plot_forecast(history: pd.DataFrame, preds: dict, out: str, name: str = "",
                  max_horizon: int = 30, turn: dict | None = None):
    """绘制历史收盘价 + 各周期预测线 + 拐点标注。

    preds: {horizon: DataFrame[date, pred_close, ...]}，每个周期一段虚线。
    turn : 拐点模型输出（含 days.top/bot 与 recent 已确认拐点），可为 None。
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = history.tail(60)
    ax.plot(hist["date"], hist["close"], label="历史真实", color="#1f77b4", lw=1.6)
    last_x = hist["date"].iloc[-1]
    last_y = float(hist["close"].iloc[-1])

    # 最近已确认的历史拐点（标记在真实曲线上）
    if turn and turn.get("recent"):
        hist_dates = {str(pd.to_datetime(d).date()): (pd.to_datetime(d), float(c))
                      for d, c in zip(hist["date"], hist["close"])}
        for r in turn["recent"][:6]:
            if r["date"] in hist_dates:
                x, y = hist_dates[r["date"]]
                ax.scatter([x], [y], marker="v" if r["kind"] == "顶部" else "^",
                           s=60, color="#d62728" if r["kind"] == "顶部" else "#2ca02c",
                           zorder=5, label=("已确认顶" if r["kind"] == "顶部" else "已确认底"))

    handled = set()
    longest = None
    for h in sorted(preds.keys()):
        pred = preds[h]
        if pred is None or pred.empty or h in handled:
            continue
        handled.add(h)
        if longest is None or len(pred) > len(longest):
            longest = pred

        # 每个周期只画 1 个点 = 对「第 h 个交易日收盘价」的**一次性**预测。
        # preds[h] 里虽然有多行，但除最后一行外都是插值（模型并没有逐日预测），
        # 把整条路径连成实线会让人误以为"预测了未来 30 天的每一天"，是错的。
        x = pd.to_datetime(pred["date"].iloc[-1])
        y = float(pred["pred_close"].iloc[-1])
        color = _HORIZON_COLORS.get(h, "#7f7f7f")
        band = float(pred["band_ret"].iloc[-1]) if "band_ret" in pred.columns else 0.0

        # 细点线仅示意"从现在指向该档目标"，不代表中间的逐日路径
        ax.plot([last_x, x], [last_y, y], ":", color=color, lw=1.1, alpha=0.5)
        if band > 0:
            ax.errorbar([x], [y], yerr=[[y * band], [y * band]], fmt="none",
                        ecolor=color, elinewidth=2.4, capsize=6, alpha=0.7, zorder=5)
        ax.scatter([x], [y], s=54, color=color, zorder=6,
                   edgecolor="white", linewidth=0.9, label=f"{h}日预测 {y:.2f}")
        # 不在点上重复写价格：1日/5日 的点和价几乎重合，标注必然叠在一起。
        # 具体数值放在图例里（图例已经带价格），报告的表格里也有精确值。

    # 预计拐点日：竖线标点估计，色带标"区间"（模型给的是天数区间，只画一根线会显得比实际更确定）
    if turn and turn.get("days") and longest is not None:
        dates = list(pd.to_datetime(longest["date"]))
        for k, (kind, color, label) in enumerate(
                (("top", "#d62728", "预计顶"), ("bot", "#2ca02c", "预计底"))):
            d = (turn["days"] or {}).get(kind)
            if not d:
                continue
            lo_i = int(round(d.get("lo", d["point"]))) - 1
            hi_i = int(round(d.get("hi", d["point"]))) - 1
            point_i = int(round(d["point"])) - 1
            lo_i = max(0, min(lo_i, len(dates) - 1))
            hi_i = max(lo_i, min(hi_i, len(dates) - 1))
            if lo_i <= hi_i and 0 <= point_i < len(dates):
                ax.axvspan(dates[lo_i], dates[hi_i], color=color, alpha=0.10)
                ax.axvline(dates[point_i], color=color, ls=":", lw=1.4, alpha=0.9)
                # 两条线可能挨得很近，纵向错开避免文字重叠；同时给出预计日期，方便直接对盘
                ax.annotate(f"{label}≈{d['point']:.0f}日\n(区间{d.get('lo',0):.0f}~{d.get('hi',0):.0f}日)\n"
                            f"约{dates[point_i].strftime('%m-%d')}",
                            xy=(dates[point_i], last_y), xytext=(5, 10 + k * 26),
                            textcoords="offset points", fontsize=8, color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75))

    ax.set_title(f"{name} 股价预测：每档只预测「第 N 个交易日」的收盘价（紫1/橙5/绿7/红30 日）")
    ax.set_xlabel("日期")
    ax.set_ylabel("收盘价")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.text(0.5, 0.012,
             "● 圆点 = 该档的目标价（一次性预测，不是逐日预测）　｜　竖线 = 该档的不确定性范围"
             "　｜　细点线仅示意方向",
             ha="center", fontsize=7.8, color="#666666")
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_candles(history: pd.DataFrame, out: str, name: str = "", bars: int = 60,
                 patterns: list[dict] | None = None):
    """K 线图（涨红跌绿）+ MA5/MA20 + 形态标注。

    纯 matplotlib 手绘（不依赖 mplfinance）：实体用矩形、上下影线用竖线。
    patterns: detect_patterns() 的输出，取最近 4 条有方向的形态标在对应 K 线上。
    """
    d = history.tail(bars).reset_index(drop=True)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, r in d.iterrows():
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        color = UP_COLOR if c >= o else DOWN_COLOR
        ax.vlines(i, l, h, color=color, lw=1.0)
        body_lo, body_hi = min(o, c), max(o, c)
        height = max(body_hi - body_lo, (h - l) * 0.02, 1e-6)
        ax.add_patch(Rectangle((i - 0.3, body_lo), 0.6, height,
                               facecolor=color, edgecolor=color, lw=0.6))
    ax.plot(d.index, d["close"].rolling(5).mean(), color="#1f77b4", lw=1.2, label="MA5")
    ax.plot(d.index, d["close"].rolling(20).mean(), color="#ff7f0e", lw=1.2, label="MA20")

    # 形态标注：只标有方向的、最近的 4 条
    if patterns:
        date_pos = {str(pd.to_datetime(x).date()): i for i, x in enumerate(d["date"])}
        shown = 0
        stack = {}      # 同一根 K 线上多条标注时纵向堆叠，避免文字互相压住
        for p in patterns:
            if shown >= 4 or p.get("direction") == 0:
                continue
            i = date_pos.get(p.get("date", ""))
            if i is None:
                continue
            r = d.loc[i]
            up = p["direction"] > 0
            y = float(r["low"]) if up else float(r["high"])
            lvl = stack.get(i, 0)
            stack[i] = lvl + 1
            off = (-16 - 14 * lvl) if up else (16 + 14 * lvl)
            ax.annotate(p["name"], xy=(i, y), xytext=(0, off),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=UP_COLOR if up else DOWN_COLOR,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                        arrowprops=dict(arrowstyle="->", lw=0.8,
                                        color=UP_COLOR if up else DOWN_COLOR))
            shown += 1

    step = max(1, len(d) // 12)
    ax.set_xticks(list(range(0, len(d), step)))
    ax.set_xticklabels([str(pd.to_datetime(x).date())[5:] for x in d["date"][::step]],
                       rotation=30, ha="right", fontsize=8)
    ax.set_title(f"{name} K 线形态（红涨绿跌；箭头为最近识别到的形态）")
    ax.set_ylabel("价格")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_equity(equity: pd.DataFrame, out: str, title: str = "TopK 轮动策略 vs 等权基准",
                metrics: dict | None = None):
    """策略净值曲线 + 回撤（下图）。"""
    if equity is None or equity.empty:
        return
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    colors = {"策略": "#d62728", "等权基准": "#7f8fa6"}
    for col in equity.columns:
        ax.plot(equity.index, equity[col], label=col,
                color=colors.get(col), lw=1.8 if col == "策略" else 1.2)
    ax.set_title(title)
    ax.set_ylabel("净值")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if "策略" in equity.columns:
        eq = equity["策略"]
        dd = eq / eq.cummax() - 1
        ax2.fill_between(dd.index, dd.values * 100, 0, color="#d62728", alpha=0.25)
        ax2.set_ylabel("回撤(%)")
        ax2.grid(alpha=0.3)

    if metrics:
        txt = (f"年化 {metrics.get('年化收益(%)')}% ｜ 夏普 {metrics.get('夏普')} "
               f"｜ 最大回撤 {metrics.get('最大回撤(%)')}% ｜ 超额 {metrics.get('超额年化(%)')}%")
        ax.text(0.01, 0.02, txt, transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_backtest(detail: pd.DataFrame, out: str, name: str = "", horizon: int = 7):
    """绘制某次回测的真实 vs 预测对比（DIRMO 下每切分为单点，用标记显示）。"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = list(range(len(detail)))
    ax.plot(x, detail["actual"], "-o", label="真实价", color="#1f77b4", markersize=7)
    ax.plot(x, detail["pred"], "-s", label="预测价", color="#d62728", markersize=7)
    for xi, (a, p) in enumerate(zip(detail["actual"], detail["pred"])):
        ax.annotate(f"{p:.0f}", (xi, p), textcoords="offset points", xytext=(0, 8), fontsize=7, color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(detail["date"], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{name} 回测对比（未来{horizon}日，真实 vs 预测，DIRMO 直接预测）")
    ax.set_ylabel("收盘价")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
