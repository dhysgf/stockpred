"""K 线形态识别（A 股常用经典形态）。

设计原则
--------
1. **全部向量化**：一次算完整个历史，供特征工程直接取列；另提供
   ``detect_patterns`` 只为界面/报告列出「最近几根K线命中了什么形态」。
2. **强度而非布尔**：每个形态给 0~1 的强度（由实体/影线比值决定，不是硬编码 1），
   聚合成 [-1, 1] 的连续分数，模型才能当因子用。
3. **必须带趋势上下文**：锤子线在下跌末端才是看涨信号，同样的形状出现在上涨末端
   是「上吊线」（看跌）。形态本身不看位置就没意义，这里用 MA5 斜率判趋势。

对外
----
- ``add_patterns(df)``：追加 pat_* 列（供 features.FEATURES 使用）
- ``detect_patterns(df, window=5)``：返回最近 window 根的形态命中列表（供展示）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 形态 → (方向, 基础强度, 中文说明)
PATTERN_INFO = {
    "bull_marubozu": (1, 0.6, "大阳线（实体饱满，多方主导）"),
    "bear_marubozu": (-1, 0.6, "大阴线（实体饱满，空方主导）"),
    "hammer": (1, 0.5, "锤子线（长下影，下跌末端承接强）"),
    "hanging_man": (-1, 0.4, "上吊线（长下影，上涨末端接力不足）"),
    "inv_hammer": (1, 0.4, "倒锤线（长上影，下跌末端试盘）"),
    "shooting_star": (-1, 0.5, "流星线（长上影，上方抛压重）"),
    "bull_engulf": (1, 0.8, "看涨吞没（阳线反包前阴线）"),
    "bear_engulf": (-1, 0.8, "看跌吞没（阴线反包前阳线）"),
    "harami_bull": (1, 0.3, "看涨孕线（缩量小实体，跌势停顿）"),
    "harami_bear": (-1, 0.3, "看跌孕线（缩量小实体，涨势停顿）"),
    "three_white": (1, 0.7, "红三兵（三连阳，重心逐日抬高）"),
    "three_crow": (-1, 0.7, "三只乌鸦（三连阴，重心逐日下移）"),
    "morning_star": (1, 0.8, "早晨之星（大阴→十字→大阳，底部反转）"),
    "evening_star": (-1, 0.8, "黄昏之星（大阳→十字→大阴，顶部反转）"),
    "gap_up": (1, 0.4, "向上跳空缺口（强势）"),
    "gap_down": (-1, 0.4, "向下跳空缺口（弱势）"),
    "breakout20": (1, 0.5, "突破 20 日新高"),
    "breakdown20": (-1, 0.5, "跌破 20 日新低"),
    "doji": (0, 0.0, "十字星（多空平衡，等待方向选择）"),
}

_BULL_KEYS = [k for k, v in PATTERN_INFO.items() if v[0] > 0]
_BEAR_KEYS = [k for k, v in PATTERN_INFO.items() if v[0] < 0]


def add_patterns(d: pd.DataFrame) -> pd.DataFrame:
    """在 K 线 DataFrame 上追加形态列。

    输入列：open, high, low, close（volume 可选）。
    输出：每个形态一列强度（0~1），以及聚合列
        pat_score  0~1（0.5=中性，>0.5 偏多，<0.5 偏空）
        pat_bull / pat_bear  最近一根的多/空形态强度（0~1）
        pat_n      最近一根命中形态个数
    """
    out = d.copy()
    o, h, l, c = out["open"], out["high"], out["low"], out["close"]
    prev_o, prev_c = o.shift(1), c.shift(1)
    prev_h, prev_l = h.shift(1), l.shift(1)
    rng = (h - l).replace(0, np.nan)
    body = c - o
    abody = body.abs()
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    # 前 5 根平均实体（不含当日，避免自比较）
    avg_body = abody.rolling(5).mean().shift(1).replace(0, np.nan)

    ma5 = c.rolling(5).mean()
    # 趋势上下文：均线方向 + 价格相对均线
    up_trend = (c > ma5) & (ma5 > ma5.shift(3))
    down_trend = (c < ma5) & (ma5 < ma5.shift(3))

    p = pd.DataFrame(index=out.index)

    # --- 单根：实体饱满 ---
    strong_body = (abody / rng > 0.65) & (abody > 1.2 * avg_body)
    p["bull_marubozu"] = np.where(strong_body & (body > 0), 1.0, 0.0)
    p["bear_marubozu"] = np.where(strong_body & (body < 0), 1.0, 0.0)

    # --- 单根：长影线（锤子/上吊、倒锤/流星）---
    small_body = abody / rng < 0.34
    long_lower = (lower >= 2 * abody) & (upper <= 0.35 * rng)
    long_upper = (upper >= 2 * abody) & (lower <= 0.35 * rng)
    p["hammer"] = np.where(long_lower & small_body & down_trend, 1.0, 0.0)
    p["hanging_man"] = np.where(long_lower & small_body & up_trend, 1.0, 0.0)
    p["inv_hammer"] = np.where(long_upper & small_body & down_trend, 1.0, 0.0)
    p["shooting_star"] = np.where(long_upper & small_body & up_trend, 1.0, 0.0)

    # --- 两根：吞没 / 孕线 ---
    p["bull_engulf"] = np.where(
        (prev_c < prev_o) & (body > 0) & (c >= prev_o) & (o <= prev_c)
        & (abody > abody.shift(1)), 1.0, 0.0)
    p["bear_engulf"] = np.where(
        (prev_c > prev_o) & (body < 0) & (c <= prev_o) & (o >= prev_c)
        & (abody > abody.shift(1)), 1.0, 0.0)
    # 孕线：当日实体完全被前一根实体包含，且明显缩小（与前后阳阴无关，方向由前一根决定）
    cur_hi = pd.concat([o, c], axis=1).max(axis=1)
    cur_lo = pd.concat([o, c], axis=1).min(axis=1)
    prev_hi = pd.concat([prev_o, prev_c], axis=1).max(axis=1)
    prev_lo = pd.concat([prev_o, prev_c], axis=1).min(axis=1)
    inside = (cur_hi <= prev_hi) & (cur_lo >= prev_lo) & (abody < 0.6 * abody.shift(1))
    p["harami_bull"] = np.where(inside & (prev_c < prev_o), 1.0, 0.0)
    p["harami_bear"] = np.where(inside & (prev_c > prev_o), 1.0, 0.0)

    # --- 三根：红三兵 / 三只乌鸦 ---
    b1, b2 = body.shift(1), body.shift(2)
    c1, c2 = c.shift(1), c.shift(2)
    p["three_white"] = np.where(
        (body > 0) & (b1 > 0) & (b2 > 0)
        & (c > c1) & (c1 > c2)
        & (body > 0.5 * avg_body), 1.0, 0.0)
    p["three_crow"] = np.where(
        (body < 0) & (b1 < 0) & (b2 < 0)
        & (c < c1) & (c1 < c2)
        & (abody > 0.5 * avg_body), 1.0, 0.0)

    # --- 三根：早晨之星 / 黄昏之星（大实体 → 缩量小实体 → 反向大实体收回一半以上）---
    b_first = body.shift(2)          # 第 1 根（大阴/大阳）
    ab_first = abody.shift(2)
    ab_mid = abody.shift(1)          # 第 2 根（星线）
    mid_first = (o.shift(2) + c.shift(2)) / 2
    p["morning_star"] = np.where(
        (b_first < 0) & (ab_mid < 0.4 * ab_first) & (body > 0) & (c > mid_first), 1.0, 0.0)
    p["evening_star"] = np.where(
        (b_first > 0) & (ab_mid < 0.4 * ab_first) & (body < 0) & (c < mid_first), 1.0, 0.0)

    # --- 缺口 ---
    p["gap_up"] = np.where(l > prev_h, 1.0, 0.0)
    p["gap_down"] = np.where(h < prev_l, 1.0, 0.0)

    # --- 突破（不含当日的前 20 日极值）---
    hi20 = h.rolling(20).max().shift(1)
    lo20 = l.rolling(20).min().shift(1)
    p["breakout20"] = np.where(c > hi20, 1.0, 0.0)
    p["breakdown20"] = np.where(c < lo20, 1.0, 0.0)

    # --- 十字星（无方向，仅标记）---
    p["doji"] = np.where(abody / rng < 0.1, 1.0, 0.0)

    # 强度加权：按 PATTERN_INFO 的基础强度缩放
    for k, (dirn, strength, _desc) in PATTERN_INFO.items():
        if k in p.columns:
            p[k] = p[k] * strength

    out = pd.concat([out, p.add_prefix("pat_")], axis=1)

    # 聚合：最近 3 根按 0.7^age 衰减求和（形态有时效性，昨天的信号不该和今天等价）
    sign_sum = pd.Series(0.0, index=out.index)
    for k in PATTERN_INFO:
        col = out.get(f"pat_{k}")
        if col is None:
            continue
        dirn = PATTERN_INFO[k][0]
        if dirn == 0:
            continue
        sign_sum = sign_sum + dirn * col
    decay = sign_sum.copy() * 0.0
    for age, w in ((0, 1.0), (1, 0.7), (2, 0.49)):
        decay = decay + sign_sum.shift(age).fillna(0.0) * w
    # tanh 压缩到 (-1,1)，避免连续同向形态把分数顶到饱和
    out["pat_score"] = 0.5 + 0.5 * np.tanh(decay / 1.5)
    out["pat_bull"] = out[[f"pat_{k}" for k in _BULL_KEYS]].sum(axis=1).clip(upper=1.0)
    out["pat_bear"] = out[[f"pat_{k}" for k in _BEAR_KEYS]].sum(axis=1).clip(upper=1.0)
    out["pat_n"] = (out[[f"pat_{k}" for k in PATTERN_INFO if k != "doji"]] > 0).sum(axis=1)
    return out


def detect_patterns(d: pd.DataFrame, window: int = 5) -> list[dict]:
    """列出最近 ``window`` 根 K 线命中的形态（供界面展示）。

    返回 [{date, name, cn, direction, strength, explain}]，按日期倒序（最新在前）。
    """
    if d is None or len(d) == 0:
        return []
    t = add_patterns(d) if "pat_score" not in d.columns else d
    tail = t.tail(max(window, 1))
    out = []
    for _, row in tail.iloc[::-1].iterrows():
        dt = pd.to_datetime(row["date"]).date().isoformat() if "date" in row else ""
        hits = []
        for k, (dirn, _s, desc) in PATTERN_INFO.items():
            col = f"pat_{k}"
            v = float(row.get(col, 0) or 0)
            if v > 0:
                hits.append((v, k, dirn, desc))
        hits.sort(reverse=True)
        for v, k, dirn, desc in hits:
            out.append({
                "date": dt, "key": k, "name": desc.split("（")[0],
                "cn": desc, "direction": dirn, "strength": round(v, 2),
            })
    return out


def pattern_summary(d: pd.DataFrame) -> dict:
    """当前技术形态面小结：最近 3 根的综合得分与多空形态列表。"""
    if d is None or len(d) == 0:
        return {"score": 0.5, "bull": 0.0, "bear": 0.0, "n": 0, "label": "无数据", "items": []}
    t = add_patterns(d) if "pat_score" not in d.columns else d
    last = t.iloc[-1]
    score = float(last.get("pat_score", 0.5))
    if score >= 0.65:
        label = "形态偏多"
    elif score <= 0.35:
        label = "形态偏空"
    else:
        label = "形态中性"
    return {
        "score": round(score, 4),
        "bull": round(float(last.get("pat_bull", 0)), 3),
        "bear": round(float(last.get("pat_bear", 0)), 3),
        "n": int(last.get("pat_n", 0)),
        "label": label,
        "items": detect_patterns(t, window=3),
    }
