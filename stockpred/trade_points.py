"""买卖点预判 + 目标价位区间。

回答两个问题
------------
1. **目标价位区间**：5 日 / 7 天 / 30 天各出一档，给出下界 / 中位 / 上界。
   - 中位 = DIRMO 直接预测的 h 日终点价（不是递归路径的终点，避免复利漂移）
   - 带宽 = max(多模型分歧, 波动率外推) × z，在 model.forecast 里算好（band_ret）
2. **买卖点**：从上面的区间 + 技术位（MA5/MA20/ATR）推导出可执行的价位。

口径纪律
--------
* 区间不是"预测精度"的承诺：它是"在这个价位上下，模型认为合理范围有多宽"。
  带宽越宽说明越不该重仓。
* 止损受风险预算约束（config.TRADE.risk_budget_pct），不会因为模型给出很远的
  下界就让人在一个必然巨亏的位置止损。
* 30 日周期在 config.NO_SKILL_HORIZONS 里（无方向预测力），
  所以 30 日只给价格区间，**不给方向判断**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _f(x, default=None):
    """安全取浮点：None/NaN 返回 default。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else v


def _horizon_rows(preds: dict, base: float, horizons: list[int]) -> list[dict]:
    """把各周期的预测路径折成"目标价位区间"行。"""
    rows = []
    for h in horizons:
        df = preds.get(h)
        if df is None or len(df) == 0:
            continue
        last = df.iloc[-1]
        target = _f(last.get("pred_close"))
        if target is None or base <= 0:
            continue
        band = _f(last.get("band_ret"), 0.0) or 0.0
        spread = _f(last.get("member_spread"), 0.0) or 0.0
        point_ret = target / base - 1.0
        rows.append({
            "horizon": h,
            "date": str(last.get("date", "")),
            "target": round(target, 2),
            "lower": round(base * (1 + point_ret - band), 2),
            "upper": round(base * (1 + point_ret + band), 2),
            "expected_ret": round(point_ret * 100, 2),
            "band_pct": round(band * 100, 2),
            "member_spread": round(spread * 100, 2),
            "prob_up": _f(last.get("prob_up")),
            "direction": str(last.get("pred_dir", "")),
            "no_skill": h in getattr(config, "NO_SKILL_HORIZONS", ()),
        })
    return rows


def build_trade_points(preds: dict, kline: pd.DataFrame, tech: pd.DataFrame,
                       decision: dict | None = None, turn: dict | None = None,
                       quote: dict | None = None) -> dict:
    """生成买卖点预判与目标价位区间。"""
    tr = getattr(config, "TRADE", {}) or {}
    horizons = list(tr.get("horizons", [5, 7, 30]))
    z = float(tr.get("band_z", 1.28))
    budget = float(tr.get("risk_budget_pct", 0.08))

    base = float(kline["close"].iloc[-1])
    rows = _horizon_rows(preds, base, horizons)
    by_h = {r["horizon"]: r for r in rows}

    # ---------------- 技术位
    def _tech(col):
        try:
            if tech is None or len(tech) == 0 or col not in tech.columns:
                return None
            v = _f(tech[col].iloc[-1])
            return v
        except Exception:
            return None

    ma5 = _tech("ma5")
    ma20 = _tech("ma20")
    atr_norm = _tech("atr_norm")
    atr_price = (atr_norm * base) if (atr_norm is not None and base) else None
    daily_vol = None
    vwin = int(tr.get("vol_window", 20))
    try:
        if len(kline) > vwin + 2:
            daily_vol = _f(kline["close"].astype(float).tail(vwin + 1).pct_change().std())
    except Exception:
        daily_vol = None

    # ---------------- 买点
    h5 = by_h.get(5) or (rows[0] if rows else None)
    h7 = by_h.get(7)
    h30 = by_h.get(30)

    buy = {"note": "缺少短周期预测，无法给出买点"}
    if h5:
        low = h5["lower"]
        high = base
        if low > base:                       # 区间整体在现价之上 → 没有回踩空间
            low, high = round(base * 0.97, 2), base
        buy = {
            "pullback_zone": [round(low, 2), round(high, 2)],
            "pullback_note": "价格回落到该区间、且 5 日线不跌破 MA20 时分批建仓",
            "breakout_trigger": h5["upper"],
            "breakout_note": "放量（量比>1.2）站稳该价上方视为趋势确认，可用剩余仓位跟进",
            "short_term_support": h5["lower"],
            "ma20_price": round(ma20, 2) if ma20 else None,
            "ma20_role": ("阻力" if (ma20 and ma20 > base) else "支撑") if ma20 else None,
            "ma5_price": round(ma5, 2) if ma5 else None,
        }

    # ---------------- 卖点
    sell = {"note": "缺少中长周期预测，无法给出卖点"}
    if h7 and h30:
        tp_low = max(h7["target"], base)
        tp_high = max(h30["upper"], tp_low)
        stop = max(h30["lower"], base * (1 - budget))
        sell = {
            "take_profit_zone": [round(tp_low, 2), round(tp_high, 2)],
            "take_profit_note": "进入该区间分批止盈：下沿减 1/3，上沿再减 1/3",
            "stop_loss": round(stop, 2),
            "stop_loss_pct": round((stop / base - 1) * 100, 2),
            "stop_loss_note": f"日收盘跌破该价离场（30 日模型下界，且单笔风险不超过 {budget*100:.0f}%）",
            "atr_stop": round(base - 1.5 * atr_price, 2) if atr_price else None,
        }

    # ---------------- 风险
    risk = {
        "atr_pct": round(atr_norm * 100, 2) if atr_norm is not None else None,
        "daily_vol_pct": round(daily_vol * 100, 2) if daily_vol is not None else None,
        "risk_level": "高" if (atr_norm or 0) > 0.035 else ("中" if (atr_norm or 0) > 0.02 else "低"),
        "risk_budget_pct": round(budget * 100, 2),
        "band_z": z,
    }

    # ---------------- 文字结论
    notes = []
    for r in rows:
        tag = "（该周期无方向预测力，只给价格区间）" if r["no_skill"] else ""
        notes.append(
            f"{r['horizon']}日目标价 {r['lower']}~{r['upper']}（中位 {r['target']}，"
            f"预期 {r['expected_ret']:+.2f}%，带宽 ±{r['band_pct']:.2f}%）{tag}"
        )
    if daily_vol is not None:
        notes.append(f"区间带宽取 max(多模型分歧, 日波动{daily_vol*100:.2f}%×√h×{z})，"
                     f"模型越不一致/波动越大 → 区间越宽 → 越不该重仓")
    if (turn or {}).get("days"):
        for kind, cn in (("top", "顶"), ("bot", "底")):
            d = turn["days"].get(kind)
            if d:
                notes.append(f"预计下一个{cn}拐点约 {d['point']:.0f} 个交易日后"
                             f"（区间 {d['lo']:.0f}~{d['hi']:.0f} 日），可作为时间维度的加减仓参考")
    if decision:
        notes.append(f"综合决策：{decision.get('advice','')}（综合分 {decision.get('score',0):+.3f}）")

    return {
        "per_horizon": rows,
        "buy_points": buy,
        "sell_points": sell,
        "risk": risk,
        "notes": notes,
        "base_price": base,
    }
