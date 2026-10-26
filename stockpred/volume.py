"""量能与买卖盘分析：量价配合、OBV、MFI、资金流向、五档委比。

分层很重要（这条是踩过的坑）
--------------------------
- **历史可得的量能指标**（OBV/MFI/量比/量价相关性/上涨量占比）→ 可以进模型当特征，
  因为每个历史交易日都能算出来，训练/预测口径一致。
- **只有当前快照的资金流**（新浪 MoneyFlow 只给"今天"的超大/大/中/小单净额）
  和 **五档盘口委比** → **不能**进模型（没有历史序列，训练时是常量，树模型无法切分），
  只能放在「决策层」作为当期证据使用。

这个区分必须显式写清楚，否则会重犯 9/29 那次「新闻情绪当特征却等于常量」的错误。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_volume_indicators(d: pd.DataFrame) -> pd.DataFrame:
    """追加历史可得的量能指标列。

    输出：mfi, obv_norm, obv_slope5, vol_expand, up_vol_ratio, pv_corr
    """
    out = d.copy()
    c, v = out["close"], out["volume"]
    ret = c.pct_change()

    # --- MFI(14)：带量的 RSI，衡量资金推动强弱 ---
    typ = (out["high"] + out["low"] + c) / 3
    mf = typ * v
    pos_mf = mf.where(typ.diff() > 0, 0.0).rolling(14).sum()
    neg_mf = mf.where(typ.diff() < 0, 0.0).rolling(14).sum()
    out["mfi"] = 100 - 100 / (1 + pos_mf / (neg_mf + 1e-9))

    # --- OBV（能量潮）：涨日加量、跌日减量 ---
    obv = (np.sign(ret.fillna(0)) * v).cumsum()
    out["obv"] = obv
    obv_mu = obv.rolling(20).mean()
    obv_sd = obv.rolling(20).std().replace(0, np.nan)
    out["obv_norm"] = (obv - obv_mu) / obv_sd
    # 5 日 OBV 斜率按 20 日成交量标准化 → 不同股票可比
    out["obv_slope5"] = (obv - obv.shift(5)) / (v.rolling(20).mean() * 5 + 1e-9)

    # --- 量能放大（相对 20 日均量）---
    out["vol_expand"] = v / (v.rolling(20).mean() + 1e-9)

    # --- 近 20 日「上涨日成交量占比」：>0.5 说明买盘更积极 ---
    up_vol = v.where(ret > 0, 0.0).rolling(20).sum()
    tot_vol = v.rolling(20).sum()
    out["up_vol_ratio"] = up_vol / (tot_vol + 1e-9)

    # --- 量价相关性（20 日）：正=放量上涨健康，负=放量下跌/缩量上涨 ---
    out["pv_corr"] = ret.rolling(20).corr(v.pct_change())

    return out


def obv_divergence(d: pd.DataFrame, window: int = 20) -> dict:
    """OBV 与价格的背离检测（顶部/底部背离是拐点的重要前兆）。"""
    if d is None or len(d) < window + 6:
        return {"top": False, "bottom": False, "text": "样本不足"}
    t = d if "obv" in d.columns else add_volume_indicators(d)
    c = t["close"]
    obv = t["obv"]
    price_new_high = c.iloc[-1] >= c.tail(window).max()
    obv_new_high = obv.iloc[-1] >= obv.tail(window).max()
    price_new_low = c.iloc[-1] <= c.tail(window).min()
    obv_new_low = obv.iloc[-1] <= obv.tail(window).min()
    top = bool(price_new_high and not obv_new_high)
    bottom = bool(price_new_low and not obv_new_low)
    if top:
        text = f"价格创 {window} 日新高但 OBV 未同步 → 顶部量能背离"
    elif bottom:
        text = f"价格创 {window} 日新低但 OBV 未同步 → 底部量能背离"
    else:
        text = "无明显量能背离"
    return {"top": top, "bottom": bottom, "text": text}


def depth_stats(quote: dict) -> dict:
    """五档盘口：委买/委卖量、委比、委差。"""
    bids = (quote or {}).get("bids") or []
    asks = (quote or {}).get("asks") or []
    bv = sum(x[0] for x in bids) if bids else 0.0
    av = sum(x[0] for x in asks) if asks else 0.0
    total = bv + av
    weibi = (bv - av) / total * 100 if total else 0.0
    if weibi >= 20:
        label = "买盘明显强于卖盘"
    elif weibi <= -20:
        label = "卖盘明显强于买盘"
    else:
        label = "买卖盘相对均衡"
    return {"bid_vol": bv, "ask_vol": av, "weibi": round(weibi, 2),
            "weicha": bv - av, "label": label}


def volume_score(d: pd.DataFrame, quote: dict | None = None, flow: dict | None = None) -> dict:
    """把量能/资金/盘口压成 0~1 的分数（0.5 中性），返回可解释的明细。

    每一项都是 [-1,1] 的贡献，最后 tanh 压缩再加 0.5，避免某一项主导。
    """
    if d is None or len(d) < 25:
        return {"score": 0.5, "label": "样本不足", "items": [], "detail": {}}

    t = d if "vol_expand" in d.columns else add_volume_indicators(d)
    last = t.iloc[-1]
    ret = float(last.get("ret", t["close"].pct_change().iloc[-1]) or 0)
    ve = float(last.get("vol_expand", 1.0) or 1.0)
    obv_sl = float(last.get("obv_slope5", 0) or 0)
    mfi = float(last.get("mfi", 50) or 50)
    upvr = float(last.get("up_vol_ratio", 0.5) or 0.5)
    pvc = float(last.get("pv_corr", 0) or 0)
    div = obv_divergence(t)

    items = []          # (名称, 取值, 贡献[-1,1], 说明)

    # 1) 量价配合
    if ret > 0 and ve >= 1.2:
        s = min(1.0, (ve - 1.0)) * 0.5
        items.append(("量价配合", f"涨 {ret*100:+.2f}%，量比 {ve:.2f}", s, "放量上涨，买盘承接积极"))
    elif ret < 0 and ve >= 1.2:
        s = -min(1.0, (ve - 1.0)) * 0.5
        items.append(("量价配合", f"跌 {ret*100:+.2f}%，量比 {ve:.2f}", s, "放量下跌，抛压较重"))
    elif ret > 0 and ve < 0.8:
        s = -0.15
        items.append(("量价配合", f"涨 {ret*100:+.2f}%，量比 {ve:.2f}", s, "缩量上涨，追高动能不足"))
    elif ret < 0 and ve < 0.8:
        s = 0.15
        items.append(("量价配合", f"跌 {ret*100:+.2f}%，量比 {ve:.2f}", s, "缩量下跌，抛压趋缓"))
    else:
        s = 0.0
        items.append(("量价配合", f"涨跌 {ret*100:+.2f}%，量比 {ve:.2f}", 0.0, "量价关系中性"))

    # 2) OBV 趋势（5 日）
    s_obv = float(np.tanh(obv_sl * 4))
    items.append(("OBV 资金潮", f"5日斜率 {obv_sl:+.3f}", s_obv * 0.3,
                  "OBV 上行=量能向多头倾斜" if s_obv > 0 else "OBV 下行=量能向空头倾斜"))

    # 3) OBV 背离
    if div["top"]:
        items.append(("量能背离", "顶部背离", -0.4, div["text"]))
    elif div["bottom"]:
        items.append(("量能背离", "底部背离", 0.4, div["text"]))
    else:
        items.append(("量能背离", "无", 0.0, div["text"]))

    # 4) MFI（>80 超买、<20 超卖，这里做温和回归项）
    s_mfi = 0.0
    if mfi >= 80:
        s_mfi = -0.2
    elif mfi <= 20:
        s_mfi = 0.2
    items.append(("MFI(14)", f"{mfi:.1f}", s_mfi,
                  "资金面超买，注意回吐" if mfi >= 80 else ("资金面超卖，或有反抽" if mfi <= 20 else "资金面中性")))

    # 5) 上涨日成交量占比（20 日）
    s_up = float(np.tanh((upvr - 0.5) * 4))
    items.append(("上涨量占比(20日)", f"{upvr*100:.0f}%", s_up * 0.25,
                  "买盘主导" if upvr > 0.55 else ("卖盘主导" if upvr < 0.45 else "买卖均衡")))

    # 6) 量价相关性
    items.append(("量价相关(20日)", f"{pvc:+.2f}", float(np.tanh(pvc * 1.5)) * 0.15,
                  "放量伴随上涨，趋势健康" if pvc > 0.2 else ("放量伴随下跌，趋势偏弱" if pvc < -0.2 else "量价关系不明显")))

    # 7) 当期资金流（不进模型，只做当期证据）
    if flow and flow.get("ok"):
        ratio = float(flow.get("net_ratio", 0) or 0)     # 净额/成交额，单位 %
        s_flow = float(np.tanh(ratio / 8.0)) * 0.6       # 净流入占成交额 8% 视为较强
        items.append(("主力资金净额", f"{flow.get('net_amount_wan', 0):+.0f} 万元（占成交 {ratio:+.2f}%）",
                      s_flow, "主力净流入" if ratio > 0 else "主力净流出"))
    else:
        items.append(("主力资金净额", "不可用", 0.0, "资金流接口未返回数据，该项不参与"))

    # 8) 五档委比（仅盘中有效）
    if quote and quote.get("bids") and quote.get("asks"):
        ds = depth_stats(quote)
        s_wb = float(np.tanh(ds["weibi"] / 40.0)) * 0.3
        items.append(("五档委比", f"{ds['weibi']:+.1f}%", s_wb, ds["label"]))
    else:
        items.append(("五档委比", "不可用", 0.0, "非交易时段或行情未返回盘口"))

    total = sum(x[2] for x in items)
    score = 0.5 + 0.5 * float(np.tanh(total))
    if score >= 0.62:
        label = "量能偏多"
    elif score <= 0.38:
        label = "量能偏空"
    else:
        label = "量能中性"
    return {
        "score": round(score, 4),
        "label": label,
        "items": [{"name": n, "value": v, "contrib": round(c, 3), "note": note} for n, v, c, note in items],
        "detail": {
            "ret": ret, "vol_expand": ve, "obv_slope5": obv_sl, "mfi": mfi,
            "up_vol_ratio": upvr, "pv_corr": pvc, "divergence": div,
            "depth": depth_stats(quote) if quote else {},
        },
    }
