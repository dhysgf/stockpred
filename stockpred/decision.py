"""联合决策：把 5 路信号加权成一个可解释的综合分。

为什么必须"可解释"
------------------
股价预测最容易被糊弄的地方就是"模型说涨"。这里每个信号都必须给出
原始取值、权重、贡献，任何一条结论都能被反查，也能看出是哪个信号在主导。

信号与量纲（全部归一到 [-1, 1]，正=偏多）
----------------------------------------
1. model  ：各周期 (prob_up - 0.5) * 2 的加权（**排除无技能的 30 日**）
2. pattern：K 线形态分 (pat_score - 0.5) * 2
3. volume ：量能/资金流分 (vol_score - 0.5) * 2
4. news   ：**大模型**对新闻的情绪研判 (sentiment-0.5)*2，按置信度收缩
5. policy ：**大模型**对政策/监管的研判 policy_impact，按置信度收缩
6. turn   ：P(底) - P(顶)（底部机会为正、顶部风险为负）

关于"大模型进决策"的边界（重要）
--------------------------------
新闻情绪与政策影响都是**外生**信息（大模型读的是新闻与政策，不是我们的模型输出），
所以这两路参与加权是成立的。
但 `llm.analyze_decision_llm`（让大模型复述/解读我们自己的量化结论）**不进加权**——
那是循环论证，等于让 LLM 给自己打分。它只做展示层的解释。

权重在 config.SIGNAL_WEIGHTS；缺失的信号会被剔除并**重新归一化**，
不会因为某项不可用就把综合分整体压低。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _clamp(x, lo=-1.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def _model_signal(probs: dict) -> tuple[float, str]:
    """多周期概率融合（只取有统计优势的周期）。"""
    usable = {h: p for h, p in (probs or {}).items()
              if h not in getattr(config, "NO_SKILL_HORIZONS", ())}
    if not usable:
        return 0.0, "无可用周期"
    prefer = {1: 0.3, 5: 0.4, 7: 0.3}
    wsum, acc, used = 0.0, 0.0, []
    for h, p in usable.items():
        w = prefer.get(h, 0.2)
        acc += w * (p - 0.5) * 2
        wsum += w
        used.append(f"{h}日{p*100:.1f}%")
    return _clamp(acc / wsum if wsum else 0.0), "、".join(used)


def _news_signal(news: dict | None) -> tuple[float, str]:
    """大模型对**新闻**的情绪研判（政策单独一路，见 _policy_signal）。

    情绪是外生信息（不是模型给自己打分），所以可以独立参与加权。
    置信度收缩：模型自己都没把握的判断，不该给足权重。
    """
    if not news:
        return 0.0, "无新闻数据"
    senti = float(news.get("sentiment", 0.5) or 0.5)
    conf = float(news.get("confidence", 0.3) or 0.3)
    raw = (senti - 0.5) * 2
    shrink = 0.4 + 0.6 * conf
    if not news.get("ok"):
        shrink *= 0.5      # 降级（词典情绪）再打对折
    return _clamp(raw * shrink), f"情绪{senti:.2f}/置信{conf:.2f}"


def _policy_signal(news: dict | None) -> tuple[float, str]:
    """大模型对**政策/监管**的研判。

    政策影响同样是外生信息，可以参与加权；但**词典降级路径没有政策理解能力**，
    此时 policy_impact 恒为 0，那不是"政策中性"而是"根本不知道政策"，
    所以在 decide() 里把它标为不可用并从归一化中剔除，而不是当 0 权重处理。
    """
    if not news:
        return 0.0, "无政策数据"
    policy = float(news.get("policy_impact", 0.0) or 0.0)
    conf = float(news.get("confidence", 0.3) or 0.3)
    shrink = 0.4 + 0.6 * conf
    return _clamp(policy * shrink), f"政策{policy:+.2f}/置信{conf:.2f}"


def _turn_signal(turn: dict | None, horizon: int = 10) -> tuple[float, str]:
    if not turn or not turn.get("probs"):
        return 0.0, "无拐点模型"
    probs = turn["probs"]
    key = horizon if horizon in probs else sorted(probs)[len(probs) // 2]
    p_top = float(probs[key].get("top", 0.0))
    p_bot = float(probs[key].get("bot", 0.0))
    return _clamp((p_bot - p_top) * 1.2), f"{key}日内 顶{p_top*100:.0f}%/底{p_bot*100:.0f}%"


def decide(model_probs: dict, pattern_score: float, volume_score: float,
           news: dict | None = None, turn: dict | None = None,
           extra_notes: list[str] | None = None) -> dict:
    """生成联合决策。返回 dict（score / advice / signals / notes ...）。"""
    s_model, model_note = _model_signal(model_probs)
    s_pat = _clamp((float(pattern_score or 0.5) - 0.5) * 2)
    s_vol = _clamp((float(volume_score or 0.5) - 0.5) * 2)
    s_news, news_note = _news_signal(news)
    s_policy, policy_note = _policy_signal(news)
    s_turn, turn_note = _turn_signal(turn)

    W = getattr(config, "SIGNAL_WEIGHTS", {})
    items = [
        ("model", "模型方向概率", s_model, W.get("model", 0.40), model_note),
        ("pattern", "K线形态", s_pat, W.get("pattern", 0.15),
         f"形态分 {float(pattern_score or 0.5):.2f}"),
        ("volume", "量能与资金", s_vol, W.get("volume", 0.15),
         f"量能分 {float(volume_score or 0.5):.2f}"),
        ("news", "大模型新闻研判", s_news, W.get("news", 0.10), news_note),
        ("policy", "大模型政策研判", s_policy, W.get("policy", 0.10), policy_note),
        ("turn", "拐点风险", s_turn, W.get("turn", 0.10), turn_note),
    ]
    # 缺失信号（明确不可用）不参与归一化。
    # policy 在词典降级时不可用：降级路径没有政策理解能力，policy_impact 恒为 0，
    # 那是"不知道"而不是"中性"，当 0 信号处理会假装这一路参与了投票。
    provider = str((news or {}).get("provider", "") or "")
    unavailable = {
        "news": news is None or not (news or {}).get("news_count", 0),
        "policy": (not news) or provider in ("fallback", "off", ""),
        "turn": not (turn or {}).get("probs"),
    }
    active = [it for it in items if not unavailable.get(it[0], False)]
    wsum = sum(it[3] for it in active) or 1.0
    score = sum(it[2] * it[3] for it in active) / wsum
    score = _clamp(score)

    if score >= config.DECISION_BUY_STRONG:
        advice, action = "强烈买入", "多信号共振偏多，可分批建仓（仍需自行控制仓位）"
    elif score >= config.DECISION_BUY:
        advice, action = "买入/增持", "偏多信号占优，可小仓位试多"
    elif score <= config.DECISION_SELL_STRONG:
        advice, action = "强烈卖出", "多信号共振偏空，注意规避"
    elif score <= config.DECISION_SELL:
        advice, action = "减持/卖出", "偏空信号占优，建议降低仓位"
    else:
        advice, action = "观望", "信号分歧或强度不足，等待更清晰的方向"

    # 一致度：同向信号占比（分歧大时降低结论可信度）
    signs = [1 if it[2] > 0.05 else (-1 if it[2] < -0.05 else 0) for it in active]
    nonzero = [s for s in signs if s != 0]
    agree = (max(nonzero.count(1), nonzero.count(-1)) / len(nonzero)) if nonzero else 0.0
    confidence = _clamp(abs(score) * 0.7 + agree * 0.3, 0.0, 1.0)

    notes = list(extra_notes or [])
    for key in ("news", "policy", "turn"):
        if unavailable.get(key):
            label = {"news": "新闻研判", "policy": "政策研判", "turn": "拐点"}[key]
            notes.append(f"{label}信号不可用，已从加权中剔除")

    return {
        "score": round(score, 4),
        "advice": advice,
        "action": action,
        "confidence": round(confidence, 3),
        "agreement": round(agree, 3),
        "signals": [{
            "key": k, "name": name,
            "value": round(v, 4),
            "weight": round(w, 3),
            "contrib": round(v * w / wsum, 4),
            "note": note,
        } for k, name, v, w, note in items],
        "notes": notes,
        "horizon": "5~10 个交易日",
    }


# =====================================================================
# 历史检验：形态 / 量能这两路信号到底有没有方向预测力？
# 这是唯一能验证"加进来是不是有用"的办法（新闻/政策无历史序列，无法回测）。
# =====================================================================
def _hist_volume_signal(panel: pd.DataFrame) -> pd.Series:
    """用**历史可得的量能分量**复原量能分（不含当日资金流/盘口，二者无历史）。"""
    obv = panel.get("obv_slope5", pd.Series(0.0, index=panel.index))
    upv = panel.get("up_vol_ratio", pd.Series(0.5, index=panel.index))
    mfi = panel.get("mfi", pd.Series(50.0, index=panel.index))
    s = (0.45 * np.tanh(obv * 4)
         + 0.35 * np.tanh((upv - 0.5) * 4)
         + 0.20 * np.tanh((mfi - 50) / 25.0))
    return 0.5 + 0.5 * np.tanh(s)


def eval_signals(panel: pd.DataFrame, horizons=(1, 5, 7, 30)) -> pd.DataFrame:
    """横截面样本外方向检验：各信号的命中率与 IC（信息系数）。

    信号只用 T 日及之前数据，收益用 T→T+h 的真实涨跌，无前视偏差。
    """
    if panel is None or panel.empty:
        return pd.DataFrame()
    p = panel.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    pat = p.get("pat_score", pd.Series(0.5, index=p.index)).astype(float)
    # 量能分可能因 MFI 暖机期产生 NaN，直接参与比较会让命中率被系统性低估，先中性填充
    vol = _hist_volume_signal(p).fillna(0.5)
    p["_sig_pat"] = ((pat - 0.5) * 2).fillna(0.0)
    p["_sig_vol"] = ((vol - 0.5) * 2).fillna(0.0)
    rows = []
    THR = 0.10      # 只统计"明确表态"的样本（|信号|>0.10）；阈值太低等于全样本
    for h in horizons:
        fwd = p.groupby("symbol")["close"].shift(-h) / p["close"] - 1
        m = fwd.notna()
        if m.sum() < 1000:
            continue
        y = (fwd[m] > 0).astype(int)
        for key, label in (("_sig_pat", "K线形态"), ("_sig_vol", "量能")):
            s = p.loc[m, key]
            strong = s.abs() > THR
            n = int(strong.sum())
            hit = ((s > 0) == (y == 1)) & strong
            ic = float(np.corrcoef(s, fwd[m])[0, 1]) if s.std() > 0 else float("nan")
            rows.append({
                "周期(日)": h, "信号": label,
                "明确表态占比(%)": round(n / len(s) * 100, 1),
                "明确时命中率(%)": round(float(hit.sum() / max(n, 1) * 100), 1),
                "IC": round(ic, 4),
                "样本数": len(s),
            })
    return pd.DataFrame(rows)
