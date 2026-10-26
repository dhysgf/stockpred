"""端到端分析编排：CLI 与 Web 共用同一条链路（单一事实来源）。

之前 run.py 与 web.py 各自复制了一份流程，结果两边都改不齐（还留了一个
``bt`` 未定义的 NameError）。现在统一走这里。

链路
----
行情 → 指数因子 → 股票池面板 → 新闻(个股/政策/市场) → 大模型文本分析
     → 涨跌/价格模型 → K线形态 → 量能与资金 → 拐点模型 → 联合决策 → 回测/图表
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import config
from .data_fetcher import load, fetch_money_flow
from .news_fetcher import collect_news
from .sentiment import analyze_news
from .features import add_technicals, build_horizon_samples, enrich_with_index, FEATURES
from .model import forecast
from .models import build_forecaster
from .patterns import pattern_summary
from .volume import volume_score
from .llm import analyze_news_llm, status as llm_status
from .turning_point import (TurningPointModel, build_turn_samples, recent_turns,
                            eval_turning)
from .decision import decide
from .trade_points import build_trade_points
from .backtest import walk_forward_backtest
from .pool import (fetch_pool, build_panel, zscore_panel, pooled_samples,
                   pooled_label_scale, cross_stats, walk_forward_backtest_pooled,
                   target_row)


def _log(verbose: bool, msg: str):
    if verbose:
        print(msg)


def _collect_levels(trade_points: dict, tech, last_close: float) -> dict:
    """汇总"客观关键价位"，供大模型写条件规则时引用。

    这些价位都是系统按固定规则算出来的（低吸区 = 5 日预测区间下沿、
    突破确认价 = 区间上沿、止损 = MA20 ± ATR…），给 LLM 当锚点，
    它写的触发条件才不会停留在"如果放量上涨"这种空话。

    注意：这里**不把量化模型的目标价预测给 LLM**（那是价格预测部分要独立对照的），
    只给"规则会用到的价位"——价位是客观的，不构成结论泄漏。
    """
    out = {}
    buy = (trade_points or {}).get("buy_points") or {}
    sell = (trade_points or {}).get("sell_points") or {}
    risk = (trade_points or {}).get("risk") or {}

    pz = buy.get("pullback_zone")
    if isinstance(pz, (list, tuple)) and len(pz) >= 2:
        out["低吸区（回踩建仓）"] = [pz[0], pz[1]]
    for key, label in (("breakout_trigger", "突破确认价（放量站稳）"),
                       ("short_term_support", "短期强支撑"),
                       ("ma20_price", "MA20 位置"),
                       ("ma5_price", "MA5 位置")):
        v = buy.get(key)
        if isinstance(v, (int, float)):
            out[label] = round(float(v), 2)
    pz2 = sell.get("take_profit_zone")
    if isinstance(pz2, (list, tuple)) and len(pz2) >= 2:
        out["止盈减仓区"] = [pz2[0], pz2[1]]
    for key, label in (("stop_loss", "止损位"), ("risk_note_price", "风险位")):
        v = risk.get(key)
        if isinstance(v, (int, float)):
            out[label] = round(float(v), 2)
    if last_close:
        out["当前收盘价"] = round(float(last_close), 2)
    # 量能阈值：突破确认要求量比 > 1.2，给 LLM 写规则时用
    out["放量确认量比阈值"] = "1.2（量比 > 1.2 且站稳突破价视为趋势确认）"
    return out


def _fit_model(samples: dict, pz, symbol: str, use_pool: bool,
               ret_scale: dict | None = None):
    """按 config.MODEL 训练模型，失败自动降级（lstm→gbdt→rf），并说明降级原因。

    降级必须显式记录：悄悄退回随机森林、却仍显示"LSTM"是最要不得的做法。
    """
    from .models import build_forecaster

    requested = str(getattr(config, "MODEL", "rf")).lower()
    chain = {"lstm": ["lstm", "gbdt", "rf"],
             "ensemble": ["ensemble", "gbdt", "rf"],
             "gbdt": ["gbdt", "rf"]}.get(requested, [requested])
    reasons = []
    ctx = {"panel": pz, "symbol": symbol} if pz is not None else None
    for name in chain:
        if name == "lstm" and ctx is None:
            reasons.append("LSTM 需要股票池面板（单股模式无法切序列）")
            continue
        try:
            m = build_forecaster(name)
            if ret_scale:
                m.ret_scale = ret_scale
            m.fit(samples, ctx)
            if m.fit_ok:
                note = m.note or ""
                if reasons:
                    note = (note + "；" if note else "") + "降级原因：" + "；".join(reasons)
                return m, note
            reasons.append(f"{name} 未产出可用周期")
        except Exception as e:
            reasons.append(f"{name}：{type(e).__name__} {str(e)[:100]}")
    # 全部失败时返回一个未训练的 RF 占位（调用方会抛"样本不足"）
    fb = build_forecaster("rf")
    fb.note = "；".join(reasons)
    return fb, fb.note


def analyze(symbol: str, name: str = "", horizons: list[int] | None = None,
            do_backtest: bool = True, use_llm: bool = True,
            verbose: bool = False, with_turn_eval: bool = False,
            llm_override: dict | None = None) -> dict:
    """跑完整分析，返回结构化结果（Web 直接渲染，CLI 打印）。

    llm_override: 网页端传入的 {"api_key","base_url","model"}，让访客用自己的额度。
    """
    t_start = time.time()
    H = list(horizons or config.HORIZONS)
    timings = {}

    # 1) 行情 + 指数
    t = time.time()
    market = load(symbol, datalen=config.KLINE_LEN, source=config.DATA_SOURCE)
    if market.kline is None or market.kline.empty:
        raise RuntimeError(f"未获取到 {symbol} 的历史数据，请检查代码是否正确")
    idx_kline = None
    try:
        idx_kline = load(config.INDEX_SYMBOL, datalen=config.KLINE_LEN,
                         source=config.DATA_SOURCE).kline
    except Exception as e:
        _log(verbose, f"      [warn] 指数获取失败：{e}")
    kline = enrich_with_index(market.kline, idx_kline)
    tech = add_technicals(kline)
    quote = market.quote or {}
    flow = fetch_money_flow(symbol)
    timings["行情"] = round(time.time() - t, 1)

    # 2) 股票池面板（横截面训练）
    t = time.time()
    try:
        pl = fetch_pool(datalen=config.KLINE_LEN, source=config.DATA_SOURCE, extra=[symbol])
        panel = build_panel(pl, idx_kline)
    except Exception as e:
        _log(verbose, f"      [warn] 股票池失败，退回单股模式：{e}")
        panel = pd.DataFrame()
    use_pool = (not panel.empty) and (symbol in set(panel.get("symbol", [])))
    timings["股票池"] = round(time.time() - t, 1)

    # 3) 新闻 + 大模型
    t = time.time()
    groups = collect_news(symbol, name=name, stock_num=20,
                          market_num=config.NEWS_NUM, policy_num=12)
    all_news = [it for k in ("stock", "policy", "market") for it in (groups.get(k) or [])]
    local_senti = analyze_news(all_news)
    last_close = float(kline["close"].iloc[-1])
    price = float(quote.get("price") or last_close)
    ret5 = float(tech["mom5"].iloc[-1] * 100) if "mom5" in tech else 0.0
    ret20 = float(tech["mom20"].iloc[-1] * 100) if "mom20" in tech else 0.0
    llm_res = (analyze_news_llm(groups, code=symbol, name=name or quote.get("name", ""),
                                price=price, ret5=ret5, ret20=ret20,
                                vol_expand=float(tech["vol_expand"].iloc[-1]),
                                net_ratio=float(flow.get("net_ratio", 0) or 0),
                                override=llm_override)
               if use_llm else None)
    if llm_res is None:
        llm_res = {"ok": False, "provider": "off", "model": "", "sentiment": local_senti["mean"],
                   "policy_impact": 0.0, "confidence": 0.0, "stance": "中性",
                   "key_drivers": [], "risks": [], "summary": "已关闭大模型分析",
                   "news_count": len(all_news), "note": "已关闭大模型分析（使用本地词典情绪）",
                   "_ts": 0}
    senti_value = float(llm_res.get("sentiment", local_senti["mean"])) if llm_res.get("ok") \
        else float(local_senti["mean"])
    timings["新闻+LLM"] = round(time.time() - t, 1)

    # 4) 涨跌/价格模型（模型可切换：rf / gbdt / lstm / ensemble）
    t = time.time()
    zstats = None
    pz = None
    if use_pool:
        pz = zscore_panel(panel)
        samples = pooled_samples(pz, senti_value, H)
        # 关键：标准化统计量必须来自**原始**截面（pz 已标准化，再取均值≈0/方差≈1，等于没做）
        zstats = cross_stats(panel)
    else:
        samples = build_horizon_samples(tech, senti_value, H)

    model, model_fallback = _fit_model(
        samples, pz if use_pool else None, symbol, use_pool=use_pool,
        ret_scale=(pooled_label_scale(panel, H)
                   if (use_pool and str(getattr(config, "LABEL_MODE", "ret")).lower() == "csrank")
                   else None))
    if not model.fit_ok:
        raise RuntimeError("有效训练样本不足，无法建模（该股上市时间过短且股票池不可用）")
    preds = forecast(model, kline, senti_value, H, zstats=zstats,
                     ctx_panel=pz if getattr(model, "needs_seq", False) else None,
                     ctx_symbol=symbol)
    model_probs = {h: float(preds[h]["prob_up"].iloc[0]) for h in H if h in preds}
    # 融合模型在预测阶段掉队的成员（如单股模式无面板 → LSTM 无法预测）要如实标注
    pf = getattr(model, "predict_failures", None)
    if pf:
        model_note = (getattr(model, "note", "") or "") + \
            "；预测阶段掉队：" + ",".join(f"{k}({v})" for k, v in pf.items())
    timings["涨跌模型"] = round(time.time() - t, 1)

    # 5) 形态 / 量能
    pat = pattern_summary(tech)
    vol = volume_score(tech, quote, flow)

    # 6) 拐点模型
    t = time.time()
    turn_res, turn_eval = {}, pd.DataFrame()
    if getattr(config, "TURN_HORIZONS", None):
        try:
            if use_pool:
                tpanel = panel
            else:
                tpanel = tech.copy()
                tpanel["symbol"] = symbol
            tfeat = [c for c in FEATURES if c in tpanel.columns]
            tsamples = build_turn_samples(tpanel, tfeat, config.TURN_HORIZONS,
                                          sentiment=senti_value)
            tmodel = TurningPointModel()
            tmodel.fit(tsamples)
            if tmodel.fit_ok:
                # 拐点模型在**原始特征**上训练（RF 对单调变换不敏感，train/predict 必须同尺度），
                # 所以这里不再套用横截面 z-score，否则训练阈值和预测输入不在同一尺度。
                from .features import last_feature_row

                Xr = last_feature_row(tech, senti_value)
                turn_res = tmodel.predict(Xr)
                turn_res["recent"] = recent_turns(kline, lookback=20)
                turn_res["horizons"] = tmodel.horizons
            if with_turn_eval and use_pool:
                turn_eval = eval_turning(panel, tfeat, config.TURN_HORIZONS,
                                         sentiment=senti_value)
        except Exception as e:
            _log(verbose, f"      [warn] 拐点模型失败：{e}")
            turn_res = {}
    timings["拐点模型"] = round(time.time() - t, 1)

    # 7) 联合决策 + 大模型对决策的二次解读（解释层，不增加信号权重）
    decision = decide(model_probs, pat["score"], vol["score"], llm_res, turn_res,
                      extra_notes=[f"30 日方向无预测力（大样本 48.3%），未参与方向加权"]
                      if 30 in getattr(config, "NO_SKILL_HORIZONS", ()) else None)
    llm_decision = {"ok": False, "note": "未开启大模型"}
    if use_llm:
        try:
            from .llm import analyze_decision_llm
            llm_decision = analyze_decision_llm(
                decision, groups, name=name or quote.get("name", ""), code=symbol,
                turn=turn_res, override=llm_override)
        except Exception as e:
            _log(verbose, f"      [warn] 决策解读失败：{e}")
            llm_decision = {"ok": False, "note": f"决策解读失败：{e}"}

    # 8) 买卖点预判 + 目标价位区间（5/7/30 日各一档）
    t = time.time()
    try:
        trade_points = build_trade_points(preds=preds, kline=kline, tech=tech,
                                          decision=decision, turn=turn_res, quote=quote)
    except Exception as e:
        _log(verbose, f"      [warn] 买卖点预判失败：{e}")
        trade_points = {}
    timings["买卖点"] = round(time.time() - t, 1)

    # 8.5) 大模型独立价格预测 + 条件规则预判（第二意见，**不参与加权**）
    #   放在买卖点之后：要先把系统算出的关键价位（低吸区/突破价/止盈区/止损位）
    #   作为客观锚点喂给大模型，它写的条件规则才能引用真实价位而不是空话。
    llm_forecast = {"ok": False, "note": "未开启大模型"}
    if use_llm:
        t = time.time()
        try:
            from .llm import analyze_forecast_llm
            levels = _collect_levels(trade_points, tech, last_close)
            llm_forecast = analyze_forecast_llm(
                kline, tech, news=llm_res, levels=levels,
                name=name or quote.get("name", ""), code=symbol,
                override=llm_override)
            if llm_forecast.get("ok"):
                _log(verbose, f"      大模型独立预测 {llm_forecast.get('model')}"
                              f"（{llm_forecast.get('elapsed', 0)}s，"
                              f"方向 {llm_forecast.get('direction')}，"
                              f"{len(llm_forecast.get('forecasts', []))} 个周期，"
                              f"{len(llm_forecast.get('rules', []))} 条规则）")
        except Exception as e:
            _log(verbose, f"      [warn] 大模型价格预测失败：{e}")
            llm_forecast = {"ok": False, "note": f"大模型价格预测失败：{e}"}
        timings["大模型预测"] = round(time.time() - t, 1)

    # 9) 回测
    t = time.time()
    bt = {"summary": pd.DataFrame(), "details": {}}
    if do_backtest:
        if len(kline) < 200:
            bt["note"] = f"历史仅 {len(kline)} 根日线，无法做样本外回测"
        elif use_pool:
            # 大池子（>10 万行）自动减少回测切分点：每折都要重训 8 个模型，
            # 300 只池跑满 5 折会让单次分析从 1 分钟涨到 15 分钟以上，不划算。
            n_split = 3 if len(panel) > 100_000 else 5
            bt = walk_forward_backtest_pooled(panel, symbol, senti_value,
                                              horizons=tuple(H), n_splits=n_split)
        else:
            bt = walk_forward_backtest(kline, senti_value, horizons=tuple(H))
    timings["回测"] = round(time.time() - t, 1)

    timings["总计"] = round(time.time() - t_start, 1)
    return {
        "symbol": symbol,
        "name": name or quote.get("name", symbol),
        "kline": kline,
        "tech": tech,
        "quote": quote,
        "flow": flow,
        "last_close": last_close,
        "last_date": str(pd.to_datetime(kline["date"].iloc[-1]).date()),
        "news_groups": groups,
        "news_local": local_senti,
        "llm": llm_res,
        "llm_status": llm_status(),
        "senti_value": senti_value,
        "model_probs": model_probs,
        "model_name": getattr(model, "name", "rf"),
        "model_note": getattr(model, "note", "") or model_fallback,
        "feature_set": getattr(config, "FEATURE_SET", "core"),
        "label_mode": getattr(config, "LABEL_MODE", "ret"),
        "preds": preds,
        "pattern": pat,
        "volume": vol,
        "turn": turn_res,
        "turn_eval": turn_eval,
        "decision": decision,
        "llm_decision": llm_decision,
        "llm_forecast": llm_forecast,
        "trade_points": trade_points,
        "backtest": bt,
        "use_pool": use_pool,
        "n_pool": len(panel["symbol"].unique()) if use_pool else 1,
        "timings": timings,
    }
