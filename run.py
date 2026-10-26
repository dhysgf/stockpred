"""命令行入口：股价走势预测 + 组合策略回测。

用法示例：
    python run.py                                      # 默认贵州茅台，全流程
    python run.py --symbol 000001 --name 平安银行
    python run.py --symbol 600519 --mode predict        # 只预测
    python run.py --model gbdt                          # 换模型（rf/gbdt/lstm/ensemble）
    python run.py --features alpha                       # 用 qlib Alpha158 精简版因子
    python run.py --strategy --topk 5                    # 跑 TopK 轮动策略回测
    python run.py --signal-eval                          # 验证各信号/拐点的预测力
    python run.py --model-eval                           # 对比 rf/gbdt/lstm 的样本外误差
"""
from __future__ import annotations

import argparse
import os
import sys

# 注意：stockpred 的模块在 import 时就按 config 快照了特征列表（FEATURES），
# 所以命令行覆盖 config 必须发生在**导入 stockpred 之前** —— 这就是这里
# 全用函数内延迟导入的原因，不要在文件顶部 import stockpred。

def _p(msg: str = ""):
    print(msg)


def _print_table(df, title: str, max_rows: int = 100):
    import pandas as pd
    print(f"\n=== {title} ===")
    if df is None or len(df) == 0:
        print("  (无数据)")
        return
    with pd.option_context("display.max_rows", max_rows, "display.width", 130,
                           "display.max_colwidth", 40):
        print(df.to_string(index=False))


def _resolve_name(symbol: str, default_symbol: str, default_name: str) -> str:
    """没给 --name 时自动补全股票简称，避免报告标题变成「000001（000001）」。

    实时行情接口（hq.sinajs.cn）的第一列就是简称，抓不到才退化成空串。
    """
    if symbol == default_symbol and default_name:
        return default_name
    try:
        from stockpred.data_fetcher import fetch_quote
        nm = (fetch_quote(symbol) or {}).get("name") or ""
        nm = nm.strip().replace(" ", "")
        if nm and nm.isdigit():   # 接口异常时会回吐代码，视为无效
            nm = ""
        return nm
    except Exception:
        return ""


# =====================================================================
# 单股分析
# =====================================================================
def run_analysis(symbol: str, name: str, mode: str, horizons: list[int],
                 use_llm: bool, with_turn_eval: bool):
    import config
    from stockpred import analyze, plot_forecast, plot_candles, detect_patterns, plot_backtest

    print(f"开始分析 {name or symbol}({symbol})：模型={config.MODEL} 特征={config.FEATURE_SET} "
          f"标签={config.LABEL_MODE}")
    res = analyze(symbol, name=name, horizons=horizons,
                  do_backtest=(mode in ("backtest", "all")),
                  use_llm=use_llm, verbose=True, with_turn_eval=with_turn_eval)

    llm = res["llm"]
    print(f"      历史 {len(res['kline'])} 根日线，最新收盘 {res['last_close']:.2f}（{res['last_date']}）")
    q = res["quote"]
    if q:
        print(f"      实时价 {q.get('price')}  委比 {q.get('weibi')}%")
    print(f"      训练方式：{'多股票池横截面（%d 只）' % res['n_pool'] if res['use_pool'] else '单股时序'}"
          f" ｜ 模型 {res['model_name']}"
          + (f"（{res['model_note']}）" if res.get("model_note") else ""))
    print(f"      耗时：{res['timings']}")

    # 技术面
    pat = res["pattern"]
    print(f"\n[技术面] K线形态：{pat['label']}（形态分 {pat['score']:.2f}，"
          f"多 {pat['bull']:.2f} / 空 {pat['bear']:.2f}）")
    for it in (pat["items"] or [])[:6]:
        flag = "利多" if it["direction"] > 0 else ("利空" if it["direction"] < 0 else "中性")
        print(f"    {it['date']}  {it['name']:<10} {flag}  强度 {it['strength']:.2f}  {it['cn']}")
    if not pat["items"]:
        print("    最近 3 根 K 线未命中经典形态")

    vol = res["volume"]
    print(f"\n[量能] {vol['label']}（量能分 {vol['score']:.2f}）")
    for it in vol["items"]:
        print(f"    {it['name']:<14} {it['value']:<28} 贡献 {it['contrib']:+.3f}  {it['note']}")
    flow = res.get("flow") or {}
    if flow.get("ok"):
        print("    资金流分单：" + "  ".join(
            f"{k} {v['net']/1e4:+.0f}万" for k, v in flow["tiers"].items()))

    # 新闻政策
    g = res["news_groups"]
    print(f"\n[新闻政策] 个股 {len(g.get('stock', []))} / 政策 {len(g.get('policy', []))} / "
          f"市场 {len(g.get('market', []))} 条")
    if llm.get("ok"):
        print(f"    大模型 {llm['provider']}/{llm['model']}（{llm.get('elapsed', 0)}s，"
              f"置信 {llm['confidence']:.2f}）立场 {llm['stance']} 情绪 {llm['sentiment']:.3f}")
        print(f"    总结：{llm['summary']}")
        for d in llm.get("key_drivers", []):
            print(f"      驱动｜{d}")
        for r in llm.get("risks", []):
            print(f"      风险｜{r}")
    else:
        print(f"    [降级] {llm.get('note', '')}")
        print(f"    本地情绪：{res['news_local']['label']}（{res['news_local']['mean']:.3f}）")

    # 每档只出一行：预测的是「第 N 个交易日收盘价」，不是逐日预测。
    # （preds[h] 里除最后一行外都是为了让图好看的插值，不能当逐日预测展示。）
    import pandas as pd

    _base = float(res.get("last_close") or 0)
    _rows = []
    for h in sorted(res["preds"]):
        df = res["preds"][h]
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        tgt = float(last["pred_close"])
        band = float(last["band_ret"]) if "band_ret" in df.columns else 0.0
        _rows.append({
            "周期(日)": h,
            "预测日": last["date"],
            "预测收盘": round(tgt, 2),
            "相对现价(%)": round((tgt / _base - 1) * 100, 2) if _base else "-",
            "涨概率": round(float(last["prob_up"]), 3),
            "方向": last["pred_dir"],
            "区间下界": round(_base * (1 + (tgt / _base - 1) - band), 2) if _base else "-",
            "区间上界": round(_base * (1 + (tgt / _base - 1) + band), 2) if _base else "-",
        })
    if _rows:
        _print_table(pd.DataFrame(_rows),
                     "各周期目标价（每档只预测第 N 个交易日，不是逐日预测）")

    tr = res.get("turn") or {}
    if tr.get("probs"):
        print("\n[拐点预测] 概率（括号内为基线）")
        for h in sorted(tr["probs"]):
            p, b = tr["probs"][h], (tr.get("base") or {}).get(h, {})
            print(f"    {h:>2}日：顶 {p.get('top',0)*100:4.1f}%（{b.get('top',0)*100:4.1f}%） "
                  f"底 {p.get('bot',0)*100:4.1f}%（{b.get('bot',0)*100:4.1f}%）")
        for kind, cn in (("top", "顶"), ("bot", "底")):
            d = (tr.get("days") or {}).get(kind)
            if d:
                print(f"    预计下一个{cn}拐点：约 {d['point']:.0f} 个交易日后"
                      f"（区间 {d['lo']:.0f}~{d['hi']:.0f} 日）")

    dec = res["decision"]
    print(f"\n[联合决策] 综合分 {dec['score']:+.3f} → {dec['advice']}"
          f"（置信 {dec['confidence']:.2f}，一致度 {dec['agreement']:.0%}）")
    print(f"    {dec['action']}")
    import pandas as pd
    rows = pd.DataFrame([{"信号": s["name"], "取值": s["value"], "权重": s["weight"],
                          "贡献": s["contrib"], "依据": s["note"]} for s in dec["signals"]])
    _print_table(rows, "各信号贡献明细")

    ld = res.get("llm_decision") or {}
    if ld.get("ok"):
        print(f"\n[大模型综合研判] {ld['provider']}/{ld['model']}（{ld.get('elapsed')}s）"
              f" 口径：{ld['final_stance']}")
        print(f"    {ld['conclusion']}")
        for x in ld.get("agrees", []):
            print(f"      支撑｜{x}")
        for x in ld.get("diverges", []):
            print(f"      警惕｜{x}")
        for x in ld.get("watch", []):
            print(f"      盯盘｜{x}")
        print("    注意：这是对量化决策的解释层，不增加信号权重、不参与加权计算。")
    else:
        print(f"\n[大模型综合研判] 未生成：{ld.get('note', '未开启大模型')}")

    # ---- 大模型独立价格预测 + 条件规则预判（第二意见，不参与加权）
    lf = res.get("llm_forecast") or {}
    if lf.get("ok"):
        print(f"\n[大模型独立预测] {lf['provider']}/{lf['model']}（{lf.get('elapsed')}s）"
              f"  方向 {lf['direction']}（置信 {lf['confidence']:.2f}）")
        for x in lf.get("forecasts", []):
            pr = f"{x['price']:.2f}" if x.get("price") is not None else "—"
            print(f"    预测 {x['horizon']}日 {x['pct']:+.2f}% 目标 {pr} "
                  f"收涨概率 {x['prob_up']:.0%} | {x.get('reason','')}")
        for x in lf.get("rules", []):
            print(f"    规则 若 {x['trigger']} → {x['action']}（{x.get('why','')}）")
        for x in lf.get("key_reasons", []):
            print(f"    理由 {x}")
        for x in lf.get("risks", []):
            print(f"    风险 {x}")
        print(f"    总结 {lf.get('summary','')}")
        print("    注意 这是大模型的独立第二意见，与量化模型对照看，不参与联合决策加权"
              "（否则等于让大模型给自己打分）。")
    else:
        print(f"\n[大模型独立预测] 未生成：{lf.get('note', '未开启大模型')}")

    # ---- 买卖点预判 + 目标价位区间
    tp = res.get("trade_points") or {}
    if tp.get("per_horizon"):
        _hs = "/".join(str(r["horizon"]) for r in tp["per_horizon"])
        print(f"\n[目标价位区间] 每 {_hs} 个交易日各出一档结果"
              f"（基准现价 {tp['base_price']:.2f}）")
        import pandas as _pd
        _rows = _pd.DataFrame([
            {"周期(日)": r["horizon"], "到达日": r["date"],
             "下界": r["lower"], "中位": r["target"], "上界": r["upper"],
             "预期(%)": r["expected_ret"], "带宽(±%)": r["band_pct"],
             "涨概率": r["prob_up"], "方向": r["direction"]}
            for r in tp["per_horizon"]])
        _print_table(_rows, "目标价位区间")

        bp, sp = tp.get("buy_points", {}), tp.get("sell_points", {})
        print(f"\n[买卖点预判]")
        if "pullback_zone" in bp:
            print(f"    分批建仓区（低吸）：{bp['pullback_zone'][0]:.2f} ~ {bp['pullback_zone'][1]:.2f}")
            print(f"        └ {bp['pullback_note']}")
            print(f"    突破确认价（追强）：{bp['breakout_trigger']:.2f}")
            print(f"        └ {bp['breakout_note']}")
            print(f"    短期强支撑：{bp['short_term_support']:.2f}")
            if bp.get("ma20_price"):
                role = bp.get("ma20_role") or "参考位"
                extra = "（现价在 MA20 下方，该位为上方压力，非支撑）" if role == "阻力" else ""
                print(f"    MA20 动态{role}：{bp['ma20_price']:.2f}{extra}")
        if "take_profit_zone" in sp:
            print(f"    止盈减仓区：{sp['take_profit_zone'][0]:.2f} ~ {sp['take_profit_zone'][1]:.2f}")
            print(f"        └ {sp['take_profit_note']}")
            print(f"    止损位：{sp['stop_loss']:.2f}（{sp['stop_loss_pct']:+.2f}%）")
            print(f"        └ {sp['stop_loss_note']}")
            if sp.get("atr_stop"):
                print(f"    ATR 参考止损：{sp['atr_stop']:.2f}（1.5×ATR）")
        rk = tp.get("risk", {})
        print(f"    风险等级 {rk.get('risk_level')}   ATR {rk.get('atr_pct')}%   "
              f"日波动 {rk.get('daily_vol_pct')}%   带宽系数 z={rk.get('band_z')}")
        for n in tp.get("notes", []):
            print(f"      · {n}")

    if mode in ("backtest", "all"):
        bt = res.get("backtest") or {}
        if bt.get("note"):
            print(f"\n[回测] {bt['note']}")
        elif bt.get("summary") is not None and len(bt["summary"]):
            _print_table(bt["summary"], "回测误差指标（样本外）")
        if res.get("turn_eval") is not None and len(res.get("turn_eval")):
            _print_table(res["turn_eval"], "拐点模型样本外评估")

    import config as _cfg
    os.makedirs(_cfg.OUTPUT_DIR, exist_ok=True)
    if mode in ("predict", "all"):
        out_png = os.path.join(_cfg.OUTPUT_DIR, f"forecast_{symbol}.png")
        plot_forecast(res["kline"], res["preds"], out_png, name=res["name"],
                      max_horizon=max(horizons), turn=res.get("turn"))
        out_cd = os.path.join(_cfg.OUTPUT_DIR, f"candles_{symbol}.png")
        plot_candles(res["kline"], out_cd, name=res["name"], bars=60,
                     patterns=detect_patterns(res["tech"], window=12))
        print(f"\n      ✓ 预测图：{out_png}\n      ✓ K线形态图：{out_cd}")
    if mode in ("backtest", "all") and (res.get("backtest") or {}).get("details"):
        for (h, i), comp in res["backtest"]["details"].items():
            plot_backtest(comp, os.path.join(_cfg.OUTPUT_DIR, f"backtest_{symbol}_h{h}_s{i}.png"),
                          name=res["name"], horizon=h)
        print(f"      ✓ 回测图已保存至 {_cfg.OUTPUT_DIR}/")
    print("\n完成。结果仅用于研究，不构成投资建议。")


# =====================================================================
# 组合策略回测
# =====================================================================
def run_strategy(symbol: str, topk: int | None, n_drop: int | None,
                 rebalance: int | None, retrain_every: int = 20):
    import config
    from stockpred.data_fetcher import load
    from stockpred.features import enrich_with_index
    from stockpred.pool import fetch_pool, build_panel
    from stockpred.strategy import walk_forward_scores, backtest_topk
    from stockpred.visualize import plot_equity
    from stockpred.decision import eval_signals

    print(f"[1/3] 抓取股票池（模型={config.MODEL}，特征={config.FEATURE_SET}）...")
    idx = load(config.INDEX_SYMBOL, datalen=config.KLINE_LEN).kline
    pl = fetch_pool(datalen=config.KLINE_LEN, extra=[symbol])
    panel = build_panel(pl, idx)
    print(f"      股票池 {len(pl)} 只，面板 {len(panel)} 行，"
          f"区间 {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    print(f"[2/3] 滚动训练并打分（每 {retrain_every} 个交易日重训）...")
    scores = walk_forward_scores(panel, sentiment=0.5, retrain_every=retrain_every)
    if scores.empty:
        print("      ✗ 打分失败（训练样本不足）")
        return
    print(f"      打分 {len(scores)} 条，覆盖 {scores['date'].nunique()} 个交易日")

    print("[3/3] TopK 轮动回测 ...")
    bt = backtest_topk(panel, scores, topk=topk, n_drop=n_drop, rebalance_days=rebalance)
    m = bt["metrics"]
    import pandas as pd
    rows = pd.DataFrame([{"指标": k, "策略": m["策略"][k], "等权基准": m["等权基准"][k]}
                         for k in m["策略"]])
    _print_table(rows, "TopK 轮动策略 · 样本外表现")
    print(f"\n  超额年化 {m['超额年化(%)']:+.1f}% ｜ 年化换手 {m['年化换手(倍)']} 倍 ｜ "
          f"换仓 {m['换仓次数']} 次 ｜ 持仓 {m['持仓数']} 只 / 每 {m['换仓周期(日)']} 日调仓")
    print(f"  打分质量：RankIC(5日) {m.get('RankIC(5日)')} ｜ ICIR {m.get('ICIR')} ｜ "
          f"IC>0 占比 {m.get('IC>0占比(%)')}%（IC 不显著为正，策略不可能持续赚钱）")
    if not bt["trades"].empty:
        _print_table(bt["trades"].tail(5)[["date", "holdings", "cost"]], "最近 5 次换仓")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out = os.path.join(config.OUTPUT_DIR, "strategy_equity.png")
    csv = os.path.join(config.OUTPUT_DIR, "strategy_equity.csv")
    bt["equity"].to_csv(csv, encoding="utf-8-sig")
    flat = dict(m["策略"])
    flat["超额年化(%)"] = m["超额年化(%)"]
    plot_equity(bt["equity"], out, title=f"TopK({m['持仓数']}) 轮动策略 vs 等权基准",
                metrics=flat)
    print(f"      ✓ 净值图：{out}（数据：{csv}）")
    print("\n注意：策略收益已扣除双边佣金（买 0.05%/卖 0.15%）与涨跌停不可交易约束；"
          "信号 t 日生成、t+1 日收盘成交、收益自 t+2 起算，不存在前视。")


# =====================================================================
# 评估：模型对比 / 信号检验
# =====================================================================
def run_model_eval(symbol: str):
    """在同一股票池、同一 walk-forward 切分下对比 rf/gbdt/lstm/ensemble。"""
    import config
    import pandas as pd
    from stockpred.data_fetcher import load
    from stockpred.pool import fetch_pool, build_panel, walk_forward_backtest_pooled

    print("[1/2] 抓取股票池 ...")
    idx = load(config.INDEX_SYMBOL, datalen=config.KLINE_LEN).kline
    pl = fetch_pool(datalen=config.KLINE_LEN, extra=[symbol])
    panel = build_panel(pl, idx)
    print(f"      面板 {len(panel)} 行")
    print("[2/2] 逐模型跑同一套 walk-forward ...")
    out = []
    for name in ("rf", "gbdt", "lstm", "ensemble"):
        config.MODEL = name
        print(f"    → {name} ...")
        try:
            bt = walk_forward_backtest_pooled(panel, symbol, 0.5)
            s = bt["summary"]
            if s is None or s.empty:
                out.append({"模型": name, "1日MAPE": "-", "5日MAPE": "-", "5日方向": "-",
                            "7日MAPE": "-", "30日MAPE": "-", "备注": "无结果"})
                continue
            row = {"模型": name}
            for h, col in ((1, "1日"), (5, "5日"), (7, "7日"), (30, "30日")):
                sub = s[s["horizon(日)"] == h]
                row[f"{col}MAPE"] = round(float(sub["MAPE(%)"].iloc[0]), 2) if len(sub) else "-"
                if h == 5:
                    row["5日方向"] = round(float(sub["方向准确率(%)"].iloc[0]), 1) if len(sub) else "-"
            row["备注"] = ""
            out.append(row)
        except Exception as e:
            out.append({"模型": name, "1日MAPE": "-", "5日MAPE": "-", "5日方向": "-",
                        "7日MAPE": "-", "30日MAPE": "-", "备注": f"{type(e).__name__}: {e}"[:60]})
    _print_table(pd.DataFrame(out), "模型对比（样本外，同一批切分点；样本数=5）")
    print("\n  说明：切分点只有 5 个，指标噪声很大；要看模型优劣请结合 --signal-eval 的大样本检验。")


def run_signal_eval(symbol: str):
    import config
    from stockpred.data_fetcher import load
    from stockpred.pool import fetch_pool, build_panel
    from stockpred.decision import eval_signals
    from stockpred.turning_point import eval_turning
    from stockpred.features import FEATURES

    print("[1/3] 抓取股票池 ...")
    idx = load(config.INDEX_SYMBOL, datalen=config.KLINE_LEN).kline
    pl = fetch_pool(datalen=config.KLINE_LEN, extra=[symbol])
    panel = build_panel(pl, idx)
    print(f"      股票池 {len(pl)} 只，面板 {len(panel)} 行，特征 {len(FEATURES)} 个")
    print("[2/3] 形态/量能信号方向检验 ...")
    df = eval_signals(panel)
    _print_table(df, "形态/量能信号的历史方向检验")
    print("\n  说明：明确表态占比 = |信号|>0.10 的样本比例；命中率 50% 即无预测力。")
    print("[3/3] 拐点模型样本外评估 ...")
    tdf = eval_turning(panel, [c for c in FEATURES if c in panel.columns], config.TURN_HORIZONS)
    _print_table(tdf, "拐点模型样本外评估")
    print("\n  说明：拐点是低频事件，精确率要跟正例率比；AUC 0.5 即无预测力。")


def run_llm_check():
    """检查大模型密钥是否生效：只说一句极短的话，几乎不消耗额度。"""
    import config
    from stockpred.llm import status, chat

    st = status()
    print("大模型状态：")
    for k in ("enabled", "ready", "providers", "model"):
        print(f"  {k:<10} {st[k]}")
    if st["hint"]:
        print(f"  hint       {st['hint']}")
    if not st["ready"]:
        print("\n未检测到密钥。获取方式（千问免费额度，2026-10）：")
        print("  1. 打开 https://bailian.console.aliyun.com （用阿里云/支付宝账号登录）")
        print("  2. 右上角地域切到『华北2(北京)』—— 免费额度只在该地域生效，选错会 403")
        print("  3. 开通「大模型服务平台百炼」→ 完成实名认证（支付宝刷脸，约 10 秒）")
        print("  4. 控制台首页领「新手免费额度」（约 7000 万 tokens / 90 天，未自动发放需手动点）")
        print("  5. 「API-KEY 管理」→ 创建 → 立即复制（明文只显示一次，格式 sk-xxx）")
        print("  6. 配置：set DASHSCOPE_API_KEY=sk-xxx   或写入项目根 secrets.json")
        print("  7. 重新运行：python run.py --llm-check")
        return 1

    print("\n试调一次（max_tokens=16）...")
    r = chat("只回复两个字：可用", max_tokens=16, temperature=0.0)
    if r["ok"]:
        print(f"  ✓ {r['provider']}/{r['model']} 返回：{r['text'][:40]}（{r.get('elapsed')}s）")
        return 0
    print(f"  ✗ 调用失败：{r['error']}")
    print("  常见原因：密钥复制不全 / 地域选错(需华北2北京) / 未开通百炼 / 额度已用尽")
    return 1


def main():
    ap = argparse.ArgumentParser(description="股价预测 + 组合策略（rf/gbdt/lstm/ensemble）")
    ap.add_argument("--symbol", default=None, help="股票代码，如 600519")
    ap.add_argument("--name", default=None, help="股票名称")
    ap.add_argument("--mode", default="all", choices=["predict", "backtest", "all"])
    ap.add_argument("--horizons", default="1,5,7,30")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--llm-check", action="store_true", help="检查大模型密钥是否可用")
    ap.add_argument("--turn-eval", action="store_true")
    ap.add_argument("--model", default=None, choices=["rf", "gbdt", "lstm", "ensemble"],
                    help="模型类型")
    ap.add_argument("--features", default=None, choices=["core", "alpha"],
                    help="特征集：core=手工精选，alpha=+qlib Alpha158 精简版")
    ap.add_argument("--label", default=None, choices=["ret", "csrank"],
                    help="回归标签：ret=收益率，csrank=截面排名（qlib CSRankNorm）")
    ap.add_argument("--strategy", action="store_true", help="跑 TopK 轮动组合策略回测")
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--n-drop", type=int, default=None)
    ap.add_argument("--rebalance", type=int, default=None, help="换仓周期（交易日）")
    ap.add_argument("--retrain", type=int, default=20, help="策略打分时模型重训间隔（交易日）")
    ap.add_argument("--signal-eval", action="store_true", help="信号有效性检验")
    ap.add_argument("--model-eval", action="store_true", help="对比 rf/gbdt/lstm/ensemble")
    ap.add_argument("--pool", default=None, choices=["default", "hs300", "zz500"],
                    help="股票池：default=27只龙头 / hs300 / zz500")
    ap.add_argument("--pool-limit", type=int, default=None, help="池子只用前 N 只（0=不限）")
    ap.add_argument("--refresh-cache", action="store_true", help="忽略本地缓存，重新抓行情")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="单次训练最大样本行数（0=不限，300 只票建议 60000~80000）")
    args = ap.parse_args()

    # 先改 config，再导入 stockpred（FEATURES 在 import 时快照）
    import config
    if args.pool:
        config.POOL_MODE = args.pool
    if args.pool_limit is not None:
        config.POOL_LIMIT = args.pool_limit
    if args.max_train_rows is not None:
        config.MAX_TRAIN_ROWS = args.max_train_rows
    if args.refresh_cache:
        config.KLINE_CACHE_TTL = 0
    if args.model:
        config.MODEL = args.model
    if args.features:
        config.FEATURE_SET = args.features
    if args.label:
        config.LABEL_MODE = args.label

    symbol = args.symbol or config.DEFAULT_SYMBOL
    name = args.name or _resolve_name(symbol, config.DEFAULT_SYMBOL, config.DEFAULT_NAME)
    try:
        if args.llm_check:
            return run_llm_check()
        if args.model_eval:
            run_model_eval(symbol)
        elif args.signal_eval:
            run_signal_eval(symbol)
        elif args.strategy:
            run_strategy(symbol, args.topk, args.n_drop, args.rebalance, args.retrain)
        else:
            horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
            run_analysis(symbol, name, args.mode, horizons, not args.no_llm, args.turn_eval)
    except Exception as e:
        print(f"\n✗ 执行失败：{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
