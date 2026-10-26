"""组合策略层：横截面打分选股 + TopK 定期换仓（参考 qlib TopkDropoutStrategy）。

为什么要做成组合层，而不是只看单只股票的"买入/观望"
----------------------------------------------------
单票方向预测的统计优势只有 1~3 个百分点，靠它做单票择时几乎必亏；
量化里真正能落地的是**横截面选股**：不去猜大盘涨跌，只在同一时点比较
"哪几只相对更强"，把资金配给排名靠前的票。qlib 的标准范式就是
「模型打分 → TopK 持仓 → 每期换掉排名最差的几只」。

严谨性约束（缺一条结论就不可信）
------------------------------
1. **信号与成交错开一天**：t 日收盘后才能算出 t 日的特征，所以 t 日信号
   在 t+1 日收盘成交，收益从 t+2 起算（保守，绝不使用当日收盘价成交）。
2. **模型定期重训**：默认每 20 个交易日重训一次，且训练集严格截止到信号日前一日。
3. **交易成本**：买入 0.05%、卖出 0.15%（含印花税），按换手额计费。
4. **涨跌停不可交易**：信号日涨跌停的票不买入（近似处理，无逐笔数据）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from .features import FEATURES
from .models import build_forecaster
from .pool import pooled_samples, pooled_label_scale, zscore_panel, cross_stats, apply_zscore


def walk_forward_scores(panel: pd.DataFrame, sentiment: float = 0.5,
                        horizons: tuple[int, ...] = (1, 5),
                        retrain_every: int = 20, min_train_days: int = 300,
                        model_name: str | None = None) -> pd.DataFrame:
    """滚动出样本外打分表：date × symbol → score（预测收益，越大越看多）。

    训练集严格使用 `date < 当日` 的数据（信号日当天不参与训练）。
    """
    name = (model_name or getattr(config, "MODEL", "rf")).lower()
    if name in ("lstm", "ensemble"):
        # 序列模型逐股逐日预测太慢（每天要为 27 只各切一次窗口），策略层只用表格模型
        name = "gbdt"
    dates = sorted(panel["date"].unique())
    rows = []
    model, last_train = None, -10 ** 9
    for i, d in enumerate(dates):
        if i < min_train_days:
            continue
        if model is None or (i - last_train) >= retrain_every:
            train = panel[panel["date"] < d]
            if train.empty:
                continue
            pz = zscore_panel(train)
            samples = pooled_samples(pz, sentiment, list(horizons))
            model = build_forecaster(name)
            if str(getattr(config, "LABEL_MODE", "ret")).lower() == "csrank":
                model.ret_scale = pooled_label_scale(train, list(horizons))
            try:
                model.fit(samples)
            except Exception:
                continue
            if not model.fit_ok:
                continue
            stats = cross_stats(train)      # 原始截面统计量（不是 pz）
            last_train = i
        today = panel[panel["date"] == d]
        if today.empty or not model.fit_ok:
            continue
        X = apply_zscore(today.copy(), stats)
        X = X.assign(sentiment=sentiment)
        feat = [c for c in FEATURES if c in X.columns]
        X = X[feat].reset_index(drop=True)
        h = 5 if 5 in model.horizons else model.horizons[-1]
        # 整批预测：单行循环在 RF/GBDT 上会被线程池开销拖死（实测 1 万次 ~7 分钟）
        rets, probs = model.predict_batch(h, X)
        for j, (_, r) in enumerate(today.iterrows()):
            rows.append({"date": d, "symbol": r["symbol"], "score": float(rets[j]),
                         "pred_ret": float(rets[j]), "prob_up": float(probs[j]),
                         "close": float(r["close"])})
    return pd.DataFrame(rows)


def _pick_topk(scores: pd.Series, prev: list[str], topk: int, n_drop: int,
               tradable: set[str]) -> list[str]:
    """qlib TopkDropout 的持仓更新：保留已有的，替换掉排名最差的 n_drop 只。"""
    ranked = [s for s in scores.sort_values(ascending=False).index if s in tradable]
    if not prev:
        return ranked[:topk]
    hold = [s for s in prev if s in scores.index]
    pool = [s for s in ranked if s not in hold]
    # 从持仓里按排名从差到好剔除 n_drop 只
    hold_sorted = sorted(hold, key=lambda s: scores.get(s, -1e9), reverse=True)
    keep = hold_sorted[: max(0, len(hold_sorted) - n_drop)]
    target = keep + pool[: max(0, topk - len(keep))]
    return target[:topk]


def score_ic(scores: pd.DataFrame, panel: pd.DataFrame, horizon: int = 5) -> dict:
    """打分的横截面预测力：每日 RankIC（Spearman）与 ICIR。

    这是 qlib 报告里最核心的一列——**IC 不显著为正，任何选股策略都不可能赚钱**，
    先看 IC 再看净值，才不会被某段行情的运气骗到。
    """
    if scores is None or scores.empty:
        return {"ic": float("nan"), "icir": float("nan"), "n": 0, "pos": float("nan")}
    close = panel.pivot_table(index="date", columns="symbol", values="close").sort_index()
    fwd = close.shift(-horizon) / close - 1
    s = scores.pivot_table(index="date", columns="symbol", values="score")
    s = s.reindex(columns=close.columns)
    ics = []
    for d in s.index:
        if d not in fwd.index:
            continue
        a, b = s.loc[d], fwd.loc[d]
        m = a.notna() & b.notna()
        if m.sum() < 5:
            continue
        ics.append(float(a[m].corr(b[m], method="spearman")))
    if not ics:
        return {"ic": float("nan"), "icir": float("nan"), "n": 0, "pos": float("nan")}
    ics = np.asarray(ics)
    return {"ic": float(np.nanmean(ics)),
            "icir": float(np.nanmean(ics) / (np.nanstd(ics) + 1e-12)),
            "n": int(len(ics)),
            "pos": float(np.mean(ics > 0)),
            "horizon": horizon}


def backtest_topk(panel: pd.DataFrame, scores: pd.DataFrame,
                  topk: int | None = None, n_drop: int | None = None,
                  rebalance_days: int | None = None) -> dict:
    """用打分表做 TopK 轮动回测，返回 {metrics, equity, holdings}。"""
    cfg = getattr(config, "STRATEGY", {})
    topk = int(topk or cfg.get("topk", 5))
    n_drop = int(n_drop or cfg.get("n_drop", 1))
    reb = int(rebalance_days or cfg.get("rebalance_days", 5))
    open_cost = float(cfg.get("open_cost", 0.0005))
    close_cost = float(cfg.get("close_cost", 0.0015))
    limit_pct = float(cfg.get("limit_pct", 0.095))

    close = panel.pivot_table(index="date", columns="symbol", values="close").sort_index()
    # fill_method=None：停牌缺口当 NaN 处理；默认 pad 会把缺口当连续持仓收益，虚增净值
    ret = close.pct_change(fill_method=None)
    dates = list(close.index)
    score_wide = scores.pivot_table(index="date", columns="symbol", values="score")
    score_wide = score_wide.reindex(columns=close.columns)

    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    targets: dict = {}          # 仅在换仓执行日写入完整目标权重（含 0），其余日期为 NaN
    prev: list[str] = []
    last_w = pd.Series(0.0, index=close.columns)
    turn_log = []
    hold_log = []

    for i, d in enumerate(dates):
        if i % reb != 0:
            continue
        if d not in score_wide.index:
            continue
        s = score_wide.loc[d].dropna()
        if s.empty:
            continue
        # 涨跌停（信号日）视为不可买入
        day_ret = ret.loc[d] if d in ret.index else pd.Series(dtype=float)
        tradable = set(day_ret[day_ret.abs() < limit_pct].index)
        target = _pick_topk(s, prev, topk, n_drop, tradable)
        if not target:
            continue
        w = pd.Series(0.0, index=close.columns)
        w[target] = 1.0 / len(target)
        # 信号 t → t+1 收盘成交 → 收益从 t+2 起算；权重平移到 i+2
        exec_i = min(i + 2, len(dates) - 1)
        targets[dates[exec_i]] = w
        buy = float((w - last_w).clip(lower=0).sum())
        sell = float((last_w - w).clip(lower=0).sum())
        cost = buy * open_cost + sell * close_cost
        turn_log.append({"date": str(pd.to_datetime(d).date()), "buy": buy, "sell": sell,
                         "cost": cost, "holdings": ",".join(target)})
        hold_log.append(target)
        prev, last_w = target, w

    # 两次换仓之间持仓不变：只有换仓日有"完整目标权重行"，其余为 NaN → ffill 即可
    if targets:
        tdf = pd.DataFrame(targets).T.reindex(columns=close.columns).reindex(close.index)
        weights = tdf.ffill().fillna(0.0)

    port_ret = (weights * ret.fillna(0.0)).sum(axis=1)
    # 交易成本在成交当日扣除（成本按换手额计，单位：组合净值比例）
    cost_series = pd.Series(0.0, index=close.index)
    for t in turn_log:
        d = pd.Timestamp(t["date"])
        if d in cost_series.index:
            cost_series.loc[d] += t["cost"]
    port_ret = port_ret - cost_series

    equity = (1 + port_ret).cumprod()
    bench_w = pd.Series(1.0 / close.shape[1], index=close.columns)
    bench_ret = (ret.fillna(0.0) * bench_w).sum(axis=1)
    bench = (1 + bench_ret).cumprod()

    # 只统计真正开始持仓之后
    first = weights.abs().sum(axis=1).gt(0)
    if first.any():
        start = first.idxmax()
        equity = equity[equity.index >= start] / float(equity[equity.index >= start].iloc[0])
        bench = bench[bench.index >= start] / float(bench[bench.index >= start].iloc[0])
        port_ret = port_ret[port_ret.index >= start]
        bench_ret = bench_ret[bench_ret.index >= start]

    ann = 244
    def _stats(r: pd.Series, eq: pd.Series):
        n = max(len(r), 1)
        total = float(eq.iloc[-1] - 1)
        cagr = (1 + total) ** (ann / n) - 1
        vol = float(r.std() * np.sqrt(ann))
        sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(ann))
        dd = float((eq / eq.cummax() - 1).min())
        win = float((r > 0).mean())
        return {"累计收益": total, "年化收益": cagr, "年化波动": vol,
                "夏普": sharpe, "最大回撤": dd, "日胜率": win}

    ps, bs = _stats(port_ret, equity), _stats(bench_ret, bench)
    n_reb = max(len(turn_log), 1)
    turnover = float(np.mean([t["buy"] + t["sell"] for t in turn_log])) * ann / max(reb, 1)
    metrics = {
        "策略": {"累计收益(%)": round(ps["累计收益"] * 100, 1),
                 "年化收益(%)": round(ps["年化收益"] * 100, 1),
                 "年化波动(%)": round(ps["年化波动"] * 100, 1),
                 "夏普": round(ps["夏普"], 2),
                 "最大回撤(%)": round(ps["最大回撤"] * 100, 1),
                 "日胜率(%)": round(ps["日胜率"] * 100, 1)},
        "等权基准": {"累计收益(%)": round(bs["累计收益"] * 100, 1),
                     "年化收益(%)": round(bs["年化收益"] * 100, 1),
                     "年化波动(%)": round(bs["年化波动"] * 100, 1),
                     "夏普": round(bs["夏普"], 2),
                     "最大回撤(%)": round(bs["最大回撤"] * 100, 1),
                     "日胜率(%)": round(bs["日胜率"] * 100, 1)},
        "超额年化(%)": round((ps["年化收益"] - bs["年化收益"]) * 100, 1),
        "年化换手(倍)": round(turnover, 1),
        "换仓次数": n_reb,
        "持仓数": topk,
        "换仓周期(日)": reb,
    }
    ic = score_ic(scores, panel, horizon=5)
    metrics["RankIC(5日)"] = round(ic["ic"], 4) if ic["n"] else None
    metrics["ICIR"] = round(ic["icir"], 3) if ic["n"] else None
    metrics["IC>0占比(%)"] = round(ic["pos"] * 100, 1) if ic["n"] else None
    return {"metrics": metrics, "equity": pd.DataFrame({"策略": equity, "等权基准": bench}),
            "trades": pd.DataFrame(turn_log), "ic": ic}
