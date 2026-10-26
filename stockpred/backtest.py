"""回测：walk-forward 样本外验证，与真实价格对比（DIRMO 直接预测）。

做法：取历史上多个切分点，用此前数据训练多周期模型，
直接预测未来 N 日收益率（不递归），再与真实价格比较，
统计 MAE / MAPE / 方向准确率。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import add_technicals, build_horizon_samples, last_feature_row
from .model import ForecastModel
import config  # 项目根目录配置


def _evaluate_cutoff(kline: pd.DataFrame, sentiment: float, cutoff_idx: int, horizons: list[int]) -> dict:
    train = kline.iloc[: cutoff_idx + 1].reset_index(drop=True)
    if len(train) < 60:
        return {}
    samples = build_horizon_samples(train, sentiment, horizons)
    if not samples:
        return {}
    model = ForecastModel()
    model.fit(samples)
    if not model.fit_ok:
        return {}

    base = float(train["close"].iloc[-1])
    tech = add_technicals(train)
    Xrow = last_feature_row(tech, sentiment)

    comp_rows, maes, mapes, dirs = [], [], [], []
    for h in horizons:
        if h not in model.regs:
            continue
        actual_idx = cutoff_idx + h
        if actual_idx >= len(kline):
            continue
        pred_ret, prob = model.predict_horizon(h, Xrow)
        pred_ret = float(max(-config.PRED_RET_CLIP, min(config.PRED_RET_CLIP, pred_ret)))
        pred_close = base * (1 + pred_ret)
        actual = float(kline["close"].iloc[actual_idx])
        dt = pd.to_datetime(kline["date"].iloc[actual_idx]).date().isoformat()
        conf = prob >= config.CONF_HI or prob <= config.CONF_LO
        comp_rows.append({
            "horizon": h, "date": dt,
            "actual": round(actual, 2), "pred": round(pred_close, 2),
            "abs_err": round(abs(pred_close - actual), 2),
            "ape": round(abs(pred_close - actual) / actual * 100, 2),
            "dir_ok": (pred_close > base) == (actual > base),
            "conf": bool(conf), "prob": round(prob, 3),
        })
        maes.append(abs(pred_close - actual))
        mapes.append(abs(pred_close - actual) / actual * 100)
        dirs.append((pred_close > base) == (actual > base))
    if not maes:
        return {}
    return {
        "mae": float(np.mean(maes)), "mape": float(np.mean(mapes)),
        "dir_acc": float(np.mean(dirs) * 100), "comp": pd.DataFrame(comp_rows),
    }


def walk_forward_backtest(kline: pd.DataFrame, sentiment: float, horizons=(1, 5, 7, 30), n_splits: int = 8) -> dict:
    """对多个历史切分点做回测，汇总各周期指标。

    返回：{
      "summary": DataFrame(horizon, MAE, MAPE(%), 方向准确率(%), 样本数),
      "details": { h: comp DataFrame }   # 各周期最近一次切分明细
    }
    """
    n = len(kline)
    # 多个切分点：越靠前的切分给长周期(30日)留出真实值，样本更稳
    cand = [n - off for off in (160, 130, 100, 80, 60, 40, 20) if n - off >= 60]
    cand = cand[:n_splits]

    agg = {h: {"mae": [], "mape": [], "dir": [], "conf_dir": []} for h in horizons}
    details = {}
    for i, c in enumerate(cand):
        res = _evaluate_cutoff(kline, sentiment, c, list(horizons))
        if not res:
            continue
        comp = res["comp"]
        for _, row in comp.iterrows():
            h = int(row["horizon"])
            if h not in agg:
                continue
            agg[h]["mae"].append(row["abs_err"])
            agg[h]["mape"].append(row["ape"])
            agg[h]["dir"].append(bool(row["dir_ok"]))
            if bool(row["conf"]):  # 仅统计"方向明确"的样本
                agg[h]["conf_dir"].append(bool(row["dir_ok"]))
        # 保留每个周期的本次切分明细（单行：预测点 vs 真实点）
        for h in set(int(x) for x in comp["horizon"].tolist()):
            details[(h, i)] = comp[comp["horizon"] == h].reset_index(drop=True)

    summary_rows = []
    for h in horizons:
        a = agg[h]
        if a["mae"]:
            n_all = len(a["dir"])
            n_conf = len(a["conf_dir"])
            summary_rows.append({
                "horizon(日)": h,
                "MAE": round(float(np.mean(a["mae"])), 2),
                "MAPE(%)": round(float(np.mean(a["mape"])), 2),
                "方向准确率(%)": round(float(np.mean(a["dir"]) * 100), 1),
                "明确时准确率(%)": round(float(np.mean(a["conf_dir"]) * 100), 1) if n_conf else "-",
                "明确占比(%)": round(n_conf / n_all * 100, 1),
                "样本数": n_all,
            })
    return {"summary": pd.DataFrame(summary_rows), "details": details}
