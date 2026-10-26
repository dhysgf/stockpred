"""预测模型：直接多周期（DIRMO）回归 + 方向分类 + 图表路径调和。

- 每个周期 h 用独立 RandomForestRegressor 直接预测「h 日收益率」，
  用独立 Classifier 预测「h 日上涨概率」。不再递归复利，避免误差累积。
- 图表用的每日路径：先递归生成 1 日路径，再按各周期 DIRMO 端点做
  比例调和，使路径终点与 DIRMO 直接预测一致（兼顾每日粒度与低误差）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .features import add_technicals, last_feature_row, apply_zscore_cols
import config  # 项目根目录配置（运行 run.py / web.py 时已加入 sys.path）


def _next_trading_dates(last_date: pd.Timestamp, steps: int) -> list[pd.Timestamp]:
    dates, d = [], last_date
    while len(dates) < steps:
        d = d + pd.Timedelta(days=1)
        if d.weekday() < 5:
            dates.append(d)
    return dates


class ForecastModel:
    def __init__(self, random_state: int = config.RANDOM_STATE):
        self.regs = {}      # h -> 收益率回归
        self.clfs = {}      # h -> 方向分类
        self.horizons = []
        self.fit_ok = False

    def fit(self, samples: dict):
        """samples: {h: (X, y_dir, y_ret)}。"""
        for h, (X, y_dir, y_ret) in samples.items():
            if len(X) < 20:
                continue
            self.regs[h] = RandomForestRegressor(
                n_estimators=250, max_depth=10, min_samples_leaf=5,
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )
            self.clfs[h] = RandomForestClassifier(
                n_estimators=250, max_depth=10, min_samples_leaf=5,
                class_weight="balanced", random_state=config.RANDOM_STATE, n_jobs=-1,
            )
            self.regs[h].fit(X, y_ret)
            self.clfs[h].fit(X, y_dir)
        self.horizons = sorted(self.regs.keys())
        self.fit_ok = bool(self.horizons)

    def predict_horizon(self, h: int, X_row: pd.DataFrame):
        """返回 (pred_ret, prob_up)。"""
        r = float(self.regs[h].predict(X_row)[0])
        p = float(self.clfs[h].predict_proba(X_row)[0, 1])
        return r, p


def forecast(model, kline: pd.DataFrame, sentiment: float, horizons: list[int],
             zstats: dict | None = None, ctx_panel: pd.DataFrame | None = None,
             ctx_symbol: str | None = None) -> dict:
    """生成各周期预测（含每日调和路径）。

    zstats：横截面标准化统计量（多股票池训练时传入），会把特征换算到训练时的尺度。
    ctx_panel/ctx_symbol：序列模型（LSTM）预测时需要目标股的历史面板，用来切最后一个窗口。
    返回 {h: DataFrame[date, pred_close, prob_up, pred_dir]}。
    """
    if not model.fit_ok:
        raise RuntimeError("模型尚未训练，请先 fit。")
    max_h = max(horizons)
    base = float(kline["close"].iloc[-1])
    last_date = pd.to_datetime(kline["date"].iloc[-1])
    dates = _next_trading_dates(last_date, max_h)
    ctx = {"panel": ctx_panel, "symbol": ctx_symbol} if ctx_panel is not None else None

    # 1) 递归生成 1 日路径（仅用于每日粒度可视化）
    #    指数因子无法预知未来，按「指数走平」沿用最新值（DIRMO 端点本身用真实当前指数值）
    #    序列模型（LSTM）无法对"合成出来的未来K线"递归预测，直接退回线性插值路径。
    idx_cols = [c for c in ("idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev") if c in kline.columns]
    keep = ["date", "open", "high", "low", "close", "volume"] + idx_cols
    work = kline[keep].copy()
    last_close = base
    rec = []
    # 递归合成 1 日路径是否可用：序列模型（LSTM）不能对"合成出来的未来K线"递归，
    # 融合模型则可以用它的非序列成员（rf/gbdt）递归 —— 故用 recurse_needs_seq 判定。
    recurse_seq = getattr(model, "recurse_needs_seq", None)
    if recurse_seq is None:
        recurse_seq = getattr(model, "needs_seq", False)
    if recurse_seq:
        rec = pd.Series([base] * max_h)
    else:
        for _ in range(max_h):
            tech = add_technicals(work)
            Xrow = apply_zscore_cols(last_feature_row(tech, sentiment), zstats)
            # 递归用非序列成员（融合模型里 LSTM 无法对合成K线预测）
            fn = getattr(model, "predict_horizon_no_seq", None)
            if callable(fn):
                r, _ = fn(model.horizons[0], Xrow)
            else:
                r, _ = model.predict_horizon(model.horizons[0], Xrow)
            r = float(max(-config.PRED_RET_CLIP, min(config.PRED_RET_CLIP, r)))
            last_close = last_close * (1 + r)
            rec.append(last_close)
            # 合成 K 线给一个极小的振幅（不要 open=high=low=close 全相等）：
            # 一字线会让振幅类因子（KMID2/RSV/ATR）退化，递归路径的自变量失真
            bar = {
                "date": dates[len(rec) - 1], "open": last_close * 0.999,
                "high": last_close * 1.003, "low": last_close * 0.997,
                "close": last_close, "volume": work["volume"].iloc[-1],
            }
            for c in idx_cols:
                bar[c] = work[c].iloc[-1]
            work = pd.concat([work, pd.DataFrame([bar])], ignore_index=True)
        rec = pd.Series(rec)  # index 0..max_h-1 对应第 1..max_h 日

    # 2) DIRMO 直接端点 + 按端点比例调和各周期路径
    tech_last = add_technicals(kline)
    Xrow = apply_zscore_cols(last_feature_row(tech_last, sentiment), zstats)

    # 区间带宽的两个来源（取更宽者）：
    #   a) 多模型分歧：ensemble 各成员对同一 h 的收益预测的标准差（认知不确定性）
    #   b) 波动率外推：最近 N 日真实日波动 × sqrt(h) × z（偶然不确定性）
    # z=1.28 约对应 80% 覆盖。band 记在每个 h 的 DataFrame 里（该 h 的终点带宽，
    # 路径各行同值），供 trade_points 生成目标价位区间。
    tr = getattr(config, "TRADE", {}) or {}
    z = float(tr.get("band_z", 1.28))
    vwin = int(tr.get("vol_window", 20))
    daily_vol = 0.0
    if len(kline) > vwin + 2:
        _c = kline["close"].astype(float).tail(vwin + 1)
        daily_vol = float(_c.pct_change().std() or 0.0)

    preds = {}
    for h in horizons:
        r, prob = model.predict_horizon(h, Xrow, ctx)
        # 新闻/政策情绪微调（透明、有界）：看涨上修、看跌下修。
        # 说明：情绪是"当前"状态量（无历史新闻序列），故不随周期复利，只做一次性调整。
        r = r + config.SENTIMENT_TILT * (sentiment - 0.5)
        r = float(max(-config.PRED_RET_CLIP, min(config.PRED_RET_CLIP, r)))
        dir_close = base * (1 + r)

        # 多模型分歧
        spread = 0.0
        pm = getattr(model, "predict_members", None)
        if callable(pm):
            try:
                mem = pm(h, Xrow, ctx) or []
            except Exception:
                mem = []
            if len(mem) > 1:
                rs_m = np.asarray([m[1] for m in mem], dtype=float)
                spread = float(np.std(rs_m))
        band = max(z * spread, z * daily_vol * math.sqrt(max(h, 1)))
        sub = np.asarray(rec.iloc[:h], dtype=float)
        # 对数空间融合：保留递归路径的日间形状，同时让终点严格等于 DIRMO 端点、
        # 起点严格等于最新收盘。这样各周期路径互相嵌套，不会出现
        # 「30 日路径第一天就高于 7 日路径终点」的割裂（等比缩放的旧做法就有这个问题）。
        eps = 1e-9
        t = np.arange(1, h + 1, dtype=float)
        log_rel = np.log(np.maximum(sub, eps) / base)
        shape = log_rel - (t / h) * log_rel[-1]
        target_log = (t / h) * np.log(max(dir_close, eps) / base)
        recon = base * np.exp(target_log + shape)
        # 置信度门控：概率不够极端时标"方向不明"，避免假确定性
        if h in getattr(config, "NO_SKILL_HORIZONS", ()):
            # 大样本评估确认该周期无方向预测力，一律不给方向
            pred_dir = "方向不明"
        elif prob >= config.CONF_HI:
            pred_dir = "涨"
        elif prob <= config.CONF_LO:
            pred_dir = "跌"
        else:
            pred_dir = "方向不明"
        preds[h] = pd.DataFrame({
            "date": [d.date().isoformat() for d in dates[:h]],
            "pred_close": np.round(recon, 2),
            "prob_up": round(prob, 3),
            "pred_dir": pred_dir,
            # 该 h 终点的不确定性带宽（相对收益率，如 0.03 表示 ±3%），路径各行同值
            "band_ret": round(float(band), 6),
            # 多模型成员分歧（相对收益率），单模型时为 0
            "member_spread": round(float(spread), 6),
        })
    return preds
