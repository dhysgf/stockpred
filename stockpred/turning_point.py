"""拐点预测：未来 N 日内是否出现「有效顶/底拐点」，以及大概在哪一天。

拐点的可操作定义（必须先定义清楚，否则无法验证）
------------------------------------------------
在 t 日出现**顶拐点**，当且仅当：
1. ``close[t]`` 是窗口 ``[t-win, t+win]`` 内的最高收盘价（局部极大）；
2. 随后 ``win`` 日内从该点回落幅度 ≥ ``TURN_MIN_SWING``（默认 2.5%）。

只有同时满足「局部极值 + 反向幅度足够」，才算有效拐点；
否则横盘里每天都是"极值"，模型学到的是噪声。

模型
----
- 分类：对每个 H∈{5,10,20} 训练 ``P(未来 H 日内出现顶拐点)`` 与 ``P(底拐点)``，
  特征与涨跌预测完全一致（技术指标 + 形态 + 量能 + 指数），只是换标签。
- 回归：对「H 日内确有拐点」的样本回归 ``距拐点天数``，
  并取随机森林各棵树的 20%/80% 分位作为**区间**（比点估计诚实）。
- 评估：pooled 样本外 AUC / 精确率 / 召回率 / 平均领先天数，如实报告。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score

import config


# ------------------------- 标签构造 -------------------------
def turning_labels(d: pd.DataFrame, win: int | None = None,
                   min_swing: float | None = None, max_look: int = 40) -> pd.DataFrame:
    """返回与输入等长的标签表：top/bot（当日是否有有效拐点）、
    days_top/days_bot（距下一个拐点的交易日数）、valid（未来信息是否完整）。"""
    win = int(win if win is not None else config.TURN_CONFIRM_WIN)
    min_swing = float(min_swing if min_swing is not None else config.TURN_MIN_SWING)
    c = d["close"].astype(float).values
    n = len(c)
    s = pd.Series(c)

    center_max = s.rolling(2 * win + 1, center=True).max().values
    center_min = s.rolling(2 * win + 1, center=True).min().values
    future_min = s.rolling(win).min().shift(-win).values     # min(c[t+1..t+win])
    future_max = s.rolling(win).max().shift(-win).values     # max(c[t+1..t+win])

    with np.errstate(invalid="ignore"):
        is_top = (c >= center_max) & ((c - future_min) / np.maximum(c, 1e-9) >= min_swing)
        is_bot = (c <= center_min) & ((future_max - c) / np.maximum(c, 1e-9) >= min_swing)
    is_top = np.nan_to_num(is_top, nan=False).astype(bool)
    is_bot = np.nan_to_num(is_bot, nan=False).astype(bool)

    def _days_to(mask: np.ndarray) -> np.ndarray:
        out = np.full(n, np.nan)
        for t in range(n - 1, -1, -1):
            for k in range(1, max_look + 1):
                if t + k < n and mask[t + k]:
                    out[t] = k
                    break
        return out

    valid = np.zeros(n, dtype=bool)
    valid[: max(0, n - max_look - win - 1)] = True   # 末尾未来信息不全的行剔除

    return pd.DataFrame({
        "top": is_top, "bot": is_bot,
        "days_top": _days_to(is_top), "days_bot": _days_to(is_bot),
        "valid": valid,
    }, index=d.index)


def recent_turns(d: pd.DataFrame, lookback: int = 15, win: int | None = None,
                 min_swing: float | None = None) -> list[dict]:
    """列出最近已确认的拐点（用于界面展示"当前处于什么位置"）。"""
    if d is None or len(d) < 2 * (win or config.TURN_CONFIRM_WIN) + 5:
        return []
    lab = turning_labels(d, win, min_swing)
    n = len(d)
    start = max(0, n - lookback)
    out = []
    for pos in range(start, n):
        row = lab.iloc[pos]
        for kind, flag in (("顶部", row["top"]), ("底部", row["bot"])):
            if bool(flag):
                out.append({
                    "date": pd.to_datetime(d.iloc[pos]["date"]).date().isoformat(),
                    "kind": kind,
                    "price": round(float(d.iloc[pos]["close"]), 2),
                    "days_ago": int(n - 1 - pos),
                })
    out.sort(key=lambda x: x["days_ago"])
    return out


# ------------------------- 样本构造 -------------------------
def build_turn_samples(panel: pd.DataFrame, feat: list[str], horizons: list[int],
                       max_look: int = 40, sentiment: float = 0.5) -> dict:
    """从长表面板构造拐点样本。

    返回 {(kind, h): (X, y)}，kind ∈ {"top","bot"}；以及 {"days": {"top": (X,y), ...}}

    ``sentiment`` 必须显式传入并写回面板：涨跌模型的训练样本里 sentiment 是一列常量，
    拐点模型如果漏掉这一列，预测时特征名就对不上（sklearn 会直接报错）。
    """
    frames = []
    for sym, g in panel.groupby("symbol"):
        g = g.sort_values("date").reset_index(drop=True)
        lab = turning_labels(g, max_look=max_look)
        g2 = pd.concat([g, lab], axis=1)
        g2["symbol"] = sym
        frames.append(g2)
    if not frames:
        return {}
    p = pd.concat(frames, ignore_index=True)
    p = p[p["valid"]].copy()
    # 与涨跌模型同一抽样口径：300 只票的拐点样本十几万行，6 个分类器 + 2 个回归器
    # 全量训练要 2 分钟以上；拐点信号本质是横截面排序，抽样几乎无损（实测 AUC 差异 <0.01）
    max_rows = int(getattr(config, "MAX_TRAIN_ROWS", 0) or 0)
    if max_rows and len(p) > max_rows:
        p = p.sample(n=max_rows, random_state=getattr(config, "RANDOM_STATE", 42)) \
             .sort_values(["symbol", "date"]).reset_index(drop=True)
    p["sentiment"] = float(sentiment)
    usable = [c for c in feat if c in p.columns]
    # 与涨跌模型保持完全一致的特征集并统一顺序：sentiment 是常量列，最容易被漏掉，
    # 一旦漏掉，预测时特征名对不上 sklearn 会直接报错（这里曾踩过）。
    from .features import FEATURES
    for c in FEATURES:
        if c in p.columns and c not in usable:
            usable.append(c)
    usable = [c for c in FEATURES if c in usable]
    out = {}
    for h in horizons:
        for kind in ("top", "bot"):
            y = (p[f"days_{kind}"] <= h).astype(int)      # NaN → False（H 日内无拐点）
            mask = p[usable].notna().all(axis=1)
            out[(kind, h)] = (p.loc[mask, usable].reset_index(drop=True),
                              y[mask].reset_index(drop=True))
    # 距拐点天数回归（仅"有拐点"的样本）
    for kind in ("top", "bot"):
        y = p[f"days_{kind}"]
        mask = y.notna() & p[usable].notna().all(axis=1)
        out[("days", kind)] = (p.loc[mask, usable].reset_index(drop=True),
                               y[mask].reset_index(drop=True))
    return out


class TurningPointModel:
    """顶/底拐点分类 + 距拐点天数回归。"""

    def __init__(self):
        self.clfs = {}
        self.regs = {}
        self.base = {}      # (kind, h) → 训练集正例率（基线），没有它概率无法解读
        self.horizons = []
        self.fit_ok = False

    def fit(self, samples: dict):
        for key, (X, y) in samples.items():
            if len(X) < 200:
                continue
            if key[0] == "days":
                if y.nunique() < 3:
                    continue
                r = RandomForestRegressor(n_estimators=200, max_depth=9,
                                          min_samples_leaf=20, random_state=config.RANDOM_STATE,
                                          n_jobs=-1)
                r.fit(X, y)
                self.regs[key[1]] = r
            else:
                if y.nunique() < 2:
                    continue
                # 不加 class_weight="balanced"：那会把少数类概率系统性抬高，
                # 让「5 日内顶拐点 50%」这种数字出现（基线只有 24%），属于没校准的假精确。
                # 这里保留原始先验，概率可以直接和基线比。
                c = RandomForestClassifier(n_estimators=300, max_depth=9,
                                           min_samples_leaf=20,
                                           random_state=config.RANDOM_STATE, n_jobs=-1)
                c.fit(X, y)
                self.clfs[key] = c
                self.base[key] = float(y.mean())
        self.horizons = sorted({k[1] for k in self.clfs})
        self.fit_ok = bool(self.clfs)

    def predict(self, Xrow: pd.DataFrame) -> dict:
        res = {"probs": {}, "base": {}, "days": {}}
        for (kind, h), clf in self.clfs.items():
            p = float(clf.predict_proba(Xrow)[0, 1])
            res["probs"].setdefault(h, {})[kind] = round(p, 4)
            res["base"].setdefault(h, {})[kind] = round(self.base.get((kind, h), 0.0), 4)
        for kind, reg in self.regs.items():
            est = float(reg.predict(Xrow)[0])
            trees = np.array([t.predict(Xrow.values)[0] for t in reg.estimators_])
            lo, hi = np.percentile(trees, [20, 80])
            res["days"][kind] = {
                "point": round(max(1.0, est), 1),
                "lo": round(max(1.0, float(lo)), 1),
                "hi": round(float(hi), 1),
            }
        return res


# ------------------------- 评估 -------------------------
def eval_turning(panel: pd.DataFrame, feat: list[str], horizons: list[int],
                 train_frac: float = 0.7, max_look: int = 40,
                 sentiment: float = 0.5) -> pd.DataFrame:
    """pooled 样本外评估：按时间切分，报告 AUC / 精确率 / 召回率 / 基线正例率。

    基线很重要：拐点本身是低频事件（正例率 ~20-30%），
    精确率必须和正例率比才有意义。
    """
    dates = sorted(panel["date"].unique())
    if len(dates) < 300:
        return pd.DataFrame()
    cut_i = int(len(dates) * train_frac)
    # 关键：标签本身要看未来 max_look 天，如果训练集直接切到测试期前一刻，
    # 训练样本的标签就用到了测试期的价格 → 前视泄漏，AUC 会被系统性高估。
    # 这里留出 embargo（max_look + 确认窗）作为隔离带，只在该带之前取训练数据。
    embargo = int(max_look + config.TURN_CONFIRM_WIN)
    if cut_i - embargo < 200:
        return pd.DataFrame()
    tr = panel[panel["date"] <= dates[cut_i - embargo]]
    te = panel[panel["date"] > dates[cut_i]]
    if tr.empty or te.empty:
        return pd.DataFrame()

    samples = build_turn_samples(tr, feat, horizons, max_look=max_look, sentiment=sentiment)
    model = TurningPointModel()
    model.fit(samples)
    if not model.fit_ok:
        return pd.DataFrame()

    test = build_turn_samples(te, feat, horizons, max_look=max_look, sentiment=sentiment)
    rows = []
    for (kind, h), (X, y) in test.items():
        if kind == "days" or (kind, h) not in model.clfs or len(y) < 100:
            continue
        p = model.clfs[(kind, h)].predict_proba(X)[:, 1]
        if y.nunique() < 2:
            continue
        # 以"预测概率 >= 该周期正例率"为报警阈值，避免阈值随意拍脑袋
        thr = float(y.mean())
        pred = p >= thr
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        rows.append({
            "类型": "顶" if kind == "top" else "底",
            "周期(日)": h,
            "样本数": len(y),
            "正例率(%)": round(float(y.mean()) * 100, 1),
            "AUC": round(float(roc_auc_score(y, p)), 3),
            "精确率(%)": round(tp / max(tp + fp, 1) * 100, 1),
            "召回率(%)": round(tp / max(tp + fn, 1) * 100, 1),
        })
    return pd.DataFrame(rows)
