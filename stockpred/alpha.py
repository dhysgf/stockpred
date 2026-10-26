"""Alpha158 精简版因子（参考 Microsoft qlib 的 Alpha158DL 定义）。

为什么要抄 qlib 而不是自己拍脑袋造因子
----------------------------------
qlib 的 Alpha158 是从 WorldQuant Alpha101 / GTJA Alpha191 等公开因子库里，
按「K 线形态 + 滚动统计 + 量价关系」系统性整理出来的，且在大规模 A 股数据上被
反复验证过（见 examples/benchmarks）。直接按它的定义实现，比自创指标可靠。

本文件是**精简版**：qlib 原版 158 个因子 × 5 个窗口 ≈ 700 列，那需要几千只股票
才有足够样本支撑；本项目股票池只有 27 只（约 1.9 万行），因子太多必然过拟合。
所以这里只保留 16 类最有信息量的算子 × 3 个窗口（5/10/20）+ 9 个 KBar 因子。

与 qlib 的对应关系（字段名加了 `a_` 前缀，避免和现有特征冲突）
------------------------------------------------------------
ROC / MA / STD / BETA / RSQR / RESI / MAX / MIN / RSV / CNTP /
IMAX / IMIN / CORR / CORD / VMA / VSTD / WVMA / VSUMP / SUMP  ← 同名同定义
（RANK 用 RSV 替代：两者高度相关，且 RANK 需要 rolling-rank，慢一个数量级）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 精简版算子列表（顺序即列顺序）
ALPHA_OPS = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI",
    "MAX", "MIN", "RSV", "CNTP", "IMAX", "IMIN",
    "CORR", "CORD", "VMA", "VSTD", "WVMA", "VSUMP", "SUMP",
]
ALPHA_WINDOWS = (5, 10, 20)


def _rolling_linreg(y: pd.Series, d: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """滚动线性回归 y ~ t（t=0..d-1），返回 (斜率, R², 残差标准差)。

    用滚动协方差/相关系数实现（pandas 内部是 C 实现的滚动窗口，比 rolling.apply 快两个数量级）：
        slope = cov(t, y) / var(t)
        R²    = corr(t, y)²
        resid = std(y) * sqrt(1 - R²)
    """
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    cov = y.rolling(d).cov(t)
    var_t = t.rolling(d).var()
    slope = cov / var_t.replace(0, np.nan)
    r2 = y.rolling(d).corr(t) ** 2
    r2 = r2.clip(0, 1)
    resid = y.rolling(d).std() * np.sqrt((1 - r2).clip(lower=0))
    return slope, r2, resid


def _idx_extreme(arr: np.ndarray, d: int, mode: str = "max") -> np.ndarray:
    """窗口内极值出现的位置（0=最早, 1=最近），qlib 的 IMAX/IMIN。"""
    out = np.full(len(arr), np.nan)
    if len(arr) < d:
        return out
    sw = sliding_window_view(arr, d)
    idx = np.argmax(sw, axis=1) if mode == "max" else np.argmin(sw, axis=1)
    out[d - 1:] = idx / d
    return out


def add_kbar(d: pd.DataFrame) -> pd.DataFrame:
    """qlib 的 9 个 KBar 因子（当日 K 线形状，全部用开盘价或振幅归一化）。

    振幅分母不要用 ``replace(0, nan)``：一字板 / 递归预测合成的 K 线会 high==low，
    那样会算出 NaN 并一路带进模型（sklearn 直接报错）。统一用 ``rng + 1e-12``。
    """
    o, h, l, c = d["open"], d["high"], d["low"], d["close"]
    rng = h - l
    safe_o = o.replace(0, np.nan)
    out = pd.DataFrame(index=d.index)
    out["a_KMID"] = (c - o) / safe_o
    out["a_KLEN"] = rng / safe_o
    out["a_KMID2"] = (c - o) / (rng + 1e-12)
    out["a_KUP"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / safe_o
    out["a_KUP2"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / (rng + 1e-12)
    out["a_KLOW"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / o.replace(0, np.nan)
    out["a_KLOW2"] = (pd.concat([o, c], axis=1).min(axis=1) - l) / (rng + 1e-12)
    out["a_KSFT"] = (2 * c - h - l) / o.replace(0, np.nan)
    out["a_KSFT2"] = (2 * c - h - l) / (rng + 1e-12)
    return out


def add_alpha158_lite(d: pd.DataFrame, windows: tuple[int, ...] = ALPHA_WINDOWS) -> pd.DataFrame:
    """追加 Alpha158 精简版因子列（a_ROC5 … a_SUMP20）。

    输入需含 open/high/low/close/volume；只用 T 日及之前数据，无前视。
    """
    out = add_kbar(d)
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    ret = c.pct_change()
    logv = np.log(v + 1)

    for w in windows:
        tag = str(w)
        out[f"a_ROC{tag}"] = c / c.shift(w) - 1
        out[f"a_MA{tag}"] = c.rolling(w).mean() / c
        out[f"a_STD{tag}"] = ret.rolling(w).std()

        slope, r2, resid = _rolling_linreg(c, w)
        out[f"a_BETA{tag}"] = slope / c
        out[f"a_RSQR{tag}"] = r2
        out[f"a_RESI{tag}"] = resid / c

        out[f"a_MAX{tag}"] = h.rolling(w).max() / c
        out[f"a_MIN{tag}"] = l.rolling(w).min() / c
        rng_w = h.rolling(w).max() - l.rolling(w).min()
        out[f"a_RSV{tag}"] = (c - l.rolling(w).min()) / (rng_w + 1e-12)

        up = (c > c.shift(1)).astype(float)
        out[f"a_CNTP{tag}"] = up.rolling(w).mean()
        out[f"a_IMAX{tag}"] = _idx_extreme(h.values.astype(float), w, "max")
        out[f"a_IMIN{tag}"] = _idx_extreme(l.values.astype(float), w, "min")

        out[f"a_CORR{tag}"] = c.rolling(w).corr(logv)
        out[f"a_CORD{tag}"] = ret.rolling(w).corr(np.log(v / v.shift(1) + 1))

        out[f"a_VMA{tag}"] = v.rolling(w).mean() / (v + 1e-12)
        out[f"a_VSTD{tag}"] = v.rolling(w).std() / (v + 1e-12)

        x = (ret.abs() * v)
        out[f"a_WVMA{tag}"] = x.rolling(w).std() / (x.rolling(w).mean() + 1e-12)
        dv = v.diff()
        out[f"a_VSUMP{tag}"] = (dv.clip(lower=0).rolling(w).sum()
                                / (dv.abs().rolling(w).sum() + 1e-12))
        dc = c.diff()
        out[f"a_SUMP{tag}"] = (dc.clip(lower=0).rolling(w).sum()
                               / (dc.abs().rolling(w).sum() + 1e-12))
    return out


def alpha_columns(windows: tuple[int, ...] = ALPHA_WINDOWS) -> list[str]:
    """返回 add_alpha158_lite 会生成的列名（顺序一致），供特征表使用。"""
    cols = ["a_KMID", "a_KLEN", "a_KMID2", "a_KUP", "a_KUP2",
            "a_KLOW", "a_KLOW2", "a_KSFT", "a_KSFT2"]
    for w in windows:
        cols += [f"a_{op}{w}" for op in ALPHA_OPS]
    return cols
