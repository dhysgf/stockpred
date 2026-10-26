"""技术指标特征工程（多周期预测版）。

特征均只用 T 日及之前数据，避免前向偏差（look-ahead bias）。
因子研究建议：用 5-10 个强因子即可，避免过多弱因子过拟合；这里按「类别」精选：
价格动量 / 均线 / 摆动指标 / 波动 / 量能 / K线形态 / 大盘 beta / 相对强弱。
"""
from __future__ import annotations

import pandas as pd

import config
from .alpha import add_alpha158_lite, alpha_columns
from .patterns import add_patterns
from .volume import add_volume_indicators

# 核心特征（顺序即模型输入顺序）
CORE_FEATURES = [
    "ret", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5",
    "ma5", "ma10", "ma20", "rsi", "macd_hist",
    "boll_pctb", "stoch_k", "cci", "atr_norm", "mom5",
    "vol", "vol_ratio", "sentiment",
    # 大盘（上证指数）因子：个股方向主要由市场 beta 驱动，是必要的外生特征
    "idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev",
    # 相对强弱：个股相对大盘的超额收益（剥离 beta 后的个股动量/反转信号）
    "rs5", "rs20",
    # K 线形态（多空形态强度聚合，见 patterns.py）
    "pat_score", "pat_bull", "pat_bear",
    # 量能（历史可得的量价指标，见 volume.py）
    "mfi", "obv_slope5", "vol_expand", "up_vol_ratio",
    # 位置：距 20 日支撑/压力（突破/破位上下文）
    "dist_high20", "dist_low20",
]

# FEATURE_SET="alpha" 时追加 qlib Alpha158 精简版因子
ALPHA_COLS = alpha_columns(getattr(config, "ALPHA_WINDOWS", (5, 10, 20)))
FEATURES = CORE_FEATURES + (ALPHA_COLS if getattr(config, "FEATURE_SET", "core") == "alpha" else [])



def add_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """在 K 线 DataFrame 上追加技术指标与滞后特征。

    输入列：open, high, low, close, volume。
    输出追加：ret, ret_lag1..5, ma5/10/20, rsi, macd_hist,
              boll_pctb, stoch_k, cci, atr_norm, mom5, vol, vol_ratio。
    """
    d = df.copy()
    c = d["close"]
    d["ret"] = c.pct_change()
    # 滞后收益（序列自相关 / 均值回复）
    for k in (1, 2, 3, 5):
        d[f"ret_lag{k}"] = d["ret"].shift(k)

    d["ma5"] = c.rolling(5).mean()
    d["ma10"] = c.rolling(10).mean()
    d["ma20"] = c.rolling(20).mean()

    # RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # MACD(12,26,9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # 布林带 %b
    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    up = mid + 2 * sd
    lo = mid - 2 * sd
    d["boll_pctb"] = (c - lo) / (up - lo + 1e-9)

    # 随机指标 %K(9)
    low9 = d["low"].rolling(9).min()
    high9 = d["high"].rolling(9).max()
    d["stoch_k"] = (c - low9) / (high9 - low9 + 1e-9)

    # CCI(20)
    typ = (d["high"] + d["low"] + c) / 3
    sma_t = typ.rolling(20).mean()
    mad = (typ - sma_t).abs().rolling(20).mean()
    d["cci"] = (typ - sma_t) / (0.015 * mad + 1e-9)

    # ATR(14) 归一化
    prev = c.shift(1)
    tr = pd.concat(
        [(d["high"] - d["low"]).abs(), (d["high"] - prev).abs(), (d["low"] - prev).abs()], axis=1
    ).max(axis=1)
    d["atr_norm"] = tr.rolling(14).mean() / (c + 1e-9)

    # 动量(5/20)
    d["mom5"] = c / c.shift(5) - 1
    d["mom20"] = c / c.shift(20) - 1

    # 波动率（20 日收益率 std）与量比
    d["vol"] = d["ret"].rolling(20).std()
    d["vol_ma5"] = d["volume"].rolling(5).mean()
    d["vol_ratio"] = d["volume"] / (d["vol_ma5"] + 1e-9)

    # 支撑/压力位置：距前 20 日高/低点的相对距离（突破用正、破位用负）
    hi20 = d["high"].rolling(20).max().shift(1)
    lo20 = d["low"].rolling(20).min().shift(1)
    d["dist_high20"] = c / (hi20 + 1e-9) - 1
    d["dist_low20"] = c / (lo20 + 1e-9) - 1

    # K 线形态（经典形态强度聚合）与量能（OBV/MFI/量比/涨量占比）
    d = add_patterns(d)
    d = add_volume_indicators(d)

    # qlib Alpha158 精简版（可选）：KBar 9 + 19 算子 × 3 窗口
    if getattr(config, "FEATURE_SET", "core") == "alpha":
        d = pd.concat([d, add_alpha158_lite(d, getattr(config, "ALPHA_WINDOWS", (5, 10, 20)))], axis=1)

    # 指数因子兜底：未调用 enrich_with_index 时填 0（模型退化为纯个股因子，仍可训练）
    for c in ("idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev"):
        if c not in d.columns:
            d[c] = 0.0
    # 相对强弱（个股 - 大盘）：任何下游消费者（含拐点模型）都必须能拿到这两列
    d["rs5"] = d["mom5"] - d["idx_mom5"]
    d["rs20"] = d["mom20"] - d["idx_mom20"]

    d["sentiment"] = 0.5  # 占位，由调用方填充
    return d


def make_feature_matrix(d: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """把带技术指标的 DataFrame 转成特征矩阵（与训练/预测口径一致）。

    均线换算为相对收盘价偏离，量纲更稳；其余指标已自带归一化。
    """
    d = d.copy()
    d["sentiment"] = sentiment
    d["ma5"] = d["ma5"] / d["close"] - 1
    d["ma10"] = d["ma10"] / d["close"] - 1
    d["ma20"] = d["ma20"] / d["close"] - 1
    # 相对强弱（个股 - 大盘）：剥离市场 beta 后的超额收益，是选股/方向的核心信号
    if "idx_mom5" in d.columns:
        d["rs5"] = d["mom5"] - d["idx_mom5"]
        d["rs20"] = d["mom20"] - d["idx_mom20"]
    else:
        d["rs5"] = 0.0
        d["rs20"] = 0.0
    return d[FEATURES]


def last_feature_row(d: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """取最新一行特征（1 行 DataFrame），供预测使用。

    预测行必须清洗一次：递归预测合成的 K 线、或新股暖机期，可能让个别因子出现
    inf/NaN。训练侧靠 mask 丢掉这些行，预测侧没有下一行可丢，只能按中性值 0 兜底
    （z-score 后 0 就是截面均值），否则 sklearn 会直接报 NaN/inf 错误。
    """
    X = make_feature_matrix(d, sentiment).iloc[[-1]].reset_index(drop=True)
    return X.replace([float("inf"), float("-inf")], float("nan")).fillna(0.0)


def apply_zscore_cols(d: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """按给定截面统计量把特征列标准化（用在「最终特征矩阵」上）。

    注意：必须在 make_feature_matrix 之后调用——ma 已归一化为相对收盘价偏离，
    若先标准化再除以 close 会破坏口径。
    """
    d = d.copy()
    for c, (mu, sd) in (stats or {}).items():
        if c in d.columns and sd:
            d[c] = (d[c] - mu) / sd
    return d


# 大盘指数因子列
_IDX_COLS = ["idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev"]


def enrich_with_index(kline: pd.DataFrame, index_kline: pd.DataFrame | None = None) -> pd.DataFrame:
    """把上证指数（或传入指数）的动量/均线偏离合并进个股 K 线。

    指数特征均由指数历史计算，按日期对齐后合并；walk-forward 切分时
    只取个股截至切分点的行，指数信息不会泄露未来 —— 安全。
    若指数不可用，相关列填 0（模型退化为仅个股因子，仍可训练）。
    """
    out = kline.copy()
    if index_kline is None or index_kline.empty:
        for c in _IDX_COLS:
            out[c] = 0.0
        return out
    idx = index_kline[["date", "close"]].copy().rename(columns={"close": "idx_close"})
    idx["idx_ret"] = idx["idx_close"].pct_change()
    idx["idx_mom5"] = idx["idx_close"] / idx["idx_close"].shift(5) - 1
    idx["idx_mom20"] = idx["idx_close"] / idx["idx_close"].shift(20) - 1
    mid = idx["idx_close"].rolling(20).mean()
    idx["idx_ma_dev"] = idx["idx_close"] / mid - 1
    out = out.merge(idx[["date"] + _IDX_COLS], on="date", how="left")
    out[_IDX_COLS] = out[_IDX_COLS].ffill().fillna(0.0)
    return out


def build_horizon_samples(d: pd.DataFrame, sentiment: float, horizons: list[int]):
    """构造多周期监督样本（DIRMO：每个周期 h 直接以 h 日收益率为目标）。

    返回 {h: (X, y_dir, y_ret)}，X 为各周期共用的最新特征矩阵
    （仅保留目标可用的行，避免用到未来数据）。
    """
    tech = add_technicals(d)
    base = make_feature_matrix(tech, sentiment)
    out = {}
    for h in horizons:
        y_ret = tech["close"].shift(-h) / tech["close"] - 1
        y_dir = (y_ret > 0).astype(int)
        # 最后 h 行无 h 日目标，需剔除；X 同步裁剪
        X = base.iloc[:-h] if h > 0 else base
        yd = y_dir.iloc[:-h] if h > 0 else y_dir
        yr = y_ret.iloc[:-h] if h > 0 else y_ret
        mask = X.notna().all(axis=1) & yr.notna()
        out[h] = (X[mask].reset_index(drop=True), yd[mask].reset_index(drop=True), yr[mask].reset_index(drop=True))
    return out
