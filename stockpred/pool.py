"""多股票池横截面训练（标准量化做法）。

单只个股的时间序列自回归信号极弱（样本仅数百天、噪声大）；
改用「多只股票池化 + 横截面标准化」后，样本量提升一两个数量级，
模型才能学到真正有效的因子（相对强弱、动量、波动等）。

要点：
- 个股特征按**当日截面** Z-score 标准化（剔除时点/市场共同波动的影响）
- 市场级共同因子（指数动量、新闻情绪）**不做**标准化，否则会被抹平
"""
from __future__ import annotations

import threading

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from .data_fetcher import load
from .features import add_technicals, enrich_with_index, FEATURES
import config  # 项目根目录配置

# 行情源最新交易日缓存（避免每只票都去打一次指数）
_LATEST_DAY = {"day": "", "ts": 0.0}
_LATEST_DAY_LOCK = threading.Lock()

# 默认股票池：A 股各主要行业龙头（流动性好、覆盖广）
DEFAULT_POOL = [
    ("600519", "贵州茅台"), ("000858", "五粮液"), ("600887", "伊利股份"),
    ("601318", "中国平安"), ("600036", "招商银行"), ("601398", "工商银行"),
    ("601288", "农业银行"), ("601166", "兴业银行"), ("000001", "平安银行"),
    ("600000", "浦发银行"), ("600030", "中信证券"), ("600276", "恒瑞医药"),
    ("000333", "美的集团"), ("000651", "格力电器"), ("600104", "上汽集团"),
    ("002594", "比亚迪"), ("300750", "宁德时代"), ("601012", "隆基绿能"),
    ("601899", "紫金矿业"), ("600028", "中国石化"), ("601857", "中国石油"),
    ("600309", "万华化学"), ("600585", "海螺水泥"), ("600019", "宝钢股份"),
    ("600900", "长江电力"), ("601888", "中国中免"), ("000002", "万科A"),
]

# 市场级共同因子：所有股票当日取值相同，标准化会被抹平，保留原值
COMMON_COLS = ["idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev", "sentiment"]

# 需要按当日截面做 Z-score 的「个股特有」特征：直接由 FEATURES 派生，
# 这样切 FEATURE_SET（core/alpha）时不用两处同步，避免"加了特征忘了进面板"的坑。
CROSS_COLS = [c for c in FEATURES if c not in COMMON_COLS]


def _kline_cache_path(code: str, datalen: int):
    from .universe import cache_dir
    return cache_dir() / "klines" / f"{code}_{datalen}.csv"


def _cached_last_date(cp) -> str:
    """读缓存 CSV 的最后一根日线日期（只读文件尾部，不全量解析）。"""
    try:
        with open(cp, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        if len(lines) < 2:
            return ""
        return lines[-1].split(",")[0].strip()[:10]
    except (OSError, IndexError):
        return ""


def _latest_market_day() -> str:
    """行情源目前最新的一根日线是哪天（YYYY-MM-DD），缓存 5 分钟。

    用于判断缓存是否"数据落后"：文件年龄 1 天以内不代表数据是新的 ——
    周末、节假日、或当天盘中抓的缓存，都可能落后 1~2 个交易日。
    """
    import time as _t
    global _LATEST_DAY
    with _LATEST_DAY_LOCK:
        if _LATEST_DAY["day"] and (_t.time() - _LATEST_DAY["ts"]) < 300:
            return _LATEST_DAY["day"]
    day = ""
    try:
        from .data_fetcher import fetch_kline
        df = fetch_kline("sh000001", datalen=3)
        if df is not None and not df.empty:
            day = str(pd.to_datetime(df["date"].iloc[-1]).date())
    except Exception:
        day = ""
    with _LATEST_DAY_LOCK:
        _LATEST_DAY["day"] = day
        _LATEST_DAY["ts"] = _t.time()
    return day


def _load_kline_cached(code: str, datalen: int, source: str, ttl: float,
                       refresh: bool = False) -> pd.DataFrame | None:
    """带磁盘缓存的单只股票 K 线读取。

    300 只票每次重抓要 20~60s（还容易被限流）；缓存后二次运行几乎瞬时。

    ★ 命中缓存要同时满足两个条件：
      1) 文件年龄 < ttl；
      2) **缓存里的最后一根日线日期 == 行情源最新交易日**。
    只看 (1) 会踩坑：周末/节假日/当天盘中抓的缓存，文件很新但数据落后 1~2 个交易日，
    于是报告页会显示"最新收盘 27.40（2026-10-21）"而用户以为拿到了当天数据。
    """
    cp = _kline_cache_path(code, datalen)
    if not refresh and ttl > 0 and cp.exists():
        try:
            import time as _t
            if (_t.time() - cp.stat().st_mtime) < ttl:
                latest = _latest_market_day()
                cached_day = _cached_last_date(cp)
                # 行情源最新交易日探不到（网络问题）时，退回只按文件年龄判断
                day_ok = (not latest) or (not cached_day) or (cached_day >= latest)
                if day_ok:
                    d = pd.read_csv(cp, parse_dates=["date"])
                    if len(d) > 0:
                        return d
        except Exception:
            pass
    d = load(code, datalen=datalen, source=source).kline
    if d is not None and len(d) > 0:
        try:
            d.to_csv(cp, index=False)
        except Exception:
            pass
    return d


def fetch_pool(datalen: int = 700, source: str = "sina", symbols=None, workers: int = 8,
               extra: list[str] | None = None, universe: str | None = None,
               limit: int | None = None, refresh: bool = False) -> dict:
    """并发抓取股票池历史行情（带磁盘缓存），返回 {symbol: DataFrame}。

    universe：default / hs300 / zz500；None 时读 config.POOL_MODE。
    extra：额外加入的标的（通常是用户查询的目标股），保证目标一定在池子里，
    否则池化模型找不到它的特征行。新股历史短也没关系 —— 它只作为预测对象，
    训练样本由池中其他老股票提供（这正是池化能预测新股的原因）。
    """
    if symbols is None:
        from .universe import get_universe
        symbols = get_universe(mode=universe, limit=limit, refresh=refresh)
    pool = [(c, n, 120) for c, n in symbols]
    for e in (extra or []):
        s = str(e).strip().lower()
        code = s[2:] if s.startswith(("sh", "sz")) else s
        if not any(code == c for c, _, _ in pool):
            # 目标股：只需少量 K 线即可（它作为预测对象，训练样本由其他老股票提供）
            pool.append((code, code, 20))
    out = {}

    def _one(item):
        code, _name, min_bars = item
        try:
            d = _load_kline_cached(code, datalen, source,
                                   float(getattr(config, "KLINE_CACHE_TTL", 86400)), refresh)
            return code, d if d is not None and len(d) >= min_bars else None
        except Exception:
            return code, None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, d in ex.map(_one, pool):
            if d is not None:
                out[code] = d
    return out


def build_panel(pool: dict, index_kline: pd.DataFrame | None) -> pd.DataFrame:
    """把股票池拼成长表面板：每行 = (某股票, 某日) 的一条样本。

    含技术指标、指数特征、相对强弱，以及 h 日收益率目标所需的 close。
    """
    frames = []
    for sym, kline in pool.items():
        k = enrich_with_index(kline, index_kline)
        t = add_technicals(k)
        # 均线已在 make_feature_matrix 里归一化，这里先按同样口径处理
        t["ma5"] = t["ma5"] / t["close"] - 1
        t["ma10"] = t["ma10"] / t["close"] - 1
        t["ma20"] = t["ma20"] / t["close"] - 1
        # 相对强弱：个股动量 - 大盘动量（剥离 beta）
        t["rs5"] = t["mom5"] - t["idx_mom5"]
        t["rs20"] = t["mom20"] - t["idx_mom20"]
        t = t[["date", "close"] + CROSS_COLS + ["idx_ret", "idx_mom5", "idx_mom20", "idx_ma_dev"]].copy()
        t["symbol"] = sym
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def cross_stats(panel: pd.DataFrame, date=None) -> dict:
    """取某个交易日（默认最新）截面上各特征的均值/标准差，用于标准化。"""
    if panel.empty:
        return {}
    d = panel[panel["date"] == date] if date is not None else panel[panel["date"] == panel["date"].max()]
    stats = {}
    for c in CROSS_COLS:
        if c not in d.columns:
            continue
        mu = float(d[c].mean())
        sd = float(d[c].std())
        stats[c] = (mu, sd if sd and not np.isnan(sd) and sd > 1e-12 else 1.0)
    return stats


def apply_zscore(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """用给定截面统计量把个股特征标准化。"""
    d = df.copy()
    for c, (mu, sd) in stats.items():
        if c in d.columns:
            d[c] = (d[c] - mu) / sd
    return d


def zscore_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """按每个交易日截面标准化整个面板（用于训练）。"""
    if panel.empty:
        return panel
    cols = [c for c in CROSS_COLS if c in panel.columns]
    out = panel.copy()
    grp = out.groupby("date")[cols]
    mu = grp.transform("mean")
    sd = grp.transform("std").replace(0, np.nan)
    out[cols] = (out[cols] - mu) / sd
    out[cols] = out[cols].fillna(0.0)
    return out


def pooled_samples(panel: pd.DataFrame, sentiment: float, horizons: list[int],
                   label_mode: str | None = None) -> dict:
    """用整个面板构造各周期的池化训练样本 {h: (X, y_dir, y_reg)}。

    LABEL_MODE="csrank" 时 y_reg 换成**当日截面排名**（qlib CSRankNorm：
    rank(pct)-0.5 再乘 3.46 使其近似单位方差）。好处是抗极值、跨股可比，
    对横截面模型更稳；代价是预测值不再直接是收益率，需要用
    ``pooled_label_scale`` 给出的收益标准差换回价格口径。
    方向分类标签始终是「真实收益 > 0」，不受影响。
    """
    mode = (label_mode or getattr(config, "LABEL_MODE", "ret")).lower()
    p = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    p["sentiment"] = sentiment
    feat = [c for c in FEATURES if c in p.columns]
    max_rows = int(getattr(config, "MAX_TRAIN_ROWS", 0) or 0)
    rng = np.random.RandomState(getattr(config, "RANDOM_STATE", 42))
    out = {}
    for h in horizons:
        fut = p.groupby("symbol")["close"].shift(-h)
        y_ret = fut / p["close"] - 1
        y_dir = (y_ret > 0).astype(int)
        if mode == "csrank":
            cs = p.assign(_y=y_ret).groupby("date")["_y"].rank(pct=True) - 0.5
            y_reg = cs * 3.46
        else:
            y_reg = y_ret
        mask = y_ret.notna() & p[feat].notna().all(axis=1)
        idx = p.index[mask]
        if max_rows and len(idx) > max_rows:
            # 抽样是为了控制"策略打分要重训几十次"的耗时，不是提升效果；
            # 用固定种子保证可复现。
            idx = pd.Index(rng.choice(idx.values, size=max_rows, replace=False))
            idx = idx.sort_values()
        out[h] = (p.loc[idx, feat].reset_index(drop=True),
                  y_dir.loc[idx].reset_index(drop=True),
                  y_reg.loc[idx].reset_index(drop=True))
    return out


def pooled_label_scale(panel: pd.DataFrame, horizons: list[int]) -> dict:
    """各周期真实收益的标准差：csrank 预测值 → 收益率 的换算系数。"""
    p = panel.sort_values(["symbol", "date"]).reset_index(drop=True)
    scale = {}
    for h in horizons:
        r = p.groupby("symbol")["close"].shift(-h) / p["close"] - 1
        sd = float(r.std())
        scale[h] = sd if sd and not np.isnan(sd) and sd > 1e-9 else 0.02
    return scale


def walk_forward_backtest_pooled(panel: pd.DataFrame, target: str, sentiment: float,
                                 horizons=(1, 5, 7, 30), n_splits: int = 5,
                                 offsets=(120, 100, 80, 60, 45)) -> dict:
    """池化模型的 walk-forward 样本外回测。

    每个切分点：用「所有股票截至该日」的数据训练 → 预测目标股 → 与真实价对比。
    """
    td = panel[panel["symbol"] == target].sort_values("date").reset_index(drop=True)
    dates = td["date"].tolist()
    closes = td["close"].tolist()
    n = len(dates)
    if n < 200:
        return {"summary": pd.DataFrame(), "details": {}}

    from .models import build_forecaster

    agg = {h: {"mae": [], "mape": [], "dir": [], "conf_dir": []} for h in horizons}
    details = {}
    for i, off in enumerate(offsets[:n_splits]):
        idx = n - 1 - off
        if idx < 120:
            continue
        cut = dates[idx]
        sub = panel[panel["date"] <= cut]
        if sub.empty:
            continue
        pz = zscore_panel(sub)
        samples = pooled_samples(pz, sentiment, list(horizons))
        model = build_forecaster()
        if str(getattr(config, "LABEL_MODE", "ret")).lower() == "csrank":
            model.ret_scale = pooled_label_scale(sub, list(horizons))
        model.fit(samples, {"panel": pz, "symbol": target})
        if not model.fit_ok:
            continue
        # 标准化统计量必须取自**原始**截面：pz 已标准化，其均值≈0/方差≈1，
        # 再拿它当 stats 等于对预测行不做任何换算（历史 bug，已修）
        stats = cross_stats(sub)
        Xr = target_row(sub, target, sentiment, stats)
        if Xr.empty:
            continue
        base = float(closes[idx])
        ctx = {"panel": pz, "symbol": target}
        for h in horizons:
            if h not in model.horizons or idx + h >= n:
                continue
            r, prob = model.predict_horizon(h, Xr, ctx)
            r = float(max(-config.PRED_RET_CLIP, min(config.PRED_RET_CLIP, r)))
            pred = base * (1 + r)
            actual = float(closes[idx + h])
            ape = abs(pred - actual) / actual * 100
            ok = (pred > base) == (actual > base)
            agg[h]["mae"].append(abs(pred - actual))
            agg[h]["mape"].append(ape)
            agg[h]["dir"].append(ok)
            if prob >= config.CONF_HI or prob <= config.CONF_LO:
                agg[h]["conf_dir"].append(ok)
            details[(h, i)] = pd.DataFrame([{
                "date": str(dates[idx + h].date()), "actual": round(actual, 2),
                "pred": round(pred, 2), "abs_err": round(abs(pred - actual), 2),
                "ape": round(ape, 2), "dir_ok": ok, "prob": round(prob, 3),
            }])

    rows = []
    for h in horizons:
        a = agg[h]
        if not a["mae"]:
            continue
        n_all, n_conf = len(a["dir"]), len(a["conf_dir"])
        rows.append({
            "horizon(日)": h,
            "MAE": round(float(np.mean(a["mae"])), 2),
            "MAPE(%)": round(float(np.mean(a["mape"])), 2),
            "方向准确率(%)": round(float(np.mean(a["dir"]) * 100), 1),
            "明确时准确率(%)": round(float(np.mean(a["conf_dir"]) * 100), 1) if n_conf else "-",
            "明确占比(%)": round(n_conf / n_all * 100, 1),
            "样本数": n_all,
        })
    return {"summary": pd.DataFrame(rows), "details": details}


def evaluate_cross_sectional(panel: pd.DataFrame, sentiment: float, horizons=(1, 5, 7, 30),
                             step: int = 30, min_train: int = 350, max_evals: int = 10) -> pd.DataFrame:
    """横截面大样本方向评估（IC/IR 式）。

    单只股票只有 5~7 个回测切分点，方向准确率的置信区间极宽，根本无法判断
    模型有没有真实预测力。这里改为：定期用「截至该日的所有股票」训练一次，
    然后对接下来 step 个交易日、池内**所有股票**都做一次方向预测，
    样本量提升到数千量级，结论才有统计意义。
    """
    from .model import ForecastModel

    dates = sorted(panel["date"].unique())
    n = len(dates)
    if n < min_train + step:
        return pd.DataFrame()

    # 预建：每只股票的日期/收盘序列 + (股票,日期)→位置 索引，避免逐行 O(n) 查找
    seq, pos = {}, {}
    for s, g in panel.groupby("symbol"):
        g = g.sort_values("date")
        dts, cls = g["date"].tolist(), g["close"].tolist()
        seq[s] = (dts, cls)
        for k, d in enumerate(dts):
            pos[(s, d)] = k

    starts = list(range(min_train, n - 40, step))
    if len(starts) > max_evals:
        starts = starts[-max_evals:]

    rows = []
    for si in starts:
        cut = dates[si - 1]
        sub = panel[panel["date"] <= cut]
        if sub.empty:
            continue
        pz = zscore_panel(sub)
        samples = pooled_samples(pz, sentiment, list(horizons))
        model = ForecastModel()
        model.fit(samples)
        if not model.fit_ok:
            continue
        for ei in range(si, min(si + step, n)):
            edate = dates[ei]
            ev = panel[panel["date"] == edate]
            if ev.empty:
                continue
            ez = zscore_panel(ev)          # 用当日截面标准化（不含未来信息）
            ez["sentiment"] = sentiment
            feat = [c for c in FEATURES if c in ez.columns]
            Xall = ez[feat].reset_index(drop=True)   # 当日所有股票，批量预测
            syms = ez["symbol"].tolist()
            for h in horizons:
                if h not in model.regs:
                    continue
                rets = model.regs[h].predict(Xall)              # 批量，比逐行快两个数量级
                probs = model.clfs[h].predict_proba(Xall)[:, 1]
                for j, sym in enumerate(syms):
                    k = pos.get((sym, edate))
                    if k is None:
                        continue
                    _dts, sc = seq[sym]
                    if k + h >= len(sc):
                        continue
                    r = float(max(-config.PRED_RET_CLIP, min(config.PRED_RET_CLIP, rets[j])))
                    base, actual = float(sc[k]), float(sc[k + h])
                    rows.append({
                        "horizon": h, "symbol": sym, "date": str(edate.date()),
                        "pred_up": r > 0, "actual_up": actual > base,
                        "correct": (r > 0) == (actual > base),
                        "prob": float(probs[j]),
                    })
    return pd.DataFrame(rows)


def target_row(panel: pd.DataFrame, symbol: str, sentiment: float, stats: dict) -> pd.DataFrame:
    """取目标股票最新一日的特征（已按最新截面标准化），供预测使用。"""
    p = panel[panel["symbol"] == symbol]
    if p.empty:
        return pd.DataFrame()
    last_date = p["date"].max()
    row = p[p["date"] == last_date].copy()
    row["sentiment"] = sentiment
    row = apply_zscore(row, stats)
    feat = [c for c in FEATURES if c in row.columns]
    return row[feat].reset_index(drop=True)
