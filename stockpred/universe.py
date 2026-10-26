"""股票池（universe）管理：成分股获取 + 本地缓存。

为什么需要这一层
---------------
原来池子是硬编码的 27 只行业龙头，样本量 1.9 万行。这个量级下：
- LSTM 的 1.7 万序列样本对深度学习来说太少；
- 组合策略 TopK=5 在 27 只里选，集中度极高，IC 0.045 的微弱信号抓不住。

扩到沪深 300 后样本量提升一个数量级（约 21 万行）。

⚠️ **幸存者偏差**：这里取的是**当前**沪深 300 成分股，用它回测历史等于
"提前知道哪些票现在还留在指数里"。qlib 的 cn_data 带历史成分股变动表，
本项目没有，所以策略回测的超额收益需要打个折扣看。这一点必须写清楚，
不能拿含幸存者偏差的净值去吹策略有效。
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import config

_UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
_EM_LIST = ("https://push2.eastmoney.com/api/qt/clist/get"
            "?pn={page}&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3&fs={fs}&fields=f12,f14")

# 常用指数/板块的东财板块代码
_BOARDS = {
    "hs300": "b:BK0500",    # 沪深300
    "zz500": "b:BK0700",    # 中证500
    "sz50": "b:BK0611",     # 上证50（板块代码，非必须）
}
_NAMES = {"hs300": "沪深300", "zz500": "中证500", "sz50": "上证50", "default": "默认龙头池"}


def cache_dir() -> Path:
    d = Path(getattr(config, "CACHE_DIR", ".cache"))
    (d / "klines").mkdir(parents=True, exist_ok=True)
    return d


def _fetch_board(fs: str, pz: int = 100, max_pages: int = 12) -> list[tuple[str, str]]:
    """分页拉取板块成分股。

    坑：pz 传 200 时东财第 2 页会重复返回第 1 页的内容（去重后只剩 200 只），
    pz=100 分页正常。所以这里固定 pz=100 并按 total 循环。
    """
    out, seen = [], set()
    total = None
    for page in range(1, max_pages + 1):
        url = _EM_LIST.format(page=page, fs=fs)
        req = urllib.request.Request(url, headers=_UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            j = json.loads(r.read().decode("utf-8", "ignore"))
        data = j.get("data") or {}
        diff = data.get("diff") or []
        if not diff:
            break
        for x in diff:
            code = str(x.get("f12", "")).strip()
            name = str(x.get("f14", "")).strip()
            if not code or code in seen:
                continue
            seen.add(code)
            out.append((code, name))
        total = int(data.get("total") or 0)
        if total and len(out) >= total:
            break
        time.sleep(0.2)
    return out


def get_universe(mode: str | None = None, limit: int | None = None,
                 refresh: bool = False) -> list[tuple[str, str]]:
    """返回 [(code, name)]。mode ∈ {default, hs300, zz500}。

    成分股列表缓存 7 天（指数调整是半年一次，没必要每次拉）。
    """
    from .pool import DEFAULT_POOL

    mode = (mode or getattr(config, "POOL_MODE", "default")).lower()
    limit = int(limit if limit is not None else getattr(config, "POOL_LIMIT", 0) or 0)
    if mode in ("default", "custom", "none"):
        pool = list(DEFAULT_POOL)
    else:
        cp = cache_dir() / f"universe_{mode}.json"
        ttl = float(getattr(config, "POOL_CACHE_TTL", 7 * 86400))
        pool = None
        if cp.exists() and not refresh and (time.time() - cp.stat().st_mtime) < ttl:
            try:
                pool = [(str(a), str(b)) for a, b in json.loads(cp.read_text("utf-8"))]
            except Exception:
                pool = None
        if not pool:
            fs = _BOARDS.get(mode)
            if not fs:
                print(f"[pool] 未知股票池 {mode}，退回默认池")
                pool = list(DEFAULT_POOL)
            else:
                try:
                    pool = _fetch_board(fs)
                    cp.write_text(json.dumps(pool, ensure_ascii=False), "utf-8")
                except Exception as e:
                    print(f"[pool] 成分股获取失败（{e}），退回默认池")
                    pool = list(DEFAULT_POOL)
    if limit and limit > 0:
        pool = pool[:limit]
    return pool


def pool_label(mode: str | None = None, n: int = 0) -> str:
    mode = (mode or getattr(config, "POOL_MODE", "default")).lower()
    return f"{_NAMES.get(mode, mode)}({n} 只)"
