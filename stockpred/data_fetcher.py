"""行情数据层：历史日 K 线 + 实时行情。

默认数据源：新浪财经（真实、免 token、本环境可用）。
预留 yfinance 通道（美股/全球），通过 config.DATA_SOURCE 切换。
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0"}


def _http_get(url: str, timeout: int = 15, retries: int = 3) -> str:
    """带重试的 GET（新浪接口偶发 DNS/超时失败，重试可显著提高稳定性）。"""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            last = e
            time.sleep(0.6 * (i + 1))
    raise last


def normalize_symbol(symbol: str) -> str:
    """把各种写法归一化为新浪需要的 sh/sz 前缀代码。

    - "600519"  -> "sh600519"
    - "000001"  -> "sz000001"
    - "sh600519"-> "sh600519"（原样）
    - "AAPL"    -> "AAPL"（美股，yfinance 用）
    """
    s = symbol.strip().lower()
    if s.startswith(("sh", "sz", "us")) and not s[2:].isdigit():
        # 形如 sh600519
        return s
    if s.startswith(("sh", "sz")) and len(s) == 8 and s[2:].isdigit():
        return s
    if s.isdigit():
        # A 股：6 开头沪市，0/3 开头深市，8/4 开头北交所
        if s[0] == "6":
            return "sh" + s
        if s[0] in ("0", "3"):
            return "sz" + s
        if s[0] in ("8", "4"):
            return "bj" + s
    return s  # 美股等原样返回


# ------------------------- 新浪数据源 -------------------------
def _sina_kline(symbol: str, datalen: int = 400) -> pd.DataFrame:
    s = normalize_symbol(symbol)
    url = (
        "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={s}&scale=240&ma=no&datalen={datalen}"
    )
    raw = _http_get(url)
    arr = json.loads(raw)
    if not arr:
        raise ValueError(f"新浪未返回历史数据：{symbol}")
    rows = []
    for b in arr:
        rows.append(
            {
                "date": b["day"][:10],
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": float(b["volume"]),
            }
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def _sina_quote(symbol: str) -> dict:
    """实时行情快照 + **五档盘口**（委买/委卖各 5 档）。

    新浪 hq 字段布局（GBK，逗号分隔，共 30+ 项）：
      0 名称 1 今开 2 昨收 3 现价 4 最高 5 最低 6 买一价 7 卖一价
      8 成交量(股) 9 成交额(元)
      10..19 买一~买五 (量, 价)   20..29 卖一~卖五 (量, 价)
      30 日期 31 时间
    """
    s = normalize_symbol(symbol)
    url = f"https://hq.sinajs.cn/list={s}"
    headers = {**UA, "Referer": "https://finance.sina.com.cn"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            raw = r.read().decode("gbk", "ignore")
    except Exception as e:
        print(f"[quote] 实时行情获取失败（不影响预测）：{e}")
        return {}
    # 返回形如：var hq_str_sh600519="贵州茅台,1244.600,...";
    try:
        payload = raw.split('"')[1]
    except Exception:
        return {}
    parts = payload.split(",")
    if len(parts) < 6:
        return {}

    def _f(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _pairs(start: int) -> list:
        out = []
        for i in range(start, start + 10, 2):
            if i + 1 < len(parts):
                vol, price = _f(parts[i]), _f(parts[i + 1])
                if vol > 0 and price > 0:
                    out.append((vol, price))
        return out

    bids = _pairs(10)   # 买一~买五：(量(股), 价)
    asks = _pairs(20)   # 卖一~卖五
    bid_vol = sum(x[0] for x in bids)
    ask_vol = sum(x[0] for x in asks)
    depth_ratio = (bid_vol - ask_vol) / (bid_vol + ask_vol) * 100 if (bid_vol + ask_vol) else 0.0

    return {
        "name": parts[0],
        "open": _f(parts[1]),
        "prev_close": _f(parts[2]),
        "price": _f(parts[3]),
        "high": _f(parts[4]),
        "low": _f(parts[5]),
        "volume": _f(parts[8]) if len(parts) > 8 else 0.0,   # 股
        "amount": _f(parts[9]) if len(parts) > 9 else 0.0,   # 元
        "bids": bids,
        "asks": asks,
        "bid_vol": bid_vol,
        "ask_vol": ask_vol,
        "weibi": round(depth_ratio, 2),      # 委比(%)
        "weicha": bid_vol - ask_vol,         # 委差(股)
        "time": (parts[30] + " " + parts[31]) if len(parts) > 31 else "",
    }


def fetch_money_flow(symbol: str) -> dict:
    """新浪个股资金流向（当日快照）：超大/大/中/小单买卖金额 + 主力净额。

    返回 ok=False 表示接口不可用，调用方应把该信号降权/跳过，而不是当 0 处理。
    """
    s = normalize_symbol(symbol)
    url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           f"MoneyFlow.ssi_ssfx_flzjtj?daima={s}")
    try:
        j = json.loads(_http_get(url))
    except Exception as e:
        return {"ok": False, "reason": f"资金流接口失败：{e}"}
    if not isinstance(j, dict) or "netamount" not in j:
        return {"ok": False, "reason": "资金流接口无数据（可能停牌/非A股）"}

    def _f(k):
        try:
            return float(j.get(k, 0) or 0)
        except Exception:
            return 0.0

    net = _f("netamount")
    # 实证口径：接口 volume 为「股」（与 hq 快照一致），trade 为价格 → 成交额 = trade*volume
    amount = _f("trade") * _f("volume")
    net_ratio = (net / amount * 100) if amount else 0.0
    tiers = {
        "超大单": (_f("r0_in"), _f("r0_out")),
        "大单": (_f("r1_in"), _f("r1_out")),
        "中单": (_f("r2_in"), _f("r2_out")),
        "小单": (_f("r3_in"), _f("r3_out")),
    }
    return {
        "ok": True,
        "net_amount": net,                       # 元
        "net_amount_wan": round(net / 1e4, 1),   # 万元
        "net_ratio": round(net_ratio, 3),        # 占成交额 %
        "tiers": {k: {"net": round(v[0] - v[1], 1), "in": v[0], "out": v[1]}
                  for k, v in tiers.items()},
        "turnover": _f("turnover"),
        "changeratio": _f("changeratio"),
        "curr_capital": _f("curr_capital"),      # 流通股本（万股）
    }


# ------------------------- yfinance 数据源（可选） -------------------------
def _yf_kline(symbol: str, datalen: int = 400) -> pd.DataFrame:
    import yfinance as yf  # 延迟导入，未安装时不影响主流程

    df = yf.download(symbol, period="2y", progress=False, auto_adjust=True)
    df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(datalen).reset_index(drop=True)


# ------------------------- 对外接口 -------------------------
def fetch_kline(symbol: str, datalen: int = 400, source: str = "sina") -> pd.DataFrame:
    """获取历史日 K 线，返回列：date, open, high, low, close, volume。"""
    if source == "yfinance":
        return _yf_kline(symbol, datalen)
    return _sina_kline(symbol, datalen)


def fetch_quote(symbol: str, source: str = "sina") -> dict:
    """获取实时行情快照。"""
    if source == "yfinance":
        return {}
    return _sina_quote(symbol)


@dataclass
class MarketData:
    symbol: str
    kline: pd.DataFrame
    quote: dict


def load(symbol: str, datalen: int = 400, source: str = "sina") -> MarketData:
    kline = fetch_kline(symbol, datalen, source)
    quote = fetch_quote(symbol, source)
    return MarketData(symbol=symbol, kline=kline, quote=quote)
