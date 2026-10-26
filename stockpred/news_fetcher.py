"""新闻抓取层：个股新闻（东财搜索）+ 政策/市场新闻（新浪滚动 + 东财搜索）。

数据源实测（2026-10-21）
------------------------
- ✅ 东财搜索 API（``search-api-web.eastmoney.com``）：**支持按关键词检索**，
  可以拿到真正的个股新闻（茅台 3914 条），返回 JSONP，需剥 ``x(...)`` 外壳。
- ✅ 新浪财经滚动资讯（``feed.mix.sina.com.cn``）：政策/市场层面头条，不能按个股过滤。
- ⚠️ tushare：需 token，保留钩子但不作为默认路径。
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass

UA = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"}
_TAG_RE = re.compile(r"<[^>]+>")


def _http_get(url: str, timeout: int = 15, encoding: str = "utf-8") -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(encoding, "ignore")


@dataclass
class NewsItem:
    title: str
    source: str = ""
    time: str = ""
    url: str = ""
    content: str = ""
    kind: str = "market"      # market / stock / policy

    def brief(self, maxlen: int = 160) -> str:
        """给大模型用的紧凑文本：标题 + 摘要。"""
        body = _TAG_RE.sub("", self.content or "").strip()
        s = self.title if not body else f"{self.title}｜{body}"
        return s[:maxlen]


def _clean(s: str) -> str:
    return _TAG_RE.sub("", s or "").replace("\u3000", " ").strip()


# ------------------------- 东财搜索（可按关键词） -------------------------
def _eastmoney_search(keyword: str, num: int = 20) -> list[NewsItem]:
    param = {
        "uid": "", "keyword": keyword, "type": ["cmsArticleWebOld"],
        "client": "web", "clientType": "web", "clientVersion": "curr",
        "param": {"cmsArticleWebOld": {
            "searchScope": "default", "sort": "time", "pageIndex": 1, "pageSize": num,
            "preTag": "<em>", "postTag": "</em>",
        }},
    }
    url = ("https://search-api-web.eastmoney.com/search/jsonp?cb=x&param="
           + urllib.parse.quote(json.dumps(param, ensure_ascii=False)))
    raw = _http_get(url)
    # 剥 JSONP：x({...}) 或 x({...});
    raw = raw.strip()
    if raw.startswith("x("):
        raw = raw[2:]
    if raw.endswith(");"):
        raw = raw[:-2]
    elif raw.endswith(")"):
        raw = raw[:-1]
    j = json.loads(raw)
    arr = ((j.get("result") or {}).get("cmsArticleWebOld") or [])
    out = []
    for x in arr:
        title = _clean(x.get("title", ""))
        if not title:
            continue
        out.append(NewsItem(
            title=title, source=_clean(x.get("mediaName", "")) or "东方财富",
            time=str(x.get("date", "")), url=x.get("url", ""),
            content=_clean(x.get("content", "")), kind="stock",
        ))
    return out


def fetch_stock_news(symbol: str, name: str = "", num: int = 20, token: str = "") -> list[NewsItem]:
    """个股公司新闻：优先东财关键词搜索（真实个股相关），失败再退回名称过滤。"""
    if token:
        try:
            import tushare as ts

            ts.set_token(token)
            pro = ts.pro_api()
            df = pro.news(src="sina", count=num)
            out = [NewsItem(title=r["title"], source="tushare", kind="stock") for _, r in df.iterrows()]
            if out:
                return out
        except Exception as e:
            print(f"[news] tushare 个股新闻失败：{e}")

    kw = (name or "").strip() or str(symbol).strip()
    try:
        items = _eastmoney_search(kw, num=num)
        for it in items:
            it.kind = "stock"
        if items:
            return items
    except Exception as e:
        print(f"[news] 东财个股新闻失败：{e}")

    # 最后退路：市场新闻里按名称命中过滤
    market = fetch_market_news(num=80)
    if not kw:
        return []
    return [n for n in market if kw in n.title][:num]


# ------------------------- 政策 / 市场新闻 -------------------------
def fetch_market_news(num: int = 40, lid: int = 2513) -> list[NewsItem]:
    """新浪财经滚动资讯（政策/市场层面头条）。lid=2513 财经。"""
    url = f"https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid={lid}&num={num}&r=0"
    try:
        j = json.loads(_http_get(url))
        data = (j.get("result") or {}).get("data") or []
        out = []
        for x in data:
            title = _clean(x.get("title", ""))
            if title:
                out.append(NewsItem(title=title, source="新浪财经",
                                    time=str(x.get("ctime", "")), url=x.get("url", ""),
                                    kind="market"))
            if len(out) >= num:
                break
        return out
    except Exception as e:
        print(f"[news] 市场新闻抓取失败：{e}")
        return []


def fetch_policy_news(num: int = 15) -> list[NewsItem]:
    """政策类新闻：东财检索「货币政策/产业政策/监管」等关键词（失败则返回空）。"""
    out, seen = [], set()
    for kw in ("货币政策", "财政政策", "产业政策"):
        try:
            for it in _eastmoney_search(kw, num=max(4, num // 3)):
                if it.title in seen:
                    continue
                seen.add(it.title)
                it.kind = "policy"
                out.append(it)
                if len(out) >= num:
                    return out
        except Exception as e:
            print(f"[news] 政策新闻({kw})失败：{e}")
    return out


def collect_news(symbol: str, name: str = "", stock_num: int = 20,
                 market_num: int = 25, policy_num: int = 12) -> dict:
    """汇总三类新闻并去重。返回 {"stock": [...], "policy": [...], "market": [...]}。"""
    stock = fetch_stock_news(symbol, name=name, num=stock_num)
    policy = fetch_policy_news(num=policy_num)
    market = fetch_market_news(num=market_num)
    # 去重（保持优先级：个股 > 政策 > 市场）
    def _dedup(lst):
        out, s = [], set()
        for it in lst:
            if it.title in s:
                continue
            s.add(it.title)
            out.append(it)
        return out
    return {"stock": _dedup(stock), "policy": _dedup(policy), "market": _dedup(market)}
