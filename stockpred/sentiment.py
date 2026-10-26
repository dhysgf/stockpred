"""中文新闻情感分析。

使用 snownlp 对每条新闻打分（0~1，越接近 1 越正面），
聚合为整体情绪分与标签。snownlp 不可用时退回到关键词词典。
"""
from __future__ import annotations

from .news_fetcher import NewsItem

_USE_SNOWNLP = True
try:
    from snownlp import SnowNLP
except Exception:  # pragma: no cover
    _USE_SNOWNLP = False

# 关键词词典兜底（snownlp 缺失时）
_POS = ["增长", "利好", "大涨", "上调", "盈利", "超预期", "复苏", "获批", "中标", "回购", "增持", "扩产"]
_NEG = ["下跌", "利空", "亏损", "下调", "暴雷", "减持", "退市", "处罚", "调查", "违约", "停产", "降级"]


def _score_one(text: str) -> float:
    if _USE_SNOWNLP:
        try:
            return float(SnowNLP(text).sentiments)
        except Exception:
            pass
    # 词典兜底
    pos = sum(t in text for t in _POS)
    neg = sum(t in text for t in _NEG)
    if pos + neg == 0:
        return 0.5
    return 0.5 + 0.5 * (pos - neg) / (pos + neg)


def analyze_news(news: list[NewsItem]) -> dict:
    """聚合新闻情绪。

    返回：mean(0~1), count, std, label(看涨/看跌/中性), detail(list of (title, score))
    """
    if not news:
        return {"mean": 0.5, "count": 0, "std": 0.0, "label": "中性(无新闻)", "detail": []}
    scores = [_score_one(n.title) for n in news]
    mean = sum(scores) / len(scores)
    std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
    if mean >= 0.55:
        label = "偏看涨"
    elif mean <= 0.45:
        label = "偏看跌"
    else:
        label = "中性"
    return {
        "mean": round(mean, 4),
        "count": len(scores),
        "std": round(std, 4),
        "label": label,
        "detail": list(zip([n.title for n in news], scores)),
    }
