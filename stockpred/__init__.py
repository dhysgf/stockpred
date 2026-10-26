"""stockpred 包：股价走势预测工具（价格 / 方向 / 拐点 + 多信号联合决策）。"""
from .data_fetcher import load, fetch_kline, fetch_quote, fetch_money_flow, normalize_symbol
from .news_fetcher import fetch_market_news, fetch_stock_news, fetch_policy_news, collect_news
from .sentiment import analyze_news
from .patterns import add_patterns, detect_patterns, pattern_summary
from .volume import add_volume_indicators, volume_score, depth_stats, obv_divergence
from .features import add_technicals, build_horizon_samples, enrich_with_index, FEATURES
from .model import ForecastModel, forecast
from .models import (build_forecaster, RFForecaster, GBDTForecaster, LSTMForecaster,
                     EnsembleForecaster)
from .alpha import add_alpha158_lite, alpha_columns
from .strategy import walk_forward_scores, backtest_topk, score_ic
from .universe import get_universe, pool_label
from .turning_point import (turning_labels, recent_turns, build_turn_samples,
                            TurningPointModel, eval_turning)
from .decision import decide, eval_signals
from .analysis import analyze
from .backtest import walk_forward_backtest
from .visualize import plot_forecast, plot_backtest, plot_candles, plot_equity

__all__ = [
    "load", "fetch_kline", "fetch_quote", "fetch_money_flow", "normalize_symbol",
    "fetch_market_news", "fetch_stock_news", "fetch_policy_news", "collect_news",
    "analyze_news",
    "add_patterns", "detect_patterns", "pattern_summary",
    "add_volume_indicators", "volume_score", "depth_stats", "obv_divergence",
    "add_technicals", "build_horizon_samples", "enrich_with_index", "FEATURES",
    "ForecastModel", "forecast",
    "build_forecaster", "RFForecaster", "GBDTForecaster", "LSTMForecaster",
    "EnsembleForecaster", "add_alpha158_lite", "alpha_columns",
    "walk_forward_scores", "backtest_topk",
    "turning_labels", "recent_turns", "build_turn_samples", "TurningPointModel", "eval_turning",
    "decide", "eval_signals", "analyze",
    "walk_forward_backtest", "plot_forecast", "plot_backtest", "plot_candles", "plot_equity",
]
