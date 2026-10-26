"""把 run.py 的文本决策报告解析成结构化 JSON，供展示站渲染「完整分析」。

为什么不让网页自己跑模型：sklearn + PyTorch + 300 只股票池跑不动浏览器，
所以线上给的是**同一次真实运行的结构化产物**，不是我编的数字。

用法：python make_snapshot.py            # 解析 output/decision_report_*.txt
      python make_snapshot.py --out deploy_site/analysis_snapshot.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import make_report as R  # 复用已有的日志解析器，避免两套正则各说各话

OUT = Path("output")
TARGET = Path("deploy_site/analysis_snapshot.json")


def _f(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _header(text: str) -> dict:
    m = re.search(r"开始分析\s*(\S+?)\((\d+)\)：模型=(\S+)\s+特征=(\S+)\s+标签=(\S+)", text)
    d = {"name": m.group(1), "code": m.group(2), "model": m.group(3),
         "features": m.group(4), "label": m.group(5)} if m else {}
    m = re.search(r"历史\s*(\d+)\s*根日线，最新收盘\s*([\d.]+)（([\d-]+)）", text)
    if m:
        d.update(hist=int(m.group(1)), close=float(m.group(2)), asof=m.group(3))
    m = re.search(r"实时价\s*([\d.]+)\s+委比\s*(-?[\d.]+)%", text)
    if m:
        d.update(realtime=float(m.group(1)), weibi=float(m.group(2)))
    m = re.search(r"训练方式：(.+?)｜\s*模型\s*(\S+)（成员：([^）]+)）", text)
    if m:
        d.update(training=m.group(1).strip(), ensemble=m.group(2), members=m.group(3))
    m = re.search(r"耗时：(\{[^}]*\})", text)
    if m:
        try:
            d["timing"] = ast.literal_eval(m.group(1))
        except (ValueError, SyntaxError):
            pass
    return d


def _patterns(text: str) -> dict:
    m = re.search(r"\[技术面\]\s*K线形态：(\S+?)（形态分\s*([\d.]+)，多\s*([\d.]+)\s*/\s*空\s*([\d.]+)）", text)
    d = {"summary": m.group(1), "score": float(m.group(2)),
         "long": float(m.group(3)), "short": float(m.group(4)), "items": []} if m else {"items": []}
    body = R._section(text, "[技术面]", ("[量能]",))
    for ln in body.splitlines():
        mm = re.match(r"^(\d{4}-\d{2}-\d{2})\s+(\S+)\s+(\S+)\s+强度\s*([\d.]+)\s+(.*)$", ln.strip())
        if mm:
            d["items"].append({"date": mm.group(1), "name": mm.group(2),
                               "dir": mm.group(3), "strength": float(mm.group(4)),
                               "note": mm.group(5).strip()})
    return d


def _volume(text: str) -> dict:
    m = re.search(r"\[量能\]\s*(\S+?)（量能分\s*([\d.]+)）", text)
    d = {"summary": m.group(1), "score": float(m.group(2)), "items": [], "flow": {}} if m else {"items": []}
    body = R._section(text, "[量能]", ("[新闻政策]",))
    for ln in body.splitlines():
        s = ln.strip()
        mm = re.match(r"^(\S+?)\s{2,}(.+?)\s{2,}贡献\s*([+-][\d.]+)\s+(.*)$", s)
        if mm:
            d["items"].append({"name": mm.group(1), "value": mm.group(2).strip(),
                               "contrib": float(mm.group(3)), "note": mm.group(4).strip()})
            continue
        mm = re.match(r"^资金流分单：(.+)$", s)
        if mm:
            for part in re.findall(r"(超大单|大单|中单|小单)\s*([+-]?[\d.]+)万", mm.group(1)):
                d["flow"][part[0]] = float(part[1])
    return d


def _news(text: str) -> dict:
    m = re.search(r"\[新闻政策\]\s*(.+?)条", text)
    d = {"counts_raw": m.group(1).strip() if m else ""}
    m = re.search(r"大模型\s*(\S+)（([\d.]+)s，置信\s*([\d.]+)）立场\s*(\S+)\s*情绪\s*([\d.]+)", text)
    if m:
        d.update(model=m.group(1), secs=float(m.group(2)), confidence=float(m.group(3)),
                 stance=m.group(4), sentiment=float(m.group(5)))
    m = re.search(r"\[新闻政策\][\s\S]{0,400}?总结：(.+)", text)
    if m:
        d["summary"] = m.group(1).strip()
    d["drivers"], d["risks"] = [], []
    for ln in R._section(text, "[新闻政策]", ("===", "[" )).splitlines():
        s = ln.strip()
        if s.startswith("驱动｜"):
            d["drivers"].append(s.split("｜", 1)[1])
        elif s.startswith("风险｜"):
            d["risks"].append(s.split("｜", 1)[1])
    return d


def _turning(text: str) -> dict:
    d = {"rows": []}
    body = R._section(text, "[拐点预测]", ("[联合决策]",))
    for ln in body.splitlines():
        s = ln.strip()
        m = re.match(r"^(\d+)日：顶\s*([\d.]+)%（([\d.]+)%）\s*底\s*([\d.]+)%（([\d.]+)%）", s)
        if m:
            d["rows"].append({"h": int(m.group(1)), "top": float(m.group(2)), "top_base": float(m.group(3)),
                              "bot": float(m.group(4)), "bot_base": float(m.group(5))})
            continue
        m = re.match(r"^预计下一个(顶|底)拐点：约\s*(\d+)\s*个交易日后（区间\s*(\d+)~(\d+)\s*日）", s)
        if m:
            d["next_" + ("top" if m.group(1) == "顶" else "bot")] = {
                "days": int(m.group(2)), "lo": int(m.group(3)), "hi": int(m.group(4))}
    return d


def _decision(text: str) -> dict:
    m = re.search(r"\[联合决策\]\s*综合分\s*([+-][\d.]+)\s*→\s*(\S+?)（置信\s*([\d.]+)，一致度\s*(\d+)%）", text)
    d = {"score": float(m.group(1)), "verdict": m.group(2),
         "confidence": float(m.group(3)), "agreement": int(m.group(4))} if m else {}
    body = R._section(text, "[联合决策]", ("===", "["))
    note = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("[")]
    if note:
        d["note"] = note[0]
    return d


def _zones(text: str) -> list[dict]:
    body = R._section(text, "=== 目标价位区间 ===", ("[买卖点预判]",))
    out = []
    for parts in R._parse_targets(body):
        if len(parts) >= 9:
            out.append({"h": int(parts[0]), "date": parts[1], "lo": _f(parts[2]), "mid": _f(parts[3]),
                        "hi": _f(parts[4]), "exp": _f(parts[5]), "band": _f(parts[6]),
                        "prob": _f(parts[7]), "dir": parts[8]})
    return out


def _horizons(text: str) -> list[dict]:
    out = []
    for parts in R._parse_targets(text):
        if len(parts) >= 8:
            out.append({"h": int(parts[0]), "date": parts[1], "pred": _f(parts[2]), "rel": _f(parts[3]),
                        "prob": _f(parts[4]), "dir": parts[5], "lo": _f(parts[6]), "hi": _f(parts[7])})
    return out


def _trade_points(text: str) -> list[dict]:
    body = R._section(text, "[买卖点预判]", ("✓",))
    out, pending = [], None
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("└"):
            if pending is not None:
                pending["note"] = (pending.get("note", "") + s.lstrip("└ ").strip()).strip()
            continue
        m = re.match(r"^(.+?)：(.+)$", s)
        if m and ("风险等级" not in s):
            pending = {"label": m.group(1).strip(), "value": m.group(2).strip()}
            out.append(pending)
            continue
        m = re.match(r"^风险等级\s*(\S+)\s+ATR\s*([\d.]+%)\s+日波动\s*([\d.]+%)\s+带宽系数\s*z=([\d.]+)", s)
        if m:
            out.append({"label": "风险等级", "value": m.group(1),
                        "note": f"ATR {m.group(2)} · 日波动 {m.group(3)} · 带宽系数 z={m.group(4)}"})
            continue
        if s.startswith("·"):
            out.append({"label": "说明", "value": s.lstrip("· ").strip()})
    return out


def _llm_summary(text: str) -> dict:
    m = re.search(r"\[大模型综合研判\]\s*(\S+)（([\d.]+)s）\s*口径：(\S+)", text)
    d = {"model": m.group(1), "secs": float(m.group(2)), "tone": m.group(3)} if m else {}
    body = R._section(text, "[大模型综合研判]", ("[目标价位区间]",))
    d["supports"], d["warns"], d["watch"] = [], [], []
    for ln in body.splitlines():
        s = ln.strip()
        if s.startswith("支撑｜"):
            d["supports"].append(s.split("｜", 1)[1])
        elif s.startswith("警惕｜"):
            d["warns"].append(s.split("｜", 1)[1])
        elif s.startswith("盯盘｜"):
            d["watch"].append(s.split("｜", 1)[1])
        elif s and not s.startswith("[") and "注意：" not in s and "model" not in d:
            d["text"] = s
        elif s and not s.startswith("[") and "注意：" not in s and "text" not in d:
            d["text"] = s
    return d


def _llm_forecast(text: str) -> dict:
    """解析大模型独立价格预测 + 条件规则预判。"""
    body = R._section(text, "[大模型独立预测]", ("[目标价位区间]", "[买卖点预判]"))
    d = {"forecasts": [], "rules": [], "reasons": [], "risks": []}
    if not body:
        return d
    m = re.search(r"\[大模型独立预测\]\s*(\S+)（([\d.]+)s）\s*方向\s*(\S+?)（置信\s*([\d.]+)）", text)
    if m:
        d.update({"model": m.group(1), "secs": float(m.group(2)),
                  "direction": m.group(3), "confidence": float(m.group(4))})
    for ln in body.splitlines():
        s = ln.strip()
        m = re.match(r"预测\s*(\d+)日\s*([+-][\d.]+)%\s*目标\s*([\d.]+|—)\s*"
                     r"收涨概率\s*(\d+)%\s*\|\s*(.*)", s)
        if m:
            d["forecasts"].append({
                "horizon": int(m.group(1)), "pct": float(m.group(2)),
                "price": None if m.group(3) == "—" else float(m.group(3)),
                "prob_up": int(m.group(4)), "reason": m.group(5).strip()})
            continue
        m = re.match(r"规则\s*若\s*(.+?)\s*→\s*(.+?)(?:（(.*)）)?$", s)
        if m:
            d["rules"].append({"trigger": m.group(1).strip(), "action": m.group(2).strip(),
                               "why": (m.group(3) or "").strip()})
            continue
        m = re.match(r"理由\s*(.+)$", s)
        if m:
            d["reasons"].append(m.group(1).strip()); continue
        m = re.match(r"风险\s*(.+)$", s)
        if m:
            d["risks"].append(m.group(1).strip()); continue
        m = re.match(r"总结\s*(.+)$", s)
        if m:
            d["summary"] = m.group(1).strip()
    return d


def build(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8")
    d = _header(text)
    d["patterns"] = _patterns(text)
    d["volume"] = _volume(text)
    d["news"] = _news(text)
    d["horizons"] = _horizons(text)
    d["turning"] = _turning(text)
    d["decision"] = _decision(text)
    d["signals"] = R._parse_signals(text)
    d["llm_summary"] = _llm_summary(text)
    d["llm_forecast"] = _llm_forecast(text)
    d["zones"] = _zones(text)
    d["trade_points"] = _trade_points(text)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(TARGET))
    a = ap.parse_args()

    snap = {}
    for p in sorted(OUT.glob("decision_report_*.txt")):
        code = re.search(r"(\d{6})", p.stem).group(1)
        try:
            snap[code] = build(p)
            if str(snap[code].get("name", "")).isdigit() or not snap[code].get("name"):
                snap[code]["name"] = R._fallback_name(code) or code
            n_sig = len(snap[code]["signals"])
            n_zone = len(snap[code]["zones"])
            print(f"  ✓ {code} {snap[code].get('name')} 信号 {n_sig} 条 / 区间 {n_zone} 档")
        except Exception as e:  # 单只失败不影响其它
            print(f"  ✗ {code} 解析失败：{type(e).__name__}: {e}")

    out = Path(a.out)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"已写出 {out}（{out.stat().st_size/1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
