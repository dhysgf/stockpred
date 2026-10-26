"""把 CLI 的文本报告 + 预测图/K线图 合成一份**小白也能看懂**的 HTML 报告。

用法：
    python make_report.py --symbol 600519 --run          # 跑分析 + 出报告（全量）
    python make_report.py --symbol 600519 --run --extra "--pool-limit 40"   # 快速版
    python make_report.py --symbol 600519               # 只重排版已有文本报告

报告结构（小白优先）
--------------------
1. 一句话结论        —— 现在到底该不该动
2. 现在该怎么做      —— 具体价位清单（买在哪、卖在哪、跌到哪走）
3. 未来能到什么价    —— 5/7/30 天价格范围，用大白话解释"范围"是什么意思
4. 为什么这么判断    —— 六路信号翻译成人话
5. 这套系统有多可信  —— 诚实说明哪些部分有统计支持、哪些只是参考
6. 图 + 专业明细     —— 想深究再展开

设计原则：**先给结论和动作，再给依据，最后给细节**。
内行看的东西（原始信号表、大模型研判全文）折叠在底部，不干扰阅读。
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import re
from pathlib import Path

OUT = Path("output")

# 信号名 → 人话解释（小白不知道"量能""拐点"是什么）
SIGNAL_PLAIN = {
    "模型方向概率": "三种 AI 模型（随机森林 / 梯度提升 / 神经网络）一起投票，猜未来几天涨还是跌",
    "K线形态": "看最近几根 K 线的形状（比如三连阳、流星线）",
    "量能与资金": "看成交量和主力大资金是流入还是流出",
    "大模型新闻研判": "让千问读最近的新闻，判断消息面偏好还是偏差",
    "大模型政策研判": "让千问判断政策 / 监管对这家公司是利还是弊",
    "拐点风险": "判断近期会不会见顶（该卖）或见底（该买）",
}

VERDICT_PLAIN = {
    "强烈买入": ("可以考虑买入", "多个信号一起指向上涨", "up"),
    "买入": ("可以考虑买入", "偏多的信号占上风", "up"),
    "买入/增持": ("可以考虑买一点", "偏多的信号略占上风", "up"),
    "增持": ("可以小幅加仓", "偏多信号占优", "up"),
    "观望": ("先别动", "看涨和看跌的力量基本抵消，现在买卖都不划算", "neutral"),
    "减持/卖出": ("建议减一点", "偏空的信号占上风", "down"),
    "强烈卖出": ("建议卖出 / 回避", "多个信号一起指向下跌", "down"),
    "卖出": ("建议卖出", "偏空信号占优", "down"),
}


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii") if path.exists() else ""


def _img_block(path: Path, title: str, note: str = "") -> str:
    if not path.exists():
        return ""
    return (f"<div class='card'><h3>{html.escape(title)}</h3>"
            f"<img src='data:image/png;base64,{_b64(path)}' alt='{html.escape(title)}'>"
            + (f"<p class='muted'>{html.escape(note)}</p>" if note else "") + "</div>")


# ------------------------------------------------------------------ 解析
def _nums(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s or "")]


def _parse_targets(text: str) -> list[list[str]]:
    rows, in_tbl = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("周期(日)"):
            in_tbl = True
            continue
        if in_tbl:
            if not s or not re.match(r"^\d", s):
                if rows:
                    break
                continue
            parts = s.split()
            if len(parts) >= 8:
                rows.append(parts)
    return rows


def _parse_zones(text: str) -> list[list[str]]:
    """解析「=== 目标价位区间 ===」那张表（9 列：周期/到达日/下界/中位/上界/预期/带宽/概率/方向）。

    注意：日志里其实有**两张**表 ——「各周期目标价」（8 列，模型直接回归出来的预测收盘价）
    和「目标价位区间」（9 列，带宽/中位/上界）。之前只解析了第一张却按第二张的列序读，
    结果整张表读空，报告里"未来能到什么价"是空白。必须分开解析。
    """
    rows, in_tbl = [], False
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("=== 目标价位区间 ==="):
            in_tbl = True
            continue
        if in_tbl:
            if not st or not re.match(r"^\d", st):
                if rows:
                    break
                continue
            parts = st.split()
            if len(parts) >= 9:
                rows.append(parts)
    return rows


def _parse_signals(text: str) -> list[dict]:
    """解析各信号贡献表（信号名不含空格，后接 3 个数字 + 依据）。"""
    out = []
    for line in _section(text, "=== 各信号贡献明细 ===", ("[大模型综合研判]",)).splitlines():
        s = line.strip()
        m = re.match(r"^(\S+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)\s*(.*)$", s)
        if m:
            out.append({"name": m.group(1), "value": float(m.group(2)),
                        "weight": float(m.group(3)), "contrib": float(m.group(4)),
                        "note": m.group(5).strip()})
    return out


def _parse_kv(text: str, start: str, keys: tuple[str, ...]) -> dict[str, str]:
    out, on = {}, False
    for line in text.splitlines():
        s = line.strip()
        if start in s:
            on = True
            continue
        if on:
            if s.startswith("[") and "买卖点" not in s:
                break
            for k in keys:
                if s.startswith(k):
                    out[k] = s.split("：", 1)[-1].strip()
    return out


def _verdict(text: str) -> tuple[str, str]:
    m = re.search(r"\[联合决策\]\s*综合分\s*([+-][\d.]+)\s*→\s*(\S+?)（", text)
    return (m.group(2), m.group(1)) if m else ("观望", "0")


def _section(text: str, head: str, end_heads: tuple[str, ...]) -> str:
    lines, on, buf = text.splitlines(), False, []
    for ln in lines:
        s = ln.strip()
        if s.startswith(head):
            on = True
            continue
        if on and s.startswith("[") and any(s.startswith(e) for e in end_heads):
            break
        if on:
            buf.append(ln)
    return "\n".join(buf).strip("\n")


# ------------------------------------------------------------------ 人话翻译
def _band_words(band: float) -> str:
    if band <= 3:
        return "范围比较窄，模型相对有把握"
    if band <= 6:
        return "范围中等"
    return "范围很宽，说明不确定性大，别重仓"


def _dir_words(v: float) -> tuple[str, str]:
    if v > 0.10:
        return "偏多（利好）", "up"
    if v < -0.10:
        return "偏空（利空）", "down"
    return "中性（没明显方向）", "neu"


def _ret_words(exp: float) -> str:
    if abs(exp) < 0.5:
        return "跟现在差不多"
    return f"比收盘价{'高' if exp > 0 else '低'} {abs(exp):.2f}%"


def _fallback_name(code: str) -> str:
    """从实时行情补全股票简称；失败返回空串。"""
    try:
        from stockpred.data_fetcher import fetch_quote
        nm = ((fetch_quote(code) or {}).get("name") or "").strip().replace(" ", "")
        return "" if (not nm or nm.isdigit()) else nm
    except Exception:
        return ""


def build(symbol: str, log_path: Path) -> Path:
    text = log_path.read_text(encoding="utf-8")
    code = re.sub(r"\D", "", symbol)[-6:]

    m = re.search(r"开始分析\s*(\S+?)\((\d+)\)：模型=(\S+)", text)
    name, model = (m.group(1), m.group(3)) if m else ("", "—")
    # 老日志里 --name 没传时，这里会拿到纯代码（如「000001」），标题就成了
    # 「000001（000001）」。拿不到真简称就回退查一次实时行情，仍失败才用代码兜底。
    if not name or name.isdigit():
        name = _fallback_name(code) or code
    # 实在查不到简称时标题只写代码，别显示成「000001（000001）」
    disp = "" if name == code else name
    m2 = re.search(r"历史\s*(\d+)\s*根日线，最新收盘\s*([\d.]+)（([\d-]+)）", text)
    n_hist, close, asof = (m2.group(1), float(m2.group(2)), m2.group(3)) if m2 else ("", 0.0, "—")
    m3 = re.search(r"实时价\s*([\d.]+)", text)
    price = float(m3.group(1)) if m3 else close

    # ---- 数据新鲜度横幅：报告的数据日 vs 行情源最新交易日 / 今天
    # 用户最容易误解的就是"数据截至 X"——以为那就是今天。
    # 这里直接告诉他：数据截止到哪天、今天已经过去几天了、为什么。
    stale_banner = ""
    try:
        from stockpred.pool import _latest_market_day
        latest_day = _latest_market_day()
    except Exception:
        latest_day = ""
    today_str = dt.datetime.now().strftime("%Y-%m-%d")
    if asof and asof != "—" and latest_day and asof < latest_day:
        gap_days = (dt.date.fromisoformat(latest_day) - dt.date.fromisoformat(asof)).days
        stale_banner = (
            f'<div class="card" style="border-color:#f59e0b;background:#fffbeb;'
            f'margin-top:12px;padding:12px 16px;font-size:14px">'
            f'<b>⚠ 这份报告的数据截至 {html.escape(asof)}，不是最新交易日。</b><br>'
            f'行情源目前最新一根日线是 <b>{html.escape(latest_day)}</b>，'
            f'两者相差 {gap_days} 天（{"跨周末" if gap_days >= 2 else "隔了一个交易日"}）。'
            f'所有预测、目标价、买卖点都是基于 {html.escape(asof)} 收盘价算的，'
            f'期间的价格变化没有计入。'
            f'<a href="/report?code={code}&refresh=1" style="color:#2563eb">→ 重新分析这只</a>'
            f'</div>')
    elif asof and asof != "—" and today_str > asof and latest_day and asof == latest_day:
        stale_banner = (
            f'<div class="card" style="border-color:#dbeafe;background:#f5f9fe;'
            f'margin-top:12px;padding:12px 16px;font-size:14px">'
            f'<b>数据截至 {html.escape(asof)}（{html.escape(latest_day)} 收盘）</b>，'
            f'这是目前能拿到的最新一根日线。'
            f'如果今天是交易日，说明<b>当天还没收盘</b>，收盘后重跑可得到更新数据。'
            f'</div>')

    verdict, score = _verdict(text)
    v_title, v_why, v_cls = VERDICT_PLAIN.get(verdict, ("先别动", "信号不明确", "neutral"))

    rows = _parse_targets(text)          # 各周期目标价（8 列）
    zrows = _parse_zones(text)           # 目标价位区间（9 列）
    sigs = _parse_signals(text)
    kv = _parse_kv(text, "[买卖点预判]", (
        "分批建仓区（低吸）", "突破确认价（追强）", "短期强支撑",
        "MA20 动态", "止盈减仓区", "止损位", "ATR 参考止损"))

    # ---- 价位（取每组第一个数字）
    buy_lo = buy_hi = brk = sup = ma20 = tp_lo = tp_hi = stop = None
    if "分批建仓区（低吸）" in kv:
        n = _nums(kv["分批建仓区（低吸）"])
        buy_lo, buy_hi = (n + [None, None])[:2]
    if "突破确认价（追强）" in kv:
        brk = (_nums(kv["突破确认价（追强）"]) or [None])[0]
    if "短期强支撑" in kv:
        sup = (_nums(kv["短期强支撑"]) or [None])[0]
    if "MA20 动态支撑" in kv:
        ma20 = (_nums(kv["MA20 动态支撑"]) or [None])[0]
    if "止盈减仓区" in kv:
        n = _nums(kv["止盈减仓区"])
        tp_lo, tp_hi = (n + [None, None])[:2]
    if "止损位" in kv:
        stop = (_nums(kv["止损位"]) or [None])[0]

    # 盘中价与收盘价可能不同：所有价位都以**最新收盘价**为基准计算，
    # 盘中价只用来判断"现在这个价在区间的什么位置"。不区分清楚小白会看到两个"现价"。
    gap_pct = ((price / close - 1) * 100) if close else 0.0
    price_note = ""
    if abs(gap_pct) >= 0.5:
        price_note = (f"（说明：下面所有价位都以<b>最新收盘价 {close:.2f}</b> 为基准算出来的；"
                      f"当前盘中价 {price:.2f}，"
                      f"{'比收盘低' if gap_pct < 0 else '比收盘高'} {abs(gap_pct):.2f}%。"
                      f"判断「现在该不该买」用的是盘中价。）")

    # 买入区上沿按构造就是基准价（区间 = 5日下界 ~ 收盘价），所以不能用"是否落在区间内"判断，
    # 必须看现价在区间里的**相对位置**：越靠下沿越划算，靠上沿等于追高。
    pos_pct = None
    if buy_lo is not None and buy_hi is not None and buy_hi > buy_lo:
        pos_pct = (price - buy_lo) / (buy_hi - buy_lo)

    if pos_pct is None:
        pos_fact, buy_hint = "", "具体操作见下面的价位清单"
    elif pos_pct < 0.25:
        pos_fact = f"现价 {price:.2f} 已接近低吸区<b>下沿</b>（{buy_lo:.2f} 元），位置不错"
        buy_hint = "要建仓的话，这个位置算划算，可以分批买"
    elif pos_pct < 0.75:
        pos_fact = (f"现价 {price:.2f} 在低吸区<b>中段</b>（{buy_lo:.2f} ~ {buy_hi:.2f} 元）")
        buy_hint = "要建仓的话位置中等，可以分几次小批量买"
    else:
        pos_fact = (f"现价 {price:.2f} 已在低吸区<b>上沿</b>（{buy_lo:.2f} ~ {buy_hi:.2f} 元）")
        buy_hint = f"这个时候买等于追高，等回落到 {buy_lo:.2f} 附近再说"

    # 操作基调必须和"一句话结论"一致，否则小白会看到"先别动"又说"可以买"而困惑
    if verdict in ("强烈买入", "买入", "买入/增持", "增持"):
        now_act = f"{buy_hint}。"
    elif verdict in ("强烈卖出", "卖出", "减持/卖出"):
        now_act = f"系统偏空，<b>不建议现在买入</b>；已持有的可以考虑逢高减仓。{pos_fact}。"
    else:  # 观望
        now_act = (f"方向不明，<b>不建议重仓押方向</b>。{pos_fact}；{buy_hint}"
                   f"（真要试也只适合小仓位）。")
        if brk:
            now_act += f" 如果它放量站上 {brk:.2f} 元，才说明趋势转强。"

    # 止盈区下沿同样按构造等于现价，必须真的涨上去 2% 才提示止盈，否则等于"原地止盈"
    in_tp = bool(tp_lo and price >= tp_lo * 1.02)
    if in_tp:
        now_act += f" 另外，当前价已比止盈区下沿高，持有多的话可以考虑先减一部分。"

    # ---- 目标区间（人话）：用「目标价位区间」表（9 列）
    trows_pro, trows_plain = "", ""
    for r in zrows:
        try:
            h, date, lo, mid, hi = int(r[0]), r[1], float(r[2]), float(r[3]), float(r[4])
            exp, band, prob = float(r[5]), float(r[6]), float(r[7])
            direction = r[8] if len(r) > 8 else ""
        except (ValueError, IndexError):
            continue
        cls = "up" if exp > 0 else ("down" if exp < 0 else "")
        trows_pro += (f"<tr><td>{h} 天</td><td>{date}</td><td>{lo:.2f}</td>"
                      f"<td class='hi'>{mid:.2f}</td><td>{hi:.2f}</td>"
                      f"<td class='{cls}'>{exp:+.2f}%</td><td>±{band:.2f}%</td>"
                      f"<td>{prob*100:.0f}%</td></tr>")
        skill = ("<span class='warn'>该周期模型判断涨跌基本靠猜（实测 48.3%，跟抛硬币差不多），"
                 "所以只给价格范围、不给方向</span>" if h == 30 else "")
        trows_plain += (
            f"<div class='hz'>"
            f"<div class='hz-h'>{h} 个交易日后 <span class='muted'>（约 {date}）</span></div>"
            f"<div class='hz-b'>价格大概率在 <b>{lo:.2f} ~ {hi:.2f} 元</b>之间，"
            f"最可能在 <b>{mid:.2f} 元</b>附近 —— {_ret_words(exp)}。</div>"
            f"<div class='muted'>上下各 ±{band:.2f}%（{_band_words(band)}）；"
            f"模型认为上涨概率 {prob*100:.0f}%。{skill}</div></div>")


    # ---- 核心数据一览：各周期目标价（8 列，模型直接回归的预测收盘价）
    core_rows = ""
    for r in rows:
        try:
            h, date = int(r[0]), r[1]
            pred, rel, prob = float(r[2]), float(r[3]), float(r[4])
            direction = r[5] if len(r) > 5 else ""
            lo, hi = float(r[6]), float(r[7])
        except (ValueError, IndexError):
            continue
        cls = "up" if rel > 0 else ("down" if rel < 0 else "")
        dcls = "up" if direction == "偏多" else ("down" if direction == "偏空" else "")
        core_rows += (f"<tr><td><b>{h} 日</b></td><td>{date}</td>"
                      f"<td class='hi'>{pred:.2f}</td>"
                      f"<td class='{cls}'>{rel:+.2f}%</td>"
                      f"<td>{prob*100:.1f}%</td>"
                      f"<td class='{dcls}'>{html.escape(direction)}</td>"
                      f"<td>{lo:.2f} ~ {hi:.2f}</td></tr>")


    # ---- 大模型对新闻/政策的研判（这两路真的进了投票，各占 10%）
    n_stance, n_senti, n_conf, n_model, n_secs = "—", None, None, "—", None
    m = re.search(r"大模型\s*(\S+)（([\d.]+)s，置信\s*([\d.]+)）立场\s*(\S+)\s*情绪\s*([\d.]+)", text)
    if m:
        n_model = m.group(1)
        n_secs = float(m.group(2))
        n_conf = float(m.group(3))
        n_stance = m.group(4)
        n_senti = float(m.group(5))
    m = re.search(r"\[新闻政策\]\s*(.+?)\s*条", text)
    n_counts = m.group(1) if m else ""
    m = re.search(r"\[新闻政策\][\s\S]{0,800}?总结：(.+)", text)
    n_summary = m.group(1).strip() if m else ""
    n_drivers, n_risks = [], []
    for ln in _section(text, "[新闻政策]", ("===", "[")).splitlines():
        st = ln.strip()
        if st.startswith("驱动｜"):
            n_drivers.append(st.split("｜", 1)[1])
        elif st.startswith("风险｜"):
            n_risks.append(st.split("｜", 1)[1])
    n_policy = None
    pol_sig = next((sg for sg in sigs if sg["name"] == "大模型政策研判"), None)
    if pol_sig:
        mm = re.search(r"([+-]?\d+\.\d+)", pol_sig.get("note", ""))
        if mm:
            n_policy = float(mm.group(1))
    scls = "up" if n_stance == "偏多" else ("down" if n_stance == "偏空" else "")
    _parts = [
        "<div class='kv3'>",
        f"<div><span>立场</span><b class='{scls}'>{html.escape(n_stance)}</b></div>",
        f"<div><span>新闻情绪（0~1）</span><b>{('%.2f' % n_senti) if n_senti is not None else '—'}</b></div>",
        f"<div><span>政策影响（−1~1）</span><b>{('%+.2f' % n_policy) if n_policy is not None else '—'}</b></div>",
        f"<div><span>置信度</span><b>{('%.2f' % n_conf) if n_conf is not None else '—'}</b></div>",
        "</div>",
        f"<p class='muted'>模型：{html.escape(n_model)}　耗时 {('%.1f' % n_secs) if n_secs is not None else '—'} 秒"
        f"　材料：{html.escape(n_counts)} 条</p>",
        f"<p><b>研判结论：</b>{html.escape(n_summary)}</p>",
    ]
    if n_drivers:
        _parts.append("<h3>关键驱动</h3><ul class='lm-list'>")
        _parts += [f"<li><span class='up'>＋</span>{html.escape(x)}</li>" for x in n_drivers]
        _parts.append("</ul>")
    if n_risks:
        _parts.append("<h3>风险</h3><ul class='lm-list'>")
        _parts += [f"<li><span class='down'>－</span>{html.escape(x)}</li>" for x in n_risks]
        _parts.append("</ul>")
    _parts.append(
        "<p class='muted'>这两路（新闻情绪 0.10 + 政策影响 0.10）是<b>真的进了加权投票</b>的，"
        "不是只写在报告里好看。但新闻的影响<b>无法回测</b>（历史上没存下当时的新闻文本），"
        "只能算「有信息价值」，不能算「已验证有效」。</p>")
    news_card = "\n".join(_parts)

    # ---- 信号（人话）
    srows = ""
    for s in sorted(sigs, key=lambda x: -abs(x["contrib"])):
        if s["name"] == "模型方向概率":
            continue
        words, cls = _dir_words(s["value"])
        plain = SIGNAL_PLAIN.get(s["name"], "")
        srows += (f"<tr><td>{html.escape(s['name'])}</td>"
                  f"<td class='{cls}'>{words}</td>"
                  f"<td>{s['weight']:.0%}</td>"
                  f"<td style='text-align:left'>{html.escape(plain)}"
                  f"<div class='muted'>依据：{html.escape(s['note'])}</div></td></tr>")

    model_sig = next((s for s in sigs if s["name"] == "模型方向概率"), None)
    model_plain = ""
    if model_sig:
        words, cls = _dir_words(model_sig["value"])
        model_plain = (f"<tr><td>模型方向概率</td><td class='{cls}'>{words}</td>"
                       f"<td>{model_sig['weight']:.0%}</td>"
                       f"<td style='text-align:left'>{html.escape(SIGNAL_PLAIN['模型方向概率'])}"
                       f"<div class='muted'>依据：{html.escape(model_sig['note'])}</div></td></tr>")

    # 谁在主导（白话）
    lead = sorted(sigs, key=lambda x: -abs(x["contrib"]))[:2]
    lead_txt = "；".join(
        f"<b>{html.escape(s['name'])}</b>（{_dir_words(s['value'])[0]}，影响 {s['contrib']:+.3f}）"
        for s in lead) or "—"

    dec_raw = _section(text, "=== 各信号贡献明细 ===", ("[大模型综合研判]",))
    llm_txt = _section(text, "[大模型综合研判]", ("[目标价位区间]",))
    notes = [l.strip(" ·") for l in text.splitlines() if l.strip().startswith("·")]

    # ---- 专业明细①：信号贡献表（带双向条形图，比 raw 文本好读）
    def _bar(v: float) -> str:
        pct = min(abs(v), 1.0) * 50
        color = "#c0392b" if v > 0 else ("#1e7f43" if v < 0 else "#c3cbd4")
        side = f"left:50%;width:{pct:.1f}%" if v >= 0 else f"right:50%;width:{pct:.1f}%"
        return f"<span class='b2'><i style='{side};background:{color}'></i></span>"

    sig_rows = ""
    max_c = max((abs(s["contrib"]) for s in sigs), default=0) or 1.0
    for s in sorted(sigs, key=lambda x: -abs(x["contrib"])):
        words, cls = _dir_words(s["value"])
        pct = abs(s["contrib"]) / max_c * 100
        sig_rows += (
            f"<tr><td style='text-align:left'><b>{html.escape(s['name'])}</b>"
            f"<div class='muted'>{html.escape(SIGNAL_PLAIN.get(s['name'], ''))}</div></td>"
            f"<td>{_bar(s['value'])}<span class='v'>{s['value']:+.3f}</span></td>"
            f"<td>{s['weight']:.0%}</td>"
            f"<td class='{'up' if s['contrib'] > 0 else ('down' if s['contrib'] < 0 else '')}'>"
            f"{s['contrib']:+.4f}"
            f"<div class='cbar'><i style='width:{pct:.1f}%;background:"
            f"{'#c0392b' if s['contrib'] > 0 else '#1e7f43'}'></i></div></td>"
            f"<td style='text-align:left' class='muted'>{html.escape(s['note'])}</td></tr>")

    # ---- 专业明细②：千问研判结构化（结论 + 支撑/警惕/盯盘）
    def _pick(tag: str) -> list[str]:
        return [m.group(1).strip() for m in
                re.finditer(rf"{tag}｜(.+)", llm_txt or "")]

    # 元信息要从**原文**里搜：_section() 遇到标题行会 continue 掉，
    # 标题不在截取结果里，从那里搜永远匹配不到（表现为报告里元信息是空的）。
    m_head = re.search(r"\[大模型综合研判\]\s*(\S+)\s*（([\d.]+)s）\s*口径：(\S+)", text)
    llm_meta = (f"{m_head.group(1)}　耗时 {m_head.group(2)}s　口径 {m_head.group(3)}"
                if m_head else "")
    concl = ""
    for ln in (llm_txt or "").splitlines():
        s = ln.strip()
        if s and not s.startswith("[") and not s.startswith("注意") and "｜" not in s:
            concl = s
            break
    supports, warns, watches = _pick("支撑"), _pick("警惕"), _pick("盯盘")

    def _ul(items, cls, icon):
        if not items:
            return "<div class='muted'>（无）</div>"
        return "".join(f"<li><span class='{cls}'>{icon}</span> {html.escape(x)}</li>" for x in items)

    llm_html = ""
    if concl or supports or warns or watches:
        llm_html = f"""
        <div class="lm-head">{html.escape(llm_meta)}</div>
        <p class="lm-concl">{html.escape(concl)}</p>
        <div class="grid2">
          <div><h4 class="h-up">支撑结论</h4><ul class="lm-list">{_ul(supports, 'up', '＋')}</ul></div>
          <div><h4 class="h-down">需要警惕</h4><ul class="lm-list">{_ul(warns, 'down', '－')}</ul></div>
        </div>
        <h4>盯盘清单</h4><ul class="lm-list">{_ul(watches, 'neu', '◎')}</ul>
        <p class="muted" style="margin-top:10px">注意：这段是大模型对我们自己量化结论的<b>解读</b>，
        不参与加权（否则等于让大模型给自己打分）。真正进投票的是新闻情绪与政策影响那两路。</p>"""
    else:
        llm_html = f"<pre>{html.escape(llm_txt or '（本次未生成大模型研判）')}</pre>"

    # ---- 大模型独立价格预测 + 条件规则预判（第二意见，与量化模型对照）
    #     注意：_section() 会把标题行本身丢掉（continue 掉了），
    #     所以元信息要从**原文**里搜，不能在截取结果里搜。
    lf_txt = _section(text, "[大模型独立预测]", ("[目标价位区间]", "[买卖点预判]"))
    lf_meta = ""
    lf_fc = []          # [(horizon, pct, price, prob_up, reason)]
    lf_rules = []       # [(trigger, action, why)]
    lf_reasons, lf_risks, lf_summary = [], [], ""
    mh = re.search(r"\[大模型独立预测\]\s*(\S+)\s*（([\d.]+)s）\s*方向\s*(\S+?)"
                   r"（置信\s*([\d.]+)）", text)
    if mh:
        lf_meta = (f"{mh.group(1)}　耗时 {mh.group(2)}s　方向 {mh.group(3)}"
                   f"（置信 {mh.group(4)}）")
    if lf_txt:
        for ln in lf_txt.splitlines():
            s = ln.strip()
            m = re.match(r"预测\s*(\d+)日\s*([+-][\d.]+)%\s*目标\s*([\d.]+|—)\s*"
                         r"收涨概率\s*(\d+)%\s*\|\s*(.*)", s)
            if m:
                lf_fc.append((int(m.group(1)), float(m.group(2)),
                              None if m.group(3) == "—" else float(m.group(3)),
                              int(m.group(4)), m.group(5).strip()))
                continue
            m = re.match(r"规则\s*若\s*(.+?)\s*→\s*(.+?)(?:（(.*)）)?$", s)
            if m:
                lf_rules.append((m.group(1).strip(), m.group(2).strip(),
                                 (m.group(3) or "").strip()))
                continue
            m = re.match(r"理由\s*(.+)$", s)
            if m:
                lf_reasons.append(m.group(1).strip()); continue
            m = re.match(r"风险\s*(.+)$", s)
            if m:
                lf_risks.append(m.group(1).strip()); continue
            m = re.match(r"总结\s*(.+)$", s)
            if m:
                lf_summary = m.group(1).strip(); continue

    lf_html = ""
    if lf_fc or lf_rules:
        # 与量化模型目标价对照：量化侧的目标价从 preds 取
        q_rows = ""
        for h, pct, price, pu, reason in lf_fc:
            cls = "up" if pct > 0 else ("down" if pct < 0 else "neu")
            price_txt = f"{price:.2f}" if price is not None else "—"
            q_rows += (
                f"<tr><td><b>{h} 日</b></td>"
                f"<td class='{cls}'>{pct:+.2f}%</td>"
                f"<td>{price_txt}</td>"
                f"<td>{pu}%</td>"
                f"<td style='text-align:left' class='muted'>{html.escape(reason)}</td></tr>")
        rule_rows = "".join(
            f"<tr><td style='text-align:left'><b>若</b> {html.escape(tr)}</td>"
            f"<td style='text-align:left'>{html.escape(ac)}</td>"
            f"<td style='text-align:left' class='muted'>{html.escape(why)}</td></tr>"
            for tr, ac, why in lf_rules)
        lf_html = f"""
        <div class="lm-head">{html.escape(lf_meta)}</div>
        {f'<p class="lm-concl">{html.escape(lf_summary)}</p>' if lf_summary else ''}
        <h4>大模型自己给的价格预测</h4>
        <table class="tbl"><thead><tr><th>周期</th><th>涨跌</th><th>目标价</th>
        <th>收涨概率</th><th style="text-align:left">理由</th></tr></thead>
        <tbody>{q_rows}</tbody></table>
        <h4 style="margin-top:16px">大模型写的条件规则（if → then）</h4>
        <table class="tbl"><thead><tr><th style="text-align:left">触发条件</th>
        <th style="text-align:left">满足后怎么做</th><th style="text-align:left">为什么</th></tr></thead>
        <tbody>{rule_rows}</tbody></table>
        <div class="grid2" style="margin-top:14px">
          <div><h4 class="h-up">它的理由</h4><ul class="lm-list">{_ul(lf_reasons, 'up', '＋')}</ul></div>
          <div><h4 class="h-down">它提示的风险</h4><ul class="lm-list">{_ul(lf_risks, 'down', '－')}</ul></div>
        </div>
        <p class="muted" style="margin-top:10px">这段是<b>大模型的独立第二意见</b>：
        它拿到的是原始行情 + 技术指标 + 它自己上一轮的新闻研判，
        <b>没有看到</b>量化模型的目标价和综合分，所以两套结论的差异是真分歧，值得对照看。
        它<b>不参与</b>联合决策加权（否则等于让大模型给自己打分）。</p>"""
    elif lf_txt:
        lf_html = f"<pre>{html.escape(lf_txt)}</pre>"

    # ---- 操作清单（缺字段就跳过那一条，不让整份报告崩掉）
    def _p(x, dash="—"):
        return dash if x is None else f"{x:.2f}"

    acts = []
    if buy_lo is not None and buy_hi is not None:
        pos_txt = ""
        if pos_pct is not None:
            pos_txt = ("现在已经在下沿附近（划算）" if pos_pct < 0.25 else
                       "现在在区间中段" if pos_pct < 0.75 else
                       "现在就在上沿（等于追高，别在这里买）")
        acts.append(
            f"<li><b>想买（低吸）</b>：回落到 <b>{buy_lo:.2f} ~ {buy_hi:.2f} 元</b>之间分批买，"
            f"<span class='muted'>{pos_txt}</span>。"
            + (f" 前提是 5 日均线没跌破 20 日均线（{ma20:.2f} 元附近）——"
               f"跌破了说明趋势转弱，先别接。" if ma20 else ""))
    if brk is not None:
        acts.append(
            f"<li><b>怕错过上涨（追强）</b>：只有放量（成交量比平时大 20% 以上）"
            f"站稳 <b>{brk:.2f} 元</b>上方，才算趋势确认，这时可以用剩下的仓位跟进。</li>")
    if tp_lo is not None and tp_hi is not None:
        acts.append(
            f"<li><b>已经持有（止盈）</b>：涨到 <b>{tp_lo:.2f} ~ {tp_hi:.2f} 元</b>区间，分批减仓"
            f"（先减三分之一，再涨再减三分之一），不要一次全卖。</li>")
    if stop is not None:
        acts.append(
            f"<li><b>跌了怎么办（止损）</b>：日收盘价跌破 <b>{stop:.2f} 元</b>就离场。"
            f"这条是硬纪律——不设止损的仓位，一次大跌就能吃掉几个月的利润。</li>")
    actions_html = "<ol>" + "".join(acts) + "</ol>" if acts else \
        "<p class='muted'>本次没有生成买卖点数据（可能是预测或模型环节失败），请查看底部原始报告。</p>"
    stop_note = (
        f"<p class='muted' style='margin-top:12px'>为什么止损定在 {stop:.2f}？"
        f"因为它取的是「30 天的悲观下沿」和「现价跌 8%」两者中更近的那个。"
        f"这样即使模型给出很远的低点，也不会让你在一个必然亏很多的位置才止损。</p>"
        if stop is not None else "")

    img_fc = _img_block(OUT / f"forecast_{code}.png", "股价预测图（点开可放大）",
                        "蓝线=历史真实价格。四个圆点分别是 1/5/7/30 个交易日后的"
                        "目标价——**每档只做一次预测**，不是逐日预测；"
                        "点上的竖线是该档的不确定性范围；细点线只是示意方向。"
                        "竖虚线+浅色带=预计拐点出现的时间与区间。")
    img_cd = _img_block(OUT / f"candles_{code}.png", "K 线形态图（近 60 个交易日）",
                        "红涨绿跌。技术分析里常看的形态就长这样。")

    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>{html.escape(disp)}({code}) 报告 · 小白版</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f5f6f8;color:#1f2328;
 font:15px/1.75 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1000px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:23px;margin:0 0 4px}}
h2{{font-size:17px;margin:0 0 4px}}
h3{{font-size:16px;margin:0 0 12px}}
.card{{background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:20px 22px;margin:16px 0}}
.muted{{color:#6b7280;font-size:13px;line-height:1.6}}
.warn{{color:#b45309;font-size:12.5px}}
.meta{{color:#5b6470;font-size:13.5px}}
.tag{{display:inline-block;font-size:12px;color:#5b6470;background:#eef2f7;
 border-radius:999px;padding:2px 10px;margin-left:8px;vertical-align:2px}}
.verdict{{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin:14px 0 8px}}
.big{{font-size:34px;font-weight:800;letter-spacing:.5px}}
.big.up{{color:#c0392b}} .big.down{{color:#1e7f43}} .big.neutral{{color:#4b5563}}
.now{{background:#f0f7ff;border-left:4px solid #3b82f6;padding:12px 16px;border-radius:6px;margin-top:12px}}
ol,ul{{margin:8px 0 0;padding-left:22px}} li{{margin:7px 0}}
.hz{{border:1px solid #e8ebef;border-radius:10px;padding:14px 16px;margin:10px 0;background:#fafbfc}}
.hz-h{{font-weight:700;font-size:15px;margin-bottom:4px}}
.hz-b{{font-size:15px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:8px}}
th,td{{padding:8px 10px;border-bottom:1px solid #edf0f3;text-align:right}}
th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}}
thead th{{background:#f6f8fa;font-weight:600;color:#5b6470}}
td.hi{{font-weight:700;background:#fff8e1}}
.up{{color:#c0392b;font-weight:600}} .down{{color:#1e7f43;font-weight:600}} .neu{{color:#6b7280}}
img{{max-width:100%;border-radius:8px;display:block;margin-top:4px}}
pre{{background:#f6f8fa;border:1px solid #edf0f3;border-radius:8px;padding:12px 14px;
 overflow-x:auto;font:12.5px/1.55 ui-monospace,Consolas,monospace}}
/* 信号贡献表：双向条 + 贡献条 */
table.sig td{{vertical-align:middle}}
.b2{{position:relative;display:inline-block;width:96px;height:9px;background:#eef1f4;
 border-radius:5px;vertical-align:middle;margin-right:8px}}
.b2:after{{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:#c3cbd4}}
.b2 i{{position:absolute;top:0;bottom:0;border-radius:5px}}
.b2 .v{{font-variant-numeric:tabular-nums}}
td .v{{font-variant-numeric:tabular-nums;font-size:12.5px;color:#4b5563}}
.cbar{{height:4px;background:#eef1f4;border-radius:3px;margin-top:5px;overflow:hidden}}
.cbar i{{display:block;height:100%;border-radius:3px}}
/* 大模型研判结构化 */
.lm-head{{display:inline-block;font-size:12.5px;color:#5b6470;background:#eef2f7;
 border-radius:999px;padding:3px 12px;margin-bottom:10px}}
.lm-concl{{font-size:15.5px;line-height:1.8;margin:0 0 12px;padding:12px 16px;
 background:#f8fafc;border-left:4px solid #3b82f6;border-radius:6px}}
.lm-list{{list-style:none;padding-left:0;margin:6px 0}}
.lm-list li{{padding:6px 0;border-bottom:1px dashed #eef1f4;font-size:14px}}
.lm-list li:last-child{{border-bottom:0}}
.lm-list .up,.lm-list .down,.lm-list .neu{{display:inline-block;width:18px;font-weight:700}}
.h-up{{color:#c0392b}} .h-down{{color:#1e7f43}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.kv3{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:10px 0}}
.kv3>div{{background:#f8fafc;border:1px solid #eef1f4;border-radius:10px;padding:9px 12px}}
.kv3 span{{display:block;font-size:12px;color:#6b7280}}
.kv3 b{{font-size:18px}}
@media(max-width:700px){{.kv3{{grid-template-columns:1fr 1fr}}}}
@media(max-width:700px){{.grid2{{grid-template-columns:1fr}}}}
summary{{cursor:pointer;color:#374151;font-weight:600;list-style:none}}
summary::-webkit-details-marker{{display:none}}
summary:before{{content:"▸ ";color:#9ca3af}}
details[open] summary:before{{content:"▾ "}}
details.card{{padding:16px 22px}}
.foot{{color:#6b7280;font-size:12.5px;margin-top:26px;line-height:1.7}}
.nav{{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 14px;font-size:13.5px}}
.nav a{{color:#2563eb;text-decoration:none;background:#eef4ff;border-radius:999px;padding:5px 14px}}
.nav a:hover{{background:#dbeafe}}
@media(max-width:700px){{.big{{font-size:28px}}}}
</style></head><body><div class="wrap">

<div class="nav"><a href="/">← 返回主页</a><a href="/analysis.html">完整分析页</a>
<a href="/report?code={code}&refresh=1">重新分析这只</a></div>

<h1>{html.escape(disp)}（{code}）</h1>
<div class="meta">数据截至 {html.escape(asof)} ｜ 最新收盘 <b>{close:.2f}</b>
{f'｜ 盘中价 {price:.2f}' if abs(gap_pct) >= 0.5 else ''}
<span class="tag">模型：{html.escape(model)}（多模型融合）</span>
<span class="tag">历史 {html.escape(str(n_hist))} 个交易日</span></div>
{stale_banner}

<!-- 1. 一句话结论 -->
<div class="card">
  <h2>一句话结论</h2>
  <div class="verdict">
    <div class="big {v_cls}">{html.escape(v_title)}</div>
    <div class="muted" style="flex:1;min-width:240px">{html.escape(v_why)}。</div>
  </div>
  <div class="now">{now_act}</div>
  {f'<p class="muted" style="margin-top:8px">{price_note}</p>' if price_note else ''}
  <p class="muted" style="margin-top:10px">系统原始结论是「{html.escape(verdict)}」，综合分
  {html.escape(score)}（这个分数范围是 -1 到 +1：越接近 +1 越看涨，越接近 -1 越看跌，
  0 附近表示多空打平）。起主要作用的两个因素是：{lead_txt}。</p>
</div>

<!-- 1.5 核心数据一览 -->
<div class="card">
  <h2>核心数据一览<span class="tag">1 / 5 / 7 / 30 个交易日</span></h2>
  <p class="muted">每个周期只做<b>一次</b>直接回归，预测「第 N 个交易日的收盘价」，不是逐日滚动。
  「方向」一栏在信号不够强时会写<b>方向不明</b>——这是刻意的，不是漏了（实测 7 日方向准确率只有 50.4%）。</p>
  <table><thead><tr><th>周期</th><th>预测日</th><th>预测收盘</th><th>相对现价</th>
  <th>上涨概率</th><th>方向</th><th>价格区间</th></tr></thead>
  <tbody>{core_rows}</tbody></table>
</div>

<!-- 1.6 大模型政策与新闻研判 -->
<div class="card">
  <h2>大模型对政策与新闻的研判<span class="tag">这两路真的进了投票</span></h2>
  {news_card}
</div>

<!-- 2. 现在该怎么做 -->
<div class="card">
  <h2>现在该怎么做<span class="tag">按价位执行</span></h2>
  <p class="muted" style="margin:-4px 0 6px">价位都以最新收盘价 {close:.2f} 为基准；下单前对照一下实时价。</p>
  {actions_html}
  {stop_note}
</div>

<!-- 3. 未来能到什么价 -->
<div class="card">
  <h2>未来 5 / 7 / 30 个交易日，价格可能在哪</h2>
  <p class="muted">先说明一件事：<b>"价格范围"不代表"预测精度"</b>。
  它说的是"按最近的真实波动和模型之间的分歧，价格大概会在这个区间里晃"。
  区间越宽，说明越不确定——这时候就别重仓。</p>
  {trows_plain}
  <h3 style="margin-top:16px">精确数字表</h3>
  <table><thead><tr><th>周期</th><th>到达日</th><th>下界</th><th>最可能</th><th>上界</th>
  <th>相对现价</th><th>区间宽度</th><th>上涨概率</th><th>方向</th></tr></thead>
  <tbody>{trows_pro}</tbody></table>
</div>

<!-- 4. 为什么这么判断 -->
<div class="card">
  <h2>为什么得出这个结论</h2>
  <p class="muted">系统同时看 6 个方面，每个方面给一个"偏多 / 偏空"的判断，再按重要性加权投票。
  下面是按影响力从大到小排的。</p>
  <table><thead><tr><th>看什么</th><th>结论</th><th>权重</th><th>这是什么意思</th></tr></thead>
  <tbody>{srows}{model_plain}</tbody></table>
  <p class="muted" style="margin-top:10px">权重合计就是 100%，所以"权重 40%"表示这一项占四成话语权。
  新闻和政策这两项已经真正算进投票了（各占 10%），不是只写在报告里好看的。</p>
</div>

<!-- 5. 可信度 -->
<div class="card">
  <h2>这套系统有多可信<span class="tag">请一定看完</span></h2>
  <ul>
    <li><b>短周期（1~5 天）方向判断，准确率约 52~53%</b>——比抛硬币（50%）好一点，但只是一点点。
      所以<b>不要把它当成"明天必涨/必跌"</b>。</li>
    <li><b>30 天方向判断，实测准确率 48.3%，还不如抛硬币</b>，所以系统干脆不给 30 天的涨跌方向，
      只给价格范围。这是刻意设计的，不是漏了。</li>
    <li><b>拐点预测（什么时候见顶/见底）有一定信号</b>，样本外 AUC 0.61~0.69（1.0 是完美，
      0.5 是瞎猜），只能当"提醒"，不能当"闹钟"。</li>
    <li><b>新闻与政策的影响无法回测</b>——因为历史上没有存下当时的新闻文本。
      所以这一路只能算"有信息价值"，不能算"已验证有效"。</li>
  </ul>
  <p class="muted" style="margin-top:10px">一句话：<b>这套系统擅长的是"给出合理的价格区间和风险边界"，
  不擅长"精确猜涨跌"</b>。把它当纪律工具（该在哪买、哪卖、哪止损），比当预言机靠谱得多。</p>
</div>

{img_fc}
{img_cd}

<!-- 6. 专业明细 -->
<div class="card">
  <h2>专业明细 · 各信号贡献<span class="tag">给内行</span></h2>
  <p class="muted">取值归一到 −1（偏空）~ +1（偏多）；贡献 = 取值 × 权重 ÷ 权重和。
  条形图向左为绿（偏空）、向右为红（偏多），长度按贡献绝对值缩放。</p>
  <table class="sig"><thead><tr><th>信号 / 含义</th><th>取值</th><th>权重</th>
  <th>贡献</th><th>依据</th></tr></thead><tbody>{sig_rows}</tbody></table>
</div>

<div class="card">
  <h2>专业明细 · 大模型综合研判</h2>
  {llm_html}
</div>

<div class="card">
  <h2>大模型自己的价格预测与条件规则<span class="tag">独立第二意见</span></h2>
  {lf_html}
</div>

<details class="card">
  <summary>名词解释</summary>
  <ul>
    <li><b>目标价位区间</b>：不是说"一定会到这个价"，而是"合理的波动范围"。下界=偏悲观，上界=偏乐观。</li>
    <li><b>最可能（中位）</b>：区间正中间那个价，模型认为概率最高的位置。</li>
    <li><b>±X%（区间宽度）</b>：范围有多宽。越窄说明模型越有把握（但仍可能错）。</li>
    <li><b>上涨概率</b>：模型认为会涨的可能性。48% 就是"略低于一半"。</li>
    <li><b>ATR</b>：最近平均每天波动多少。1.77% 表示一天涨跌 1.8% 属于正常。</li>
    <li><b>MA20（20 日均线）</b>：过去 20 天收盘价的平均值。价格在它上面通常算强势，跌破常被当转弱信号。</li>
    <li><b>拐点</b>：从涨转跌（顶）或从跌转涨（底）的那个转折点。</li>
    <li><b>多模型融合（ensemble）</b>：让随机森林、梯度提升、神经网络三种模型各自预测再取平均，
      减少单个模型的偶然性。</li>
    <li><b>观望</b>：多空力量接近，这时候买和卖都不划算，等信号更清楚再动。</li>
  </ul>
</details>

<details class="card">
  <summary>完整原始报告（最细，含所有技术细节）</summary>
  <pre>{html.escape(text)}</pre>
</details>

<div class="foot">
生成时间 {dt.datetime.now():%Y-%m-%d %H:%M} ｜ 数据来源：新浪财经（行情）、东方财富（新闻）、千问（文本研判）<br>
<b>本报告是算法量化输出，不构成投资建议。</b>股市有风险，任何模型都可能出错，
请结合自己的资金情况和风险承受能力做决定。
</div>
</div></body></html>"""

    outp = OUT / f"report_{code}.html"
    outp.write_text(doc, encoding="utf-8")
    return outp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="600519")
    ap.add_argument("--log", default="", help="已有文本报告路径；不填则用 output/decision_report_<code>.txt")
    ap.add_argument("--run", action="store_true", help="先生成再跑一次分析（会重新训练，耗时数分钟）")
    ap.add_argument("--mode", default="predict", choices=["predict", "backtest", "all"])
    ap.add_argument("--extra", default="", help="透传给 run.py 的额外参数，如 '--pool-limit 50 --no-llm'")
    a = ap.parse_args()
    code = re.sub(r"\D", "", a.symbol)[-6:]

    if a.run:
        import subprocess
        import sys
        # -u 关闭缓冲，否则 run.py 的进度会攒到结束才一次性吐出来，等待页看不到实时进度
        cmd = [sys.executable, "-u", "run.py", "--symbol", a.symbol, "--mode", a.mode]
        if a.extra:
            cmd += a.extra.split()
        print("跑分析：" + " ".join(cmd))
        # 流式读取：一边跑一边把进度打到 stdout（等待页就能看到实时进度），
        # 一边收集成最终报告文本。用 capture_output 会把 3~6 分钟的输出全憋在最后。
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                bufsize=1)
        keep = []
        logp = OUT / f"decision_report_{code}.txt"
        # 边跑边落盘到**临时文件**，跑完再替换正式文件：
        #   - 中途能看到进度（以前只在跑完后一次性写，中途文件是空的，排查时看不出跑到哪）
        #   - 但如果直接写正式文件，中途被 kill 会把**上一份好报告**覆盖成半截残缺内容
        #     （实测 601456 就这样被截断成 1 行，快照解析直接失败）
        tmp_logp = logp.with_suffix(".txt.tmp")
        with open(tmp_logp, "w", encoding="utf-8") as lf:
            for ln in proc.stdout:
                ln = ln.rstrip("\n")
                if re.search(r"Warning|NUMEXPR|NUMEXPR_INSTALLED|^  from pandas|UserWarning", ln):
                    continue
                keep.append(ln)
                lf.write(ln + "\n")
                lf.flush()
                print(ln, flush=True)          # 实时进度
        proc.wait()
        logp.write_text("\n".join(keep).strip() + "\n", encoding="utf-8")
        try:
            tmp_logp.unlink()
        except OSError:
            pass
        print(f"文本报告：{logp}（exit={proc.returncode}）")
        if proc.returncode != 0:
            print("分析执行失败，末尾输出：")
            print("\n".join(keep[-15:]))
            return 1
        a.log = str(logp)

    log = Path(a.log) if a.log else OUT / f"decision_report_{code}.txt"
    if not log.exists():
        print(f"找不到文本报告：{log}\n先加 --run 跑一次，或用 --log 指定已有报告。")
        return 2
    p = build(a.symbol, log)
    print(f"已生成小白版报告：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
