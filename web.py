"""股票选择 / 预测 Web 页面（零额外依赖，仅用标准库 http.server）。

启动：
    python web.py            # http://127.0.0.1:8010

功能：
  - 输入股票代码查看当日行情 + 1/5/7/30 日预测
  - K 线形态识别、量能与资金流、新闻政策（大模型解读）
  - 拐点预测（顶/底概率、预计日期区间）
  - 多信号联合决策（给出每个信号的权重与贡献，可反查）
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import mimetypes
import os
import re
import secrets as _secrets
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import matplotlib

matplotlib.use("Agg")
import pandas as pd

import config
from stockpred import analyze, plot_forecast, plot_candles, detect_patterns
from stockpred.llm import status as llm_status, chat as llm_chat, _override_target

HOST = "127.0.0.1"
PORT = 8010

BASE = os.path.dirname(os.path.abspath(__file__))


def _pick_site_dir(base: str) -> str:
    """静态站目录：优先 deploy_site（历史命名），回退 docs（GitHub Pages 目录）。

    两个名字都认，是为了让本地 Web 与 Pages 共用同一份文件，
    避免目录改名后本地「完整分析页 / 预生成报告」静默 404。
    """
    for name in ("deploy_site", "docs"):
        cand = os.path.join(base, name)
        if os.path.isdir(cand):
            return cand
    return os.path.join(base, "deploy_site")


SITE = _pick_site_dir(BASE)    # 静态站（首页 / 完整分析页 / 预生成报告）
# 报告复用上限（墙钟小时）。真正的"过期"判定用 _report_is_stale：
# 比 **报告里的数据日** 和 **数据源最新可用交易日**，而不是文件年龄 ——
# 周末/节假日里文件可能才 1 天却已经落后 2 个交易日（周一早上最典型）。
REPORT_TTL = 72 * 3600
JOBS: dict[str, dict] = {}                      # code -> {"proc","start","log"}
_JOB_LOCK = threading.Lock()

# 失败熔断：同一只票连续失败这么多次就冷却（避免永远跑不出来的代码无限起进程）
MAX_CONSECUTIVE_FAILS = 3
FAIL_COOLDOWN = 30 * 60                         # 冷却 30 分钟
_FAIL_COUNT: dict[str, dict] = {}               # code -> {"n": 次数, "at": 最后失败时间}
# 全局并发上限：一次分析吃 1.5GB 内存 + 3~6 分钟 CPU，同时最多跑这么多个
MAX_CONCURRENT_JOBS = 2

# 数据源最新可用交易日（缓存 10 分钟，避免每个请求都打新浪）
_LAST_TRADING_DAY: dict = {"day": "", "ts": 0.0}


def _latest_trading_day() -> str:
    """从行情源探测"目前最新一根日线是哪天"（YYYY-MM-DD）。

    为什么需要它：判断报告是否过期，不能看文件年龄，要看**数据日**。
    周一早上 9:50 时，周五生成的报告才 3 天大，但已经落后一个交易日 ——
    用户看到的"最新收盘"其实是上周五的，而页面上写着"数据截至 10-21"，很容易误以为是今天。
    """
    import time as _t
    with _JOB_LOCK:
        if _LAST_TRADING_DAY["day"] and (_t.time() - _LAST_TRADING_DAY["ts"]) < 600:
            return _LAST_TRADING_DAY["day"]
    day = ""
    try:
        from stockpred.data_fetcher import fetch_kline
        df = fetch_kline("sh000001", datalen=5)      # 上证指数：交易日历的可靠代理
        if df is not None and not df.empty:
            day = str(pd.to_datetime(df["date"].iloc[-1]).date())
    except Exception:
        day = ""
    with _JOB_LOCK:
        _LAST_TRADING_DAY["day"] = day
        _LAST_TRADING_DAY["ts"] = _t.time()
    return day


def _report_data_day(path: str) -> str:
    """从已生成的报告 HTML 里抠出它的"数据截至"日期（YYYY-MM-DD）。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = f.read(20000)
        m = re.search(r"数据截至\s*(\d{4}-\d{2}-\d{2})", head)
        if m:
            return m.group(1)
        m = re.search(r"最新收盘\s*[\d.]*（(\d{4}-\d{2}-\d{2})）", head)
        return m.group(1) if m else ""
    except OSError:
        return ""


def _report_is_stale(path: str) -> bool:
    """报告是否该重跑：数据日落后于数据源最新交易日，或文件年龄超 TTL。"""
    if not os.path.exists(path):
        return True
    if (time.time() - os.path.getmtime(path)) >= REPORT_TTL:
        return True
    latest = _latest_trading_day()
    day = _report_data_day(path)
    if latest and day and day < latest:
        return True
    return False



def _pid_alive(pid: int) -> bool:
    """判断进程是否还活着。Windows 上不能用 os.kill(pid, 0)（会真的终止进程）。"""
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    except Exception:
        return False


def _report_job(code: str, force: bool = False, api_key: str = "",
                model: str = "", base_url: str = "") -> dict | None:
    """确保这只股票的 report_<code>.html 在生成中/已生成。返回正在跑的 job 或 None。

    为什么用子进程而不是在当前线程里算：一次完整分析要 3~5 分钟，
    浏览器/Cloudflare 都等不了那么久（会 524），所以「提交任务 → 前端自动轮询」。

    model / base_url：访客在首页选的模型必须传进子进程，否则网页上选「免费额度模型」
    对 /report 路径完全无效（子进程只认 config/secrets 里的默认模型）。
    """
    out = config.OUTPUT_DIR
    rp = os.path.join(out, f"report_{code}.html")
    with _JOB_LOCK:
        j = JOBS.get(code)
        if j and j["proc"].poll() is None:
            return j                      # 已有任务在跑
        if j and j["proc"].poll() is not None:
            JOBS.pop(code, None)
        if not force and not _report_is_stale(rp):
            return None                   # 现成的够新，直接用
        # ★ 全局并发上限：一次分析要 1.5GB 内存 + 3~6 分钟 CPU。
        #   多个访客同时提交不同代码 → 内存吃光、彼此抢 CPU，全部变慢甚至卡死。
        #   超过上限就不再起新任务，让用户排队（等待页会继续轮询）。
        running = sum(1 for v in JOBS.values()
                      if v.get("proc") is not None and v["proc"].poll() is None)
        if running >= MAX_CONCURRENT_JOBS:
            return {"proc": None, "start": time.time(), "log": out, "file": rp,
                    "queued": running}
        # ★ 失败熔断：像 159562 这种**永远跑不出报告**的代码（ETF/退市/非 A 股），
        #   以前每次请求都会再起一个子进程 → 实测 20 分钟内堆了 100+ 个 1.5GB 的
        #   孤儿进程把机器打满。现在同一只票连续失败 N 次就冷却，不再起进程。
        fail = _FAIL_COUNT.get(code, {"n": 0, "at": 0.0})
        if (fail["n"] >= MAX_CONSECUTIVE_FAILS
                and (time.time() - fail["at"]) < FAIL_COOLDOWN):
            return {"proc": None, "start": fail["at"], "log": out, "file": rp,
                    "failed": fail["n"]}
        logp = os.path.join(out, f"_job_{code}.log")
        # 服务重启后内存里的 JOBS 会丢：靠 pid 文件判断"已经有人在跑这只"，
        # 否则会起第二份训练抢 CPU（两份同时跑比一份慢一倍）。
        pidf = os.path.join(out, f"_job_{code}.pid")
        if os.path.exists(pidf):
            try:
                old_pid = int(open(pidf, encoding="utf-8").read().strip() or 0)
            except (OSError, ValueError):
                old_pid = 0
            if old_pid and _pid_alive(old_pid):
                return {"proc": None, "start": os.path.getmtime(pidf), "log": logp, "file": rp}
        env = os.environ.copy()
        env.pop("PYTHONUSERBASE", None)   # psychopy 之类改过这个变量会坑子进程
        env["PYTHONIOENCODING"] = "utf-8"
        if api_key:
            # 访客自带的 Key 只通过环境变量传给子进程，不落盘、不进日志、不进 URL
            env["DASHSCOPE_API_KEY"] = api_key
        # 访客选的模型/端点要传进子进程，否则网页上选「免费额度模型」对 /report 路径无效。
        # 走 LLM_OVERRIDE_* 而不是改 config：override 明确带 base_url，
        # 不会把"所有供应商的模型"都改成同一个名字（那会 401）。
        if model:
            env["LLM_OVERRIDE_MODEL"] = model
        if base_url:
            env["LLM_OVERRIDE_BASE"] = base_url
        else:
            env.pop("LLM_OVERRIDE_BASE", None)
        if api_key:
            env["LLM_OVERRIDE_KEY"] = api_key
        else:
            env.pop("LLM_OVERRIDE_KEY", None)
        # 首次运行（clone 下来 output/ 不存在）时日志目录要先建出来，
        # 否则 open(logp, "w") 会抛 FileNotFoundError 直接把请求掐断。
        try:
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        except OSError:
            pass
        try:
            fh = open(logp, "w", encoding="utf-8")
        except PermissionError:
            # 日志文件被杀毒/网盘/索引器瞬时锁住是真实发生过的（用户端报过 502），
            # 等 1 秒重试，还不行就换一个带时间戳的备用名，绝不在这里抛异常。
            time.sleep(1)
            try:
                fh = open(logp, "w", encoding="utf-8")
            except OSError:
                logp = os.path.join(out, f"_job_{code}_{int(time.time())}.log")
                fh = open(logp, "w", encoding="utf-8")
        proc = subprocess.Popen([sys.executable, "-u", "make_report.py", "--symbol", code, "--run"],
                                cwd=BASE, stdout=fh, stderr=subprocess.STDOUT, env=env)
        try:
            with open(os.path.join(out, f"_job_{code}.pid"), "w", encoding="utf-8") as pf:
                pf.write(str(proc.pid))
        except OSError:
            pass
        j = {"proc": proc, "start": time.time(), "log": logp, "file": rp, "fh": fh}
        JOBS[code] = j
        return j


def _note_job_result(code: str, ok: bool):
    """任务结束时记账：成功清零，连续失败计数 +1（供 _report_job 熔断用）。"""
    with _JOB_LOCK:
        if ok:
            _FAIL_COUNT.pop(code, None)
        else:
            f = _FAIL_COUNT.setdefault(code, {"n": 0, "at": 0.0})
            f["n"] += 1
            f["at"] = time.time()



def _sanitize_log(raw: str) -> str:
    """把本机绝对路径脱敏，避免把用户目录/工具链路径暴露到公网页面上。"""
    raw = re.sub(r"[A-Za-z]:[\\/]Users[\\/][^\s\"']*", "<本机路径>", raw)
    raw = re.sub(r"/home/[^\s\"']*", "<本机路径>", raw)
    raw = re.sub(r"/Users/[^\s\"']*", "<本机路径>", raw)
    return raw

def _job_wait_page(code: str, j: dict) -> str:
    waited = int(time.time() - j["start"])
    tail = ""
    try:
        raw = open(j["log"], encoding="utf-8", errors="replace").read()
        raw = _sanitize_log(raw)
        lines = [ln for ln in raw.splitlines() if ln.strip()][-6:]
        tail = html.escape("\n".join(lines))
    except OSError:
        pass
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>正在分析 {code} …</title>
<style>body{{font:15px/1.7 -apple-system,'Microsoft YaHei';background:#f5f6f8;margin:0}}
.wrap{{max-width:760px;margin:60px auto;background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:26px 30px}}
h1{{font-size:20px;margin:0 0 8px}}.muted{{color:#6b7280;font-size:13.5px}}
pre{{background:#f6f8fa;border:1px solid #edf0f3;border-radius:8px;padding:12px;font-size:12px;
overflow-x:auto;white-space:pre-wrap}}a{{color:#2563eb}}</style></head><body><div class="wrap">
<h1>正在分析 {code} …</h1>
<p class="muted">已等待 <b>{waited}</b> 秒。完整分析要训 300 只股票池的多模型，通常 <b>3~5 分钟</b>，
这一页每 6 秒自动刷新，不用手点。</p>
<p class="muted">生成完会自动显示报告。离开这一页也没关系，任务在后台继续跑。</p>
{('<p class="muted">最新进度：</p><pre>' + tail + '</pre>') if tail else ''}
<p id="doneBox" style="display:none;font-size:15px;margin-top:16px">
  <a href="/report/view?code={code}" style="display:inline-block;background:#2ca02c;color:#fff;
  text-decoration:none;border-radius:8px;padding:12px 22px;font-weight:600">报告已生成？点这里直接看 →</a></p>
<p id="jumpTip" class="muted" style="display:none">报告已生成，正在自动跳转…</p>
<script>
(function () {{
  var code = "{code}";
  function tick() {{
    fetch("/report/status?code=" + code, {{cache: "no-store"}})
      .then(function (r) {{ return r.json(); }})
      .then(function (j) {{
        if (j && j.ready) {{
          document.getElementById("jumpTip").style.display = "block";
          location.replace(j.url);          // 直接跳到报告，不再停留在等待页
          return;
        }}
        setTimeout(tick, 4000);
      }})
      .catch(function () {{ setTimeout(tick, 6000); }});
  }}
  setTimeout(tick, 3000);
}})();
</script>
<p class="muted"><a href="/">← 返回主页</a>　·　<a href="/analysis.html">看完整分析页</a></p>
</div></body></html>"""


def _report_form(msg: str = "") -> str:
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>输入股票代码</title>
<style>body{{font:15px/1.7 -apple-system,'Microsoft YaHei';background:#f5f6f8;margin:0}}
.wrap{{max-width:620px;margin:80px auto;background:#fff;border:1px solid #dfe3e8;border-radius:12px;padding:26px 30px}}
input{{width:100%;padding:10px 12px;border:1px solid #dfe3e8;border-radius:8px;font-size:15px}}
button{{margin-top:12px;background:#2ca02c;color:#fff;border:0;border-radius:8px;padding:11px 22px;font-size:15px;cursor:pointer}}
.warn{{background:#fffbeb;border-left:3px solid #f59e0b;padding:10px 14px;border-radius:6px;font-size:13.5px}}</style>
</head><body><div class="wrap"><h1 style="font-size:20px;margin:0 0 10px">输入股票代码</h1>
{('<div class="warn">' + html.escape(msg) + '</div>') if msg else ''}
<form method="get" action="/report"><input name="code" placeholder="600519 / 000001 / 601318" autofocus>
<button type="submit">开始分析（3~5 分钟）</button></form>
<p style="font-size:13px;color:#6b7280"><a href="/">← 返回主页</a></p></div></body></html>"""

# 访客自带的密钥只存在**服务端内存**里，按会话（cookie）隔离：
# - 不写日志、不落盘、不进 URL（避免被浏览器历史/代理记录）
# - 服务重启即清空
SESSIONS: dict[str, dict] = {}
_SESS_LOCK = threading.Lock()

COMMON_ENDPOINTS = [
    ("通义千问（阿里云百炼）", "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
    ("DeepSeek", "https://api.deepseek.com/v1", "deepseek-chat"),
    ("智谱 GLM", "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash"),
    ("本地 Ollama", "http://localhost:11434/v1", "qwen2.5"),
]

# 模型下拉项：(显示名, base_url, model, 是否免费额度)
# 免费额度说明：阿里云百炼对新用户每个模型各送 100 万 tokens（约千万字符），
# 选这些模型不烧钱；qwen-turbo 偶尔会被限流(429)，所以排在后面。
MODEL_CHOICES = [
    ("千问 qwen-plus（百炼免费额度 100 万 tokens）",
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus", True),
    ("千问 qwen-flash（百炼免费额度，最快最省）",
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-flash", True),
    ("千问 qwen3.5-plus（百炼免费额度）",
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.5-plus", True),
    ("千问 qwen3.7-plus（百炼免费额度，研判更细）",
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen3.7-plus", True),
    ("千问 qwen-turbo（百炼免费额度，偶尔限流）",
     "https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-turbo", True),
    ("DeepSeek deepseek-chat（需充值，按量很便宜）",
     "https://api.deepseek.com/v1", "deepseek-chat", False),
    ("智谱 GLM-4-flash（有免费额度，需注册）",
     "https://open.bigmodel.cn/api/paas/v4", "glm-4-flash", True),
    ("本地 Ollama（完全免费，需本机已装模型）",
     "http://localhost:11434/v1", "qwen2.5", True),
]

FAVORITES = [
    ("600519", "贵州茅台"), ("000001", "平安银行"), ("600036", "招商银行"),
    ("300750", "宁德时代"), ("601318", "中国平安"), ("000858", "五粮液"),
    ("688836", "宇树科技"),
]


PAGE_HEAD = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>股价走势预测 · 形态/量能/新闻/拐点</title>
<style>
 *{box-sizing:border-box}
 body{font-family:"Microsoft YaHei",SimHei,system-ui,sans-serif;margin:0;background:#f5f7fa;color:#222}
 header{background:#1f3a5f;color:#fff;padding:16px 24px;font-size:20px;font-weight:700}
 .wrap{max-width:1020px;margin:auto;padding:20px}
 .card{background:#fff;border-radius:10px;padding:18px 20px;margin-bottom:18px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
 form{display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end}
 input[type=text],input[type=password]{padding:9px 12px;border:1px solid #ccd;border-radius:6px;font-size:15px;width:180px}
 button{padding:9px 18px;border:0;border-radius:6px;background:#2ca02c;color:#fff;font-size:15px;cursor:pointer}
 button:hover{background:#258b25}
 .fav a{display:inline-block;margin:6px 6px 0 0;padding:6px 12px;background:#eef3f8;border-radius:16px;color:#1f3a5f;text-decoration:none;font-size:13px}
 .fav a:hover{background:#dbe7f3}
 .price{font-size:34px;font-weight:800;color:#d62728}
 .price.down{color:#2ca02c}
 .sub{color:#777;font-size:13px;margin-top:4px}
 table{border-collapse:collapse;width:100%;font-size:14px;margin-top:8px}
 th,td{border:1px solid #eee;padding:6px 8px;text-align:center}
 th{background:#f0f4f8}
 .up{color:#d62728}.down{color:#2ca02c}
 .tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:12px;background:#eef7ee;color:#2ca02c}
 .tag.warn{background:#fdf0e8;color:#c25b12}
 .tag.bad{background:#fdecec;color:#d62728}
 .tag.neutral{background:#eef1f5;color:#555}
 img{max-width:100%;border:1px solid #eee;border-radius:8px;margin-top:8px}
 .muted{color:#999;font-size:12px}
 .err{color:#d62728;padding:10px}
 .verdict{display:flex;align-items:center;gap:18px;flex-wrap:wrap}
 .verdict .big{font-size:30px;font-weight:800}
 .bar{position:relative;height:12px;background:linear-gradient(90deg,#2ca02c 0%,#e9eef4 50%,#d62728 100%);border-radius:8px;width:320px}
 .bar i{position:absolute;top:-4px;width:4px;height:20px;background:#1f3a5f;border-radius:2px}
 .kv{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:#555}
 h3{margin:0 0 8px 0;font-size:16px}
 ul{margin:6px 0 0 18px;padding:0}li{font-size:13px;line-height:1.7}
</style></head><body>"""


# ------------------------- 渲染组件 -------------------------
def _quote_card(res: dict) -> str:
    q = res["quote"]
    name = res["name"]
    if q and q.get("price"):
        price = float(q["price"])
        prev = float(q.get("prev_close") or res["last_close"])
        chg = price - prev
        pct = chg / prev * 100 if prev else 0
        cls = "up" if chg >= 0 else "down"
        arrow = "▲" if chg >= 0 else "▼"
        wb = q.get("weibi", 0)
        wb_cls = "up" if wb > 0 else ("down" if wb < 0 else "")
        return f"""<div class="card">
          <div class="sub">{html.escape(name)}（{html.escape(res['symbol'])}） 当日实时行情</div>
          <div class="price {cls}">{price:.2f} <span style="font-size:18px">{arrow} {chg:+.2f} ({pct:+.2f}%)</span></div>
          <div class="sub">今开 {q.get('open','-')} ｜ 昨收 {q.get('prev_close','-')} ｜ 最高 {q.get('high','-')} ｜ 最低 {q.get('low','-')}</div>
          <div class="kv" style="margin-top:8px">
            <span>五档委比 <b class="{wb_cls}">{wb:+.2f}%</b>（{q.get('bid_vol',0):.0f} 手买 / {q.get('ask_vol',0):.0f} 手卖）</span>
            <span>成交额 {(q.get('amount',0) or 0)/1e8:.2f} 亿</span>
            <span>数据 {res['last_date']}</span>
          </div>
        </div>"""
    return f"""<div class="card">
      <div class="sub">{html.escape(name)}（{html.escape(res['symbol'])}）最新收盘价（实时行情暂不可用）</div>
      <div class="price">{res['last_close']:.2f}</div>
      <div class="sub">数据日期 {res['last_date']}</div></div>"""


def _decision_card(res: dict) -> str:
    d = res["decision"]
    s = d["score"]
    pos = (s + 1) / 2 * 100
    cls = "up" if s > 0.05 else ("down" if s < -0.05 else "")
    rows = "".join(
        f"<tr><td>{html.escape(x['name'])}</td><td>{x['value']:+.3f}</td>"
        f"<td>{x['weight']:.2f}</td><td class='{'up' if x['contrib']>0 else ('down' if x['contrib']<0 else '')}'>{x['contrib']:+.3f}</td>"
        f"<td style='text-align:left'>{html.escape(x['note'])}</td></tr>"
        for x in d["signals"])
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in d.get("notes", []))
    return f"""<div class="card">
      <h3>联合决策（多信号加权，可反查）</h3>
      <div class="verdict">
        <div class="big {cls}">{html.escape(d['advice'])}</div>
        <div>
          <div class="bar"><i style="left:{pos:.1f}%"></i></div>
          <div class="sub">综合分 {s:+.3f}（-1 空 ↔ +1 多）｜ 置信度 {d['confidence']:.2f} ｜ 信号一致度 {d['agreement']:.0%} ｜ 适用 {html.escape(d['horizon'])}</div>
        </div>
      </div>
      <div class="sub" style="margin-top:6px">{html.escape(d['action'])}</div>
      <table><thead><tr><th>信号</th><th>取值(-1~1)</th><th>权重</th><th>贡献</th><th>依据</th></tr></thead>
      <tbody>{rows}</tbody></table>
      {f'<ul class="muted">{notes}</ul>' if notes else ''}
      <div class="muted">取值 = 各信号归一到 [-1,1] 后的分值；贡献 = 取值×权重/权重和（缺失信号已剔除并重新归一）。
      权重见 config.SIGNAL_WEIGHTS，可自行调整。</div>
    </div>"""


def _trade_points_card(res: dict) -> str:
    """目标价位区间（5/7/30 日各一档）+ 买卖点预判。"""
    tp = res.get("trade_points") or {}
    rows = tp.get("per_horizon") or []
    if not rows:
        return ""
    trs = ""
    for r in rows:
        cls = "up" if r["expected_ret"] > 0 else ("down" if r["expected_ret"] < 0 else "")
        dir_txt = r["direction"] + ("（无方向预测力）" if r.get("no_skill") else "")
        trs += (f"<tr><td>{r['horizon']} 日</td><td>{html.escape(str(r['date']))}</td>"
                f"<td>{r['lower']:.2f}</td><td><b>{r['target']:.2f}</b></td><td>{r['upper']:.2f}</td>"
                f"<td class='{cls}'>{r['expected_ret']:+.2f}%</td><td>±{r['band_pct']:.2f}%</td>"
                f"<td>{r['prob_up'] if r['prob_up'] is not None else '-'}</td>"
                f"<td>{html.escape(dir_txt)}</td></tr>")
    bp, sp, rk = tp.get("buy_points", {}), tp.get("sell_points", {}), tp.get("risk", {})
    bl = ""
    if "pullback_zone" in bp:
        ma20 = ""
        if bp.get("ma20_price"):
            role = bp.get("ma20_role") or "参考位"
            ma20 = (f"<div>MA20 动态{html.escape(role)}：<b>{bp['ma20_price']:.2f}</b>"
                    + ("（现价在 MA20 下方，该位为上方压力）" if role == "阻力" else "") + "</div>")
        bl += (f"<div>分批建仓区（低吸）：<b>{bp['pullback_zone'][0]:.2f} ~ {bp['pullback_zone'][1]:.2f}</b>"
               f"<div class='muted'>{html.escape(bp['pullback_note'])}</div></div>"
               f"<div>突破确认价（追强）：<b>{bp['breakout_trigger']:.2f}</b>"
               f"<div class='muted'>{html.escape(bp['breakout_note'])}</div></div>"
               f"<div>短期强支撑：<b>{bp['short_term_support']:.2f}</b></div>{ma20}")
    sl = ""
    if "take_profit_zone" in sp:
        atr = (f"<div>ATR 参考止损：{sp['atr_stop']:.2f}（1.5×ATR）</div>"
               if sp.get("atr_stop") else "")
        sl += (f"<div>止盈减仓区：<b>{sp['take_profit_zone'][0]:.2f} ~ {sp['take_profit_zone'][1]:.2f}</b>"
               f"<div class='muted'>{html.escape(sp['take_profit_note'])}</div></div>"
               f"<div>止损位：<b class='down'>{sp['stop_loss']:.2f}</b>（{sp['stop_loss_pct']:+.2f}%）"
               f"<div class='muted'>{html.escape(sp['stop_loss_note'])}</div></div>{atr}")
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in tp.get("notes", []))
    return f"""<div class="card">
      <h3>目标价位区间 + 买卖点预判</h3>
      <div class="muted">每 5 / 7 / 30 个交易日各出一档结果。带宽 = max(多模型分歧, 波动率外推) × {rk.get('band_z')}，
      越宽说明模型分歧或波动越大 → 越不该重仓。</div>
      <table><thead><tr><th>周期</th><th>到达日</th><th>下界</th><th>中位</th><th>上界</th>
      <th>预期</th><th>带宽</th><th>涨概率</th><th>方向</th></tr></thead><tbody>{trs}</tbody></table>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:10px">
        <div><h4>买点</h4>{bl}</div>
        <div><h4>卖点</h4>{sl}</div>
      </div>
      <div class="muted" style="margin-top:8px">风险等级 {html.escape(str(rk.get('risk_level')))} ｜
      ATR {rk.get('atr_pct')}% ｜ 日波动 {rk.get('daily_vol_pct')}% ｜ 单笔风险预算 {rk.get('risk_budget_pct')}%</div>
      {f'<ul class="muted">{notes}</ul>' if notes else ''}
    </div>"""


def _llm_decision_card(res: dict) -> str:
    """大模型对联合决策的二次解读（解释层，不是独立信号、不参与加权）。"""
    d = res.get("llm_decision") or {}
    if not d.get("ok"):
        return f"""<div class='card'><h3>大模型综合研判</h3>
          <div class='muted'>{html.escape(d.get('note') or '未生成解读')}</div>
          <div class='muted'>说明：本段是把量化决策交给大模型做的二次解读，属于解释层，
          <b>不参与加权、也不是独立信号</b>。</div></div>"""
    stance_cls = "up" if d["final_stance"] == "偏多" else ("down" if d["final_stance"] == "偏空" else "muted")

    def _ul(items):
        if not items:
            return "<li class='muted'>（无）</li>"
        return "".join(f"<li>{html.escape(x)}</li>" for x in items)

    return f"""<div class='card'>
      <h3>大模型综合研判（对量化决策的二次解读）</h3>
      <div class="kv"><span class='tag'>{html.escape(d['provider'])} / {html.escape(d['model'])}</span>
        <span>大模型口径：<b class='{stance_cls}'>{html.escape(d['final_stance'])}</b></span>
        <span class='muted'>耗时 {d.get('elapsed', 0)}s</span></div>
      <div style="margin-top:8px;font-size:14px;line-height:1.7">{html.escape(d['conclusion'])}</div>
      <div class="kv" style="margin-top:10px">
        <div style="flex:1"><b>支撑该结论</b><ul>{_ul(d.get('agrees'))}</ul></div>
        <div style="flex:1"><b>矛盾/需警惕</b><ul>{_ul(d.get('diverges'))}</ul></div>
        <div style="flex:1"><b>未来 5~10 日盯什么</b><ul>{_ul(d.get('watch'))}</ul></div>
      </div>
      <div class="muted">⚠️ 本段是"解释层"：把上面的数字翻译成人能读懂的判断，
      <b>不增加新的信号权重、不参与联合决策计算</b>。数字仍是主结论，文字只是辅助理解。</div>
    </div>"""


def _turn_card(res: dict) -> str:
    tr = res.get("turn") or {}
    if not tr.get("probs"):
        return """<div class="card"><h3>拐点预测</h3>
          <div class="muted">样本不足，拐点模型未训练成功。</div></div>"""
    rows = ""
    for h in sorted(tr["probs"]):
        p = tr["probs"][h]
        b = (tr.get("base") or {}).get(h) or {}
        pt, pb = p.get("top", 0), p.get("bot", 0)
        bt_, bb = b.get("top", 0), b.get("bot", 0)
        rows += (f"<tr><td>{h} 日</td>"
                 f"<td class='{'down' if pt > bt_ else ''}'>{pt*100:.1f}% <span class='muted'>(基线{bt_*100:.1f}%)</span></td>"
                 f"<td class='{'up' if pb > bb else ''}'>{pb*100:.1f}% <span class='muted'>(基线{bb*100:.1f}%)</span></td></tr>")
    days = tr.get("days") or {}
    day_txt = "；".join(
        f"预计{'顶' if k=='top' else '底'}拐点约 <b>{v['point']:.0f}</b> 个交易日后（区间 {v['lo']:.0f}~{v['hi']:.0f} 日）"
        for k, v in days.items() if v)
    recent = tr.get("recent") or []
    rec_txt = "、".join(f"{r['date']} {r['kind']}（{r['price']}，{r['days_ago']}日前）" for r in recent[:4]) or "近 20 个交易日无已确认拐点"
    return f"""<div class="card">
      <h3>拐点预测（顶 / 底）</h3>
      <table><thead><tr><th>周期</th><th>出现顶拐点概率</th><th>出现底拐点概率</th></tr></thead><tbody>{rows}</tbody></table>
      <div class="sub" style="margin-top:8px">{day_txt}</div>
      <div class="sub">近期已确认拐点：{html.escape(rec_txt)}</div>
      <div class="muted">拐点定义：收盘价的局部极值（左右各 {config.TURN_CONFIRM_WIN} 根内最高/最低）且随后反向幅度 ≥ {config.TURN_MIN_SWING*100:.1f}%。
      概率是「绝对概率」，必须对照括号里的基线看：高于基线才代表模型认为更可能发生。</div>
    </div>"""


def _pattern_card(res: dict) -> str:
    pat = res["pattern"]
    items = pat.get("items") or []
    if items:
        lis = "".join(
            f"<li><b>{it['date']}</b> {html.escape(it['name'])}"
            f"（{'利多' if it['direction']>0 else ('利空' if it['direction']<0 else '中性')}，强度 {it['strength']:.2f}）"
            f"<span class='muted'> — {html.escape(it['cn'])}</span></li>"
            for it in items[:8])
    else:
        lis = "<li>最近 3 根 K 线未命中经典形态</li>"
    return f"""<div class="card">
      <h3>技术面 · K 线形态识别</h3>
      <div class="kv"><span>形态分 <b>{pat['score']:.2f}</b>（0.5 中性）</span>
        <span>多头强度 {pat['bull']:.2f}</span><span>空头强度 {pat['bear']:.2f}</span>
        <span>{html.escape(pat['label'])}</span></div>
      <ul>{lis}</ul>
      <div class="muted">识别 19 类经典形态（吞没/锤子/星线/红三兵/缺口/突破等），带趋势上下文判定方向，
      并按 0.7^age 时衰减聚合；同一形状出现在上涨末端会被判为看跌（如上吊线/流星线）。</div>
    </div>"""


def _volume_card(res: dict) -> str:
    vol = res["volume"]
    rows = "".join(
        f"<tr><td style='text-align:left'>{html.escape(i['name'])}</td><td>{html.escape(str(i['value']))}</td>"
        f"<td class='{'up' if i['contrib']>0 else ('down' if i['contrib']<0 else '')}'>{i['contrib']:+.3f}</td>"
        f"<td style='text-align:left'>{html.escape(i['note'])}</td></tr>"
        for i in vol["items"])
    flow = res.get("flow") or {}
    flow_html = ""
    if flow.get("ok"):
        t = flow["tiers"]
        flow_html = (f"<div class='sub' style='margin-top:8px'>主力净额 {flow['net_amount_wan']:+.0f} 万元"
                     f"（占成交额 {flow['net_ratio']:+.2f}%）｜换手 {flow.get('turnover', 0):.2f}%</div>"
                     f"<table><thead><tr><th>分单</th><th>买入(万)</th><th>卖出(万)</th><th>净额(万)</th></tr></thead><tbody>"
                     + "".join(f"<tr><td>{k}</td><td>{v['in']/1e4:.0f}</td><td>{v['out']/1e4:.0f}</td>"
                               f"<td class='{'up' if v['net']>0 else 'down'}'>{v['net']/1e4:+.0f}</td></tr>"
                               for k, v in t.items())
                     + "</tbody></table>")
    else:
        flow_html = f"<div class='muted'>资金流不可用：{html.escape(flow.get('reason', '未知'))}</div>"
    return f"""<div class="card">
      <h3>量能与买卖盘</h3>
      <div class="kv"><span>量能分 <b>{vol['score']:.2f}</b></span><span>{html.escape(vol['label'])}</span></div>
      <table><thead><tr><th>指标</th><th>取值</th><th>贡献</th><th>解读</th></tr></thead><tbody>{rows}</tbody></table>
      {flow_html}
      <div class="muted">资金流/五档盘口只有当前快照、没有历史序列，因此**不作为模型特征**，只在决策层作当期证据。</div>
    </div>"""


def _news_card(res: dict) -> str:
    llm = res["llm"]
    g = res["news_groups"]
    counts = (f"个股 {len(g.get('stock', []))} 条 / 政策 {len(g.get('policy', []))} 条 / "
              f"市场 {len(g.get('market', []))} 条")
    if llm.get("ok"):
        head = (f"<div class='kv'><span class='tag'>大模型 {html.escape(llm['provider'])} / "
                f"{html.escape(llm['model'])}</span>"
                f"<span>立场 <b>{html.escape(llm['stance'])}</b></span>"
                f"<span>情绪 {llm['sentiment']:.3f}</span><span>政策影响 {llm['policy_impact']:+.2f}</span>"
                f"<span>置信 {llm['confidence']:.2f}</span>"
                f"<span>耗时 {llm.get('elapsed', 0)}s{'（缓存）' if llm.get('cached') else ''}</span></div>")
        summary = f"<div style='margin-top:8px'>{html.escape(llm.get('summary',''))}</div>"
        drivers = "".join(f"<li>{html.escape(x)}</li>" for x in llm.get("key_drivers", []))
        risks = "".join(f"<li>{html.escape(x)}</li>" for x in llm.get("risks", []))
        body = (f"{summary}<div class='kv' style='margin-top:8px'><div><b>关键驱动</b><ul>{drivers}</ul></div>"
                f"<div><b>风险</b><ul>{risks}</ul></div></div>")
    else:
        head = "<div class='kv'><span class='tag warn'>未接入大模型</span></div>"
        body = (f"<div class='sub'>{html.escape(llm.get('note', ''))}</div>"
                f"<div style='margin-top:6px'>本地词典情绪：{html.escape(res['news_local']['label'])}"
                f"（{res['news_local']['mean']:.3f}，样本 {res['news_local']['count']} 条）</div>"
                "<div class='muted'>配置 DASHSCOPE_API_KEY（千问）或 DEEPSEEK_API_KEY 后重启，"
                "即可获得事件级/政策级归因。</div>")
    titles = "".join(f"<li>{html.escape(it.title)}</li>"
                     for it in (g.get("stock") or [])[:6] + (g.get("policy") or [])[:3])
    return f"""<div class="card">
      <h3>新闻与政策（大模型解读）</h3>
      <div class="sub">{counts}</div>
      {head}{body}
      <div style="margin-top:8px"><b class="muted">抓到的部分标题</b><ul class="muted">{titles}</ul></div>
    </div>"""


def _pred_tables(res: dict) -> str:
    """各周期目标价：**每档只有一行**。

    原来的实现把 preds[h] 的每一行都列出来，30 日就列 30 行，
    看起来像"逐日预测了未来 30 天"。实际模型每档只做一次 DIRMO 预测
    （直接预测第 N 个交易日收盘价），中间那些点是为了画图连续的插值，
    不能当逐日预测展示。
    """
    base = float(res.get("last_close") or 0)
    rows = ""
    for h in sorted(res.get("preds", {})):
        df = res["preds"].get(h)
        if df is None or df.empty:
            continue
        last = df.iloc[-1]
        tgt = float(last["pred_close"])
        band = float(last["band_ret"]) if "band_ret" in df.columns else 0.0
        dcls = "up" if last["pred_dir"] == "涨" else ("down" if last["pred_dir"] == "跌" else "muted")
        ret = (tgt / base - 1) * 100 if base else 0.0
        lo = base * (1 + ret / 100 - band) if base else tgt
        hi = base * (1 + ret / 100 + band) if base else tgt
        rows += (f"<tr><td>{h} 日</td><td>{last['date']}</td><td><b>{tgt:.2f}</b></td>"
                 f"<td class='{'up' if ret > 0 else ('down' if ret < 0 else '')}'>{ret:+.2f}%</td>"
                 f"<td>{lo:.2f} ~ {hi:.2f}</td>"
                 f"<td>{float(last['prob_up']):.3f}</td>"
                 f"<td class='{dcls}'>{last['pred_dir']}</td></tr>")
    if not rows:
        return ""
    return f"""<div class="card"><h3>各周期目标价</h3>
      <div class="muted">每档只预测「第 N 个交易日的收盘价」，是<b>一次性</b>预测，
      <b>不是逐日预测</b>（不会预测未来 30 天的每一天）。区间 = 目标价 ± 不确定性带宽。</div>
      <table><thead><tr><th>周期</th><th>预测日</th><th>预测收盘</th><th>相对现价</th>
      <th>区间</th><th>上涨概率</th><th>方向</th></tr></thead>
      <tbody>{rows}</tbody></table></div>"""


def _backtest_card(res: dict) -> str:
    bt = res.get("backtest") or {}
    summ = bt.get("summary")
    if bt.get("note"):
        return f"<div class='card'><h3>回测</h3><div class='muted'>{html.escape(bt['note'])}</div></div>"
    if summ is None or len(summ) == 0:
        return ""
    cols = list(summ.columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    rows = "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>" for _, r in summ.iterrows())
    turn_eval = res.get("turn_eval")
    extra = ""
    if turn_eval is not None and len(turn_eval):
        ec = list(turn_eval.columns)
        extra = ("<h3 style='margin-top:14px'>拐点模型样本外评估</h3><table><thead><tr>"
                 + "".join(f"<th>{c}</th>" for c in ec) + "</tr></thead><tbody>"
                 + "".join("<tr>" + "".join(f"<td>{r[c]}</td>" for c in ec) + "</tr>"
                           for _, r in turn_eval.iterrows()) + "</tbody></table>")
    return f"""<div class="card"><h3>回测误差指标（样本外，与真实价对比）</h3>
      <table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>
      {extra}
      <div class="muted">方向准确率接近 50% 是常态；「明确时准确率」指模型概率明显偏离 0.5 时的准确率。
      30 日方向经大样本验证无预测力，故不显示涨跌方向。</div></div>"""


def render_result(res: dict) -> str:
    if res.get("error"):
        return (PAGE_HEAD + f"<div class='wrap'><div class='card err'>{html.escape(res['error'])}</div>"
                + "<div class='card'><a href='/'>← 返回重新输入</a></div></div></body></html>")
    timings = res.get("timings") or {}
    meta = (f"<div class='card'><span class='tag'>模型 {html.escape(str(res.get('model_name','rf')))}</span> "
            f"<span class='tag neutral'>特征集 {html.escape(str(res.get('feature_set','core')))}</span> "
            f"<span class='tag neutral'>标签 {html.escape(str(res.get('label_mode','ret')))}</span> "
            f"<span class='muted'>{html.escape(res.get('model_note') or '')}</span></div>")
    body = (
        _quote_card(res)
        + meta
        + _decision_card(res)
        + _trade_points_card(res)
        + _llm_decision_card(res)
        + _turn_card(res)
        + _pattern_card(res)
        + (f"<div class='card'><h3>K 线形态图（近 60 根，红涨绿跌）</h3>"
           f"<img src='data:image/png;base64,{res['candles_b64']}'></div>"
           if res.get("candles_b64") else "")
        + _volume_card(res)
        + _news_card(res)
        + _pred_tables(res)
        + f"<div class='card'><h3>预测走势图（含预计拐点竖线）</h3>"
          f"<img src='data:image/png;base64,{res['chart_b64']}'></div>"
        + _backtest_card(res)
        + f"<div class='card muted'>结果仅用于技术研究，不构成投资建议。"
          f"耗时 {html.escape(str(timings))}</div>"
        + "<div class='card'><a href='/'>← 返回重新选择股票</a></div>"
    )
    return PAGE_HEAD + f"<div class='wrap'>{body}</div></body></html>"


def _mask(k: str) -> str:
    if not k:
        return "未设置"
    return (k[:4] + "…" + k[-4:]) if len(k) > 10 else "已设置"


def render_home(sess: dict | None = None, notice: str = "") -> str:
    sess = sess or {}
    favs = " ".join(f"<a href='/analyze?code={c}&name={html.escape(n)}'>{html.escape(n)} {c}</a>"
                    for c, n in FAVORITES)
    st = llm_status()
    has_own = bool(sess.get("api_key"))
    if has_own:
        llm_line = (f"<span class='tag'>本次会话使用你填的 Key：{html.escape(_mask(sess['api_key']))}"
                    f" / {html.escape(sess.get('model') or '默认')}</span>")
    elif st["ready"]:
        llm_line = (f"<span class='tag'>未填 Key，将使用服务器默认：{html.escape(st['providers'][0])} / "
                    f"{html.escape(st['model'])}</span>")
    else:
        llm_line = ("<span class='tag warn'>未填 Key，且服务器无默认</span> "
                    "<span class='muted'>新闻/政策研判会降级为本地词典情绪</span>")

    tips = "　".join(f"{html.escape(n)}" for n, _, _ in COMMON_ENDPOINTS)
    notice_html = f"<div class='card' style='border-color:#86efac;background:#f0fdf4'>{html.escape(notice)}</div>" if notice else ""

    # 模型下拉：标注哪些是免费额度；当前会话选中的项打勾
    cur_model = sess.get("model") or ""
    cur_base = (sess.get("base_url") or "").rstrip("/")
    opts = []
    for label, base, model, free in MODEL_CHOICES:
        sel = " selected" if (model == cur_model and base == cur_base) else ""
        tag = "（免费额度）" if free else ""
        opts.append(f"<option value='{html.escape(base)}|{html.escape(model)}'{sel}>"
                    f"{html.escape(label)}</option>")
    model_options = "".join(opts)

    body = f"""{notice_html}<div class="card">
      <h2>股价走势预测 · 形态 / 量能 / 新闻政策 / 拐点</h2>
      <div class="muted" style="margin-bottom:10px">输入<b>任意 A 股 6 位代码</b>即可预测
      （沪深300 池内外的股票都支持，池外会自动并入训练池）。</div>
      <form method="get" action="/analyze">
        <div><label>股票代码</label><br>
          <input type="text" name="code" placeholder="如 600519 / 000001 / 688836" required></div>
        <div><label>名称（可选）</label><br>
          <input type="text" name="name" placeholder="如 贵州茅台"></div>
        <div><button type="submit">查询并预测</button></div>
      </form>
      <div class="fav" style="margin-top:12px">热门：{favs}</div>
      <div style="margin-top:14px">{llm_line}</div>
    </div>

    <div class="card">
      <h3>大模型设置</h3>
      <div class="muted"><b>Key 输入框是可选的</b>：留空就用服务器内置的 Key（已配置好，直接可用）；
      想用自己的额度再填，填了只对<b>你这次浏览</b>生效，仅存在服务端内存里
      （不写日志、不落盘、不进网址），关掉浏览器或服务重启即失效。</div>
      <form method="post" action="/settings">
        <div style="margin-top:10px"><label>API Key（留空 = 用服务器内置 Key）</label><br>
          <input type="password" name="api_key" placeholder="留空即可；或填 sk-... / dashscope key"
                 style="width:100%;max-width:520px" autocomplete="off"></div>
        <div style="margin-top:8px"><label>模型（下拉里带「免费额度」标记的不烧钱）</label><br>
          <select name="model_choice" style="width:100%;max-width:520px;padding:9px 10px;
                  border:1px solid #dfe3e8;border-radius:8px;font-size:14.5px">
            {model_options}
            <option value="||{html.escape(cur_model)}" >自定义（下方手填）</option>
          </select></div>
        <div style="margin-top:8px"><label>模型名（自定义时手填，留空按地址自动选）</label><br>
          <input type="text" name="model" value="{html.escape(cur_model)}"
                 placeholder="qwen-plus / qwen3.7-plus / deepseek-chat" style="width:100%;max-width:520px"></div>
        <div style="margin-top:8px"><label>接口地址（OpenAI 兼容，下拉已自动填）</label><br>
          <input type="text" name="base_url"
                 value="{html.escape(sess.get('base_url') or COMMON_ENDPOINTS[0][1])}"
                 style="width:100%;max-width:520px"></div>
        <div style="margin-top:10px">
          <button type="submit">保存</button>
          <button type="submit" formaction="/verify">保存并测试连接</button>
        </div>
      </form>
      <div class="muted" style="margin-top:8px">常用端点：{tips}</div>
      <div class="muted" style="margin-top:6px">
        <b>免费额度怎么用</b>：阿里云百炼对新用户每个模型各送 100 万 tokens（够跑上千次研判），
        选上面标「免费额度」的模型即可；<code>qwen-turbo</code> 免费但偶尔被限流(429)，
        限流时换 <code>qwen-plus</code> 或 <code>qwen3.7-plus</code>。</div>
    </div>

    <div class="card muted">
      数据：新浪财经（历史K线/实时行情/五档盘口/资金流/滚动资讯）+ 东方财富搜索（个股与政策新闻）。<br>
      信号：多模型融合（RF+GBDT+LSTM）涨跌与价格预测 + K线形态 + 量能资金 + 新闻政策（大模型）+ 拐点模型，加权成联合决策。<br>
      首次分析约需 3-6 分钟（含 300 只股票池训练），页面会一直等待直至完成，请不要重复刷新。
    </div>"""
    return PAGE_HEAD + f"<div class='wrap'>{body}</div></body></html>"


def render_verify(ok: bool, msg: str, sess: dict) -> str:
    cls = "border-color:#86efac;background:#f0fdf4" if ok else "border-color:#fecaca;background:#fef2f2"
    return PAGE_HEAD + f"""<div class='wrap'>
      <div class='card' style='{cls}'>
        <h2>{'✓ 连接成功' if ok else '✗ 连接失败'}</h2>
        <div style="margin-top:8px">{html.escape(msg)}</div>
      </div>
      <div class='card muted'>当前会话密钥：{html.escape(_mask(sess.get('api_key', '')))}　
        /　模型：{html.escape(sess.get('model') or '自动')}　
        /　地址：{html.escape(sess.get('base_url') or '')}</div>
      <p><a href="/">← 返回首页</a></p>
    </div></body></html>"""


# ------------------------- HTTP 服务 -------------------------
def analyze_page(symbol: str, name: str, llm_override: dict | None = None) -> dict:
    try:
        res = analyze(symbol, name=name, horizons=list(config.HORIZONS),
                      llm_override=llm_override)
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        chart_path = os.path.join(config.OUTPUT_DIR, f"_web_forecast_{symbol}.png")
        plot_forecast(res["kline"], res["preds"], chart_path, name=res["name"],
                      max_horizon=max(config.HORIZONS), turn=res.get("turn"))
        with open(chart_path, "rb") as f:
            res["chart_b64"] = base64.b64encode(f.read()).decode()
        cd_path = os.path.join(config.OUTPUT_DIR, f"_web_candles_{symbol}.png")
        plot_candles(res["kline"], cd_path, name=res["name"], bars=60,
                     patterns=detect_patterns(res["tech"], window=12))
        with open(cd_path, "rb") as f:
            res["candles_b64"] = base64.b64encode(f.read()).decode()
        res["horizons"] = list(config.HORIZONS)
        return res
    except Exception as e:
        return {"error": f"分析失败：{e}"}


class Handler(BaseHTTPRequestHandler):
    _sid: str | None = None

    # ---------------- 会话（cookie 只存随机 id，密钥留在服务端内存）
    def _session(self) -> tuple[str, dict]:
        cookie = self.headers.get("Cookie", "") or ""
        m = re.search(r"(?:^|;\s*)sid=([A-Za-z0-9_-]{10,64})", cookie)
        sid = m.group(1) if m else None
        with _SESS_LOCK:
            if not sid or sid not in SESSIONS:
                sid = _secrets.token_urlsafe(18)
                SESSIONS[sid] = {}
            sess = SESSIONS[sid]
        self._sid = sid
        return sid, sess

    @staticmethod
    def _override_of(sess: dict) -> dict | None:
        ov = {"api_key": sess.get("api_key", ""), "base_url": sess.get("base_url", ""),
              "model": sess.get("model", "")}
        return ov if _override_target(ov) else None

    def _send(self, html_text: str, code: int = 200):
        data = html_text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        # 动态页（等待页/表单）绝不能被浏览器缓存，否则用户会一直看到旧状态
        self.send_header("Cache-Control", "no-store, must-revalidate")
        if self._sid:
            self.send_header("Set-Cookie",
                             f"sid={self._sid}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, to: str):
        self.send_response(303)
        self.send_header("Location", to)
        if self._sid:
            self.send_header("Set-Cookie",
                             f"sid={self._sid}; Path=/; HttpOnly; SameSite=Lax")
        self.end_headers()

    # ---------------- 静态文件（deploy_site）与报告生成
    def _serve_file(self, full: str) -> bool:
        if not os.path.isfile(full):
            return False
        ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript", "application/json"):
            ctype += "; charset=utf-8"
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)
        return True

    def _serve_stale_report(self, full: str, code: str) -> bool:
        """报告存在但数据已落后：照常输出内容，但顶部插入"数据已过期"提示条。

        不在这里自动重跑（浏览是高频行为，自动起训练进程会打满机器），
        过期与否交给用户点「重新分析」触发。
        """
        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return False
        day = _report_data_day(full)
        latest = _latest_trading_day()
        gap = ""
        if day and latest:
            try:
                gap = (dt.date.fromisoformat(latest) - dt.date.fromisoformat(day)).days
            except ValueError:
                gap = ""
        tip = (
            f'<div style="position:sticky;top:0;z-index:99;background:#fffbeb;'
            f'border-bottom:2px solid #f59e0b;padding:12px 18px;font-size:14px;'
            f'font-family:-apple-system,Segoe UI,Microsoft YaHei,sans-serif">'
            f'<b>⚠ 这份报告的数据截至 {html.escape(day)}，不是最新交易日</b>'
            + (f'（行情源最新是 {html.escape(latest)}，相差 {gap} 天）' if gap != "" else "")
            + f'。页面内容按 {html.escape(day)} 收盘价计算，期间的价格变化未计入。'
            f'<a href="/report?code={code}&refresh=1" '
            f'style="color:#2563eb;font-weight:600;margin-left:8px">→ 重新分析这只（约 3~6 分钟）</a>'
            f'<a href="/" style="color:#646a73;margin-left:12px">← 返回主页</a></div>'
        )
        # 插在 <body> 之后（报告模板都是 <body> 紧跟内容）
        if "<body" in text:
            i = text.index("<body")
            j = text.index(">", i) + 1
            text = text[:j] + tip + text[j:]
        else:
            text = tip + text
        raw = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)
        return True

    def _serve_static(self, rel: str) -> bool:
        rel = (rel or "").lstrip("/").replace("\\", "/")
        if not rel or rel.endswith("/"):
            return False
        full = os.path.normpath(os.path.join(SITE, rel))
        root = os.path.normpath(SITE)
        if not full.startswith(root + os.sep):
            return False
        return self._serve_file(full)

    def _handle_report(self, qs: dict):
        # 这个方法在 do_GET 之外被调用，会话要自己取（上次漏了这行，访客 Key 一律拿不到，
        # 而且 NameError 会把连接直接掐断 → 浏览器看到 502）
        _, sess = self._session()
        raw = (qs.get("code", [""])[0] or "").strip()
        code = re.sub(r"\D", "", raw)
        code = code[-6:] if len(code) > 6 else code
        if len(code) != 6:
            self._send(_report_form("请输入 6 位股票代码，例如 600519 / 000001 / 601318"))
            return
        force = (qs.get("refresh", ["0"])[0] == "1")
        rp = os.path.join(config.OUTPUT_DIR, f"report_{code}.html")
        # output/ 里没有、但静态站里带了预生成报告时，直接走查看页复用，
        # 不必为一个已经存在的报告再等 3~6 分钟重新训练（要强制重跑加 refresh=1）。
        if not force and not os.path.exists(rp) and os.path.exists(
                os.path.join(SITE, f"report_{code}.html")):
            self._redirect(f"/report/view?code={code}")
            return

        # 1) 已有且够新的报告：303 跳到查看页（等待页刷新到这儿就会自动变成报告）
        #    新鲜度按**数据日**判（见 _report_is_stale），不按文件年龄。
        if not force and not _report_is_stale(rp):
            self._redirect(f"/report/view?code={code}")
            return

        # 2) 没有/过期：提交或复用后台任务（带访客自带的 Key，没有就用服务器默认）
        j = _report_job(code, force=force,
                        api_key=(sess.get("api_key") or "").strip(),
                        model=(sess.get("model") or "").strip(),
                        base_url=(sess.get("base_url") or "").strip())
        if j is None:                      # 抢在别的请求里刚生成完
            self._redirect(f"/report/view?code={code}")
            return
        if j.get("failed"):                # 熔断中：这只票反复失败，不再起新进程
            self._send(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>暂时无法生成 {code}</title><style>body{{font:15px/1.7 -apple-system,'Microsoft YaHei';
padding:40px;background:#f5f6f8}}</style></head><body>
<h2>这只票暂时跑不出报告（已连续失败 {j['failed']} 次，冷却中）</h2>
<p>常见原因：代码不存在 / 不是 A 股股票 / 已退市 / 停牌 / 数据源拿不到历史数据。</p>
<p>请确认代码是 6 位 A 股代码；如果是 ETF、港股或美股，本系统不覆盖。</p>
<p><a href="/report?code={code}&refresh=1">强制重试</a>　·　<a href="/">← 返回主页</a></p>
</body></html>""", code=200)
            return
        if j.get("queued"):                # 并发已满：排队，不起新进程
            self._send(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>排队中 {code}</title><style>body{{font:15px/1.7 -apple-system,'Microsoft YaHei';
padding:40px;background:#f5f6f8}}.wrap{{max-width:640px;margin:40px auto;background:#fff;
border:1px solid #dfe3e8;border-radius:12px;padding:26px 30px}}</style></head><body>
<div class="wrap"><h2>服务器正在跑其它分析，你排在第 {j['queued'] + 1} 位</h2>
<p>一次分析要 1.5GB 内存和 3~6 分钟 CPU，同时最多并行 {MAX_CONCURRENT_JOBS} 个，
所以这里限流了。这一页每 20 秒自动刷新，轮到你就自动开始。</p>
<p>如果一直排不上，可以先关掉页面、过几分钟再回来提交。</p>
<p><a href="/">← 返回主页</a></p></div></body></html>""", code=200)
            return
        if j["proc"] is None or j["proc"].poll() is None:   # 还在跑 → 等待页（自动跳转）
            self._send(_job_wait_page(code, j))
            return
        # 3) 任务已结束
        if os.path.exists(rp):
            _note_job_result(code, True)
            self._redirect(f"/report/view?code={code}")
            return
        _note_job_result(code, False)
        tail = ""
        try:
            tail = _sanitize_log(open(j["log"], encoding="utf-8", errors="replace").read())[-1200:]
        except OSError:
            pass
        self._send(f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<title>生成失败 {code}</title><style>body{{font:15px/1.7 -apple-system,'Microsoft YaHei';padding:40px;
background:#f5f6f8}}pre{{background:#fff;border:1px solid #edf0f3;border-radius:8px;padding:12px;
font-size:12px;white-space:pre-wrap;overflow-x:auto}}</style></head><body>
<h2>没能生成 {code} 的报告</h2>
<p>常见原因：代码不存在 / 停牌 / 数据源暂时不可用 / 该股在沪深300池外导致样本不足。</p>
<pre>{html.escape(tail)}</pre>
<p><a href="/report?code={code}&refresh=1">重试</a>　·　<a href="/">← 返回主页</a></p>
</body></html>""", code=200)

    def do_GET(self):
        _, sess = self._session()
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/__local":
            # 哨兵端点：只存在于本地服务。docs/index.html 用它判断
            # "当前是本地 Web（可真实训练）还是 GitHub Pages 静态站（只能看快照）"。
            raw = b"local"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/report":
            self._handle_report(parse_qs(parsed.query))
            return
        if path == "/report/status":
            scode = re.sub(r"\D", "", (parse_qs(parsed.query).get("code", [""])[0] or ""))[-6:]
            spath = os.path.join(config.OUTPUT_DIR, f"report_{scode}.html")
            ready = len(scode) == 6 and os.path.exists(spath)
            data = ('{"ready": true, "url": "/report/view?code=' + scode + '"}' if ready
                    else '{"ready": false}')
            raw = data.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        if path == "/report/view":
            vcode = re.sub(r"\D", "", (parse_qs(parsed.query).get("code", [""])[0] or ""))[-6:]
            vpath = os.path.join(config.OUTPUT_DIR, f"report_{vcode}.html")
            # ★ 新鲜度处理：/report/view 以前直接吐文件、从不检查过期，
            #   于是从首页点「看报告」链接进来的人会一直看到几天前的旧数据
            #   （实测：301330 报告数据日停在 10-21，而数据源已到 10-23）。
            #
            #   ★★ 但**绝不能在这里自动重跑**：浏览是高频只读行为，
            #   一过期就起训练子进程 → 用户刷新几次就能堆出几十个 1.5GB 进程把机器打满
            #   （实测 159562 一只票 20 分钟堆了 100+ 个）。
            #   正确做法：**照常把旧报告给出来**（内容仍有用），但在页面顶部
            #   明确标注数据已落后多少天，并给出「重新分析」按钮，由用户主动点。
            vforce = (parse_qs(parsed.query).get("refresh", ["0"])[0] == "1")
            if len(vcode) == 6 and not os.path.exists(vpath):
                # 回退到静态站里的同代码报告（docs/report_XXX.html）
                fallback = os.path.join(SITE, f"report_{vcode}.html")
                if os.path.exists(fallback):
                    vpath = fallback
            exists = len(vcode) == 6 and os.path.exists(vpath)
            if exists:
                if vforce:
                    self._redirect(f"/report?code={vcode}&refresh=1")
                    return
                if not _report_is_stale(vpath):
                    self._serve_file(vpath)
                    return
                # 过期但文件在：给旧报告 + 顶部提示条（不自动重跑）
                self._serve_stale_report(vpath, vcode)
                return
            self._send("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
                       "<meta http-equiv='refresh' content='5'></head><body style='font:15px -apple-system,"
                       "'Microsoft YaHei';padding:40px;background:#f5f6f8'>"
                       f"<h2>{vcode} 的报告还没生成好</h2><p>正在跑，约 3~6 分钟。这一页每 5 秒自动刷新。</p>"
                       "<p><a href='/report?code=" + vcode + "'>回到分析页</a>　·　<a href='/'>← 返回主页</a></p>"
                       "</body></html>", code=404)
            return
        if path in ("/", "/index.html") and self._serve_static("index.html"):
            return
        if path in ("/", "/index.html", "/classic"):
            notice = "设置已保存，本次会话生效。" if "saved=1" in parsed.query else ""
            self._send(render_home(sess, notice))
            return
        if self._serve_static(path):
            return
        if parsed.path == "/analyze":
            qs = parse_qs(parsed.query)
            code = (qs.get("code", [""])[0] or "").strip()
            name = (qs.get("name", [""])[0] or "").strip()
            if not code:
                self._send(render_home(sess))
                return
            self._send(render_result(analyze_page(code, name, self._override_of(sess))))
            return
        self._send(render_result({"error": "页面不存在"}), code=404)

    def do_POST(self):
        _, sess = self._session()
        parsed = urlparse(self.path)
        n = int(self.headers.get("Content-Length") or 0)
        form = parse_qs(self.rfile.read(n).decode("utf-8")) if n else {}

        def field(k):
            return (form.get(k, [""])[0] or "").strip()

        if parsed.path in ("/settings", "/verify"):
            # 空 key 表示"清空，改回服务器默认"；
            # 但**模型选择与 Key 解耦**：Key 留空也能单独换模型（走服务器内置 Key）。
            sess["api_key"] = field("api_key")
            sess["model"] = field("model")
            sess["base_url"] = field("base_url")
            # 下拉选了 "base|model" 就以它为准（自定义项为 "||手填名"，走上面的手填值）
            choice = field("model_choice")
            if choice:
                base_c, _, model_c = choice.partition("|")
                if base_c:
                    sess["base_url"] = base_c
                if model_c:
                    sess["model"] = model_c

            if parsed.path == "/settings":
                self._redirect("/?saved=1")
                return

            # 测试连接：发一个极小的请求确认 key / 模型 / 地址都对
            ov = self._override_of(sess)
            if not ov:
                self._send(render_verify(
                    False, "没有可用的 Key：服务器未配置内置密钥，且你也没填自己的。", sess))
                return
            r = llm_chat("只回复两个字：可用", system="你是一个连接测试。",
                         max_tokens=16, override=ov)
            if r.get("ok"):
                self._send(render_verify(
                    True, f"模型 {ov['model']} 正常返回：{r['text'][:60]}"
                          f"（{r.get('elapsed', 0)}s）", sess))
            else:
                self._send(render_verify(False, f"调用失败：{r.get('error', '未知错误')}", sess))
            return

        self._send(render_result({"error": "页面不存在"}), code=404)

    def log_message(self, *args):
        pass


def main():
    import argparse

    ap = argparse.ArgumentParser(description="股价预测网页")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    # 首次运行把依赖目录建好（output 放报告、.cache 放行情缓存），
    # 避免 clone 后第一次点击分析就因目录缺失报 500。
    for d in (config.OUTPUT_DIR, getattr(config, "CACHE_DIR", ".cache")):
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            pass
    srv = ThreadingHTTPServer((HOST, args.port), Handler)
    print(f"✓ 页面已启动： http://{HOST}:{args.port}  （Ctrl+C 退出）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
