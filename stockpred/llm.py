"""免费大模型接入：新闻/政策文本分析 → 结构化情绪与事件。

设计要点（都是踩坑后的结论）
--------------------------
1. **多供应商、OpenAI 兼容协议**：千问 DashScope / DeepSeek / 本地 Ollama 都用
   ``/chat/completions``，一份代码通吃，换供应商只改 base_url + model。
2. **密钥不进代码**：只从环境变量或项目根 ``secrets.json`` 读。
3. **无密钥必须能跑**：自动降级到本地 snownlp 词典情绪（明确标注 provider=fallback），
   绝不假装有大模型结论。
4. **输出严格 JSON 并做校验**：模型返回的分数可能越界/带 markdown 代码块，
   统一清洗 + 钳制到合法区间；解析失败按失败处理，不猜。
5. **同日同批新闻缓存**：避免页面刷新重复烧 token。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

_CACHE_DIR = Path(".cache") / "llm"
_SECRET_FILE = Path("secrets.json")

# 供应商 → 可能的环境变量名
_ENV_KEYS = {
    "qwen": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "ALIYUN_API_KEY"),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "ollama": (),   # 本地无需密钥
}


# ------------------------- 密钥 / 供应商解析 -------------------------
def _read_secrets() -> dict:
    if not _SECRET_FILE.exists():
        return {}
    try:
        return json.loads(_SECRET_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _key_for(provider: str) -> str:
    for env in _ENV_KEYS.get(provider, ()):
        v = os.environ.get(env, "").strip()
        if v:
            return v
    sec = _read_secrets()
    for k in ("llm_key", f"{provider}_key", f"{provider}_api_key",
              "dashscope_key" if provider == "qwen" else ""):
        if k and sec.get(k):
            return str(sec[k]).strip()
    # 通用兜底：显式配置了 LLM_PROVIDER 时允许 llm_key 直接用
    return ""


def _base_url(provider: str) -> str:
    sec = _read_secrets()
    if sec.get("llm_base") and sec.get("llm_provider") in (None, provider):
        return str(sec["llm_base"]).rstrip("/")
    return config.LLM_BASE_URLS.get(provider, "").rstrip("/")


def _model_for(provider: str) -> str:
    """模型名优先级：config.LLM_MODEL > secrets 里 {provider}_model/llm_model > config 默认。

    百炼免费额度是按模型独立计的（每个模型各 1M tokens），而且新用户额度里
    往往没有老名字 `qwen-plus`，只有 `qwen3.7-plus` 这类带版本号的新名字 ——
    所以允许在 secrets.json 里按供应商指定模型，不用改代码。
    """
    if config.LLM_MODEL:
        return config.LLM_MODEL
    sec = _read_secrets()
    for k in (f"{provider}_model", "llm_model"):
        if sec.get(k):
            return str(sec[k])
    return config.LLM_DEFAULT_MODELS.get(provider, "")


def available_providers() -> list[str]:
    """返回当前真正可用的供应商（按优先级）。"""
    out = []
    if not getattr(config, "LLM_ENABLED", True):
        return out
    order = [config.LLM_PROVIDER] if config.LLM_PROVIDER not in ("auto", "none") else \
            ["qwen", "deepseek", "ollama"]
    for p in order:
        if p == "ollama":
            # 本地服务需真的在监听，否则请求会拖慢整条链路
            if _ollama_alive():
                out.append(p)
            continue
        if _key_for(p) and _base_url(p):
            out.append(p)
    return out


def _ollama_alive(timeout: int = 2) -> bool:
    try:
        req = urllib.request.Request(_base_url("ollama") + "/models",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ------------------------- 底层调用 -------------------------
def _post_json(url: str, payload: dict, headers: dict, timeout: int) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _env_override() -> dict | None:
    """读环境变量里的 override（web.py 用它把访客选的模型传进子进程）。

    为什么不直接改 config.LLM_MODEL：那个是**全局默认**，会把所有供应商的模型
    都改成同一个名字（比如访客选了 deepseek-chat，千问也会去调它 → 401）。
    这里显式带上 base_url，只覆盖"访客选的那一个端点"。

    环境变量：
      LLM_OVERRIDE_BASE   端点（OpenAI 兼容）
      LLM_OVERRIDE_MODEL  模型名
      LLM_OVERRIDE_KEY    访客自带的 Key（可空 → 借服务器内置 Key）
    """
    base = os.environ.get("LLM_OVERRIDE_BASE", "").strip()
    model = os.environ.get("LLM_OVERRIDE_MODEL", "").strip()
    key = os.environ.get("LLM_OVERRIDE_KEY", "").strip()
    if not (base or model):
        return None
    return {"api_key": key, "base_url": base, "model": model}


def _override_target(override: dict | None) -> tuple[str, str, str, str] | None:
    """把网页传入的 {api_key, base_url, model} 变成 (label, base, model, key)。

    网页端允许用户自带密钥：这时**完全按用户填的走**，不再看环境变量 / secrets.json。
    model 留空时按 base_url 猜供应商，取 config 里的默认模型名。

    ★ 只有模型/地址、没填 Key 时：**借用服务器内置的 Key**（按 base_url 反查供应商）。
    这样「只换模型（换到免费额度模型）、不填 Key」也能生效 —— 网页上 Key 输入框
    才是真正"可选"而不是"必填"。
    """
    if not override:
        return None
    key = str(override.get("api_key") or "").strip()
    base = str(override.get("base_url") or "").strip().rstrip("/")
    model = str(override.get("model") or "").strip()
    if not base:
        # 只填了 Key 没填地址 → 走默认地址
        base = (config.LLM_BASE_URLS.get("qwen") or "").rstrip("/")
    if not base:
        return None
    label = "custom"
    server_provider = ""
    if not key:
        # 访客没填 Key：借服务器内置的 Key（按地址反查供应商）
        for p, u in (config.LLM_BASE_URLS or {}).items():
            if u and u.rstrip("/") == base:
                key = _key_for(p)
                if key:
                    label = f"server:{p}"
                    server_provider = p
                break
        if not key:
            return None          # 服务器也没 Key → 交回默认路径
    if not model:
        if server_provider:
            # 借服务器 Key 时沿用服务器配置的模型（如 secrets.json 里的 qwen3.7-plus），
            # 不要退回 config 默认的 qwen-plus —— 那是另一套额度。
            model = _model_for(server_provider)
        else:
            for p, u in (config.LLM_BASE_URLS or {}).items():
                if u and u.rstrip("/") in base:
                    model = config.LLM_DEFAULT_MODELS.get(p, "")
                    break
        model = model or "qwen-plus"
    return (label, base, model, key)


def chat(prompt: str, system: str = "", provider: str = "", temperature: float = 0.2,
         max_tokens: int = 900, override: dict | None = None) -> dict:
    """单轮对话。返回 {"ok", "text", "provider", "model", "error"}。

    override: 网页里填的 {"api_key","base_url","model"}。一旦提供就优先使用，
              忽略环境变量与 secrets.json（方便别人用自己的额度来试）。
    """
    targets: list[tuple[str, str, str, str]] = []
    # 优先级：显式 override（网页传参）> 环境变量 override（web.py 传给子进程）
    #          > 服务器默认（secrets.json / 环境变量 / config）
    ov = _override_target(override) or _override_target(_env_override())
    fallback: list[tuple[str, str, str, str]] = []
    if ov:
        targets.append(ov)
        # 借用服务器 Key 时允许回退：用户选了 qwen-turbo 这类容易被限流的模型，
        # 失败了还能退到服务器默认模型（用的是同一把 Key，不额外烧额度）。
        # 用户自带 Key 则不回退 —— 完全按用户填的走。
        if ov[0].startswith("server:"):
            for p in available_providers():
                base, model = _base_url(p), _model_for(p)
                if base and model:
                    fallback.append((p, base, model, _key_for(p)))
    else:
        for p in ([provider] if provider else available_providers()):
            base, model = _base_url(p), _model_for(p)
            if base and model:
                targets.append((p, base, model, _key_for(p)))
    if not targets:
        return {"ok": False, "error": "no_provider", "text": "",
                "provider": "", "model": ""}

    last_err = ""
    tried: set[tuple[str, str, str]] = set()
    for label, base, model, key in targets + fallback:
        sig = (base, model, key)
        if sig in tried:
            continue          # 同一个目标别重复试
        tried.add(sig)
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        payload = {"model": model, "messages": msgs, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        t0 = time.time()
        try:
            j = _post_json(base + "/chat/completions", payload, headers, config.LLM_TIMEOUT)
            text = ((j.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            if not text:
                last_err = f"{label}: 返回为空"
                continue
            return {"ok": True, "text": text.strip(), "provider": label, "model": model,
                    "elapsed": round(time.time() - t0, 2), "error": ""}
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            last_err = f"{label}: HTTP {e.code} {body}"
            # 401/403 是密钥问题，换下一个供应商；429 限流也换
        except Exception as e:
            last_err = f"{label}: {type(e).__name__} {e}"
    return {"ok": False, "error": last_err, "text": "", "provider": "", "model": ""}


# ------------------------- 结构化分析 -------------------------
_SYSTEM = (
    "你是 A 股资深策略分析师。只根据给定新闻，不编造事实，不使用未来信息。"
    "严格输出 JSON，不要输出任何解释文字或 markdown 代码块。"
)

_PROMPT_TMPL = """请分析下面这些与「{name}（{code}）」相关的新闻/政策材料，并给出对**未来 5~10 个交易日**股价的影响判断。

当前市场状态：
- 最新收盘价：{price}
- 近5日涨跌：{ret5:+.2f}%
- 近20日涨跌：{ret20:+.2f}%
- 当日量比（相对20日均量）：{vol_expand:.2f}
- 主力资金净额占成交额：{net_ratio:+.2f}%

新闻材料（共 {n} 条，标注类型）：
{news_block}

输出 JSON，字段与取值范围必须严格遵守：
{{
  "sentiment": 0到1的小数,          // 这些材料整体对该股的情绪，0.5=中性
  "policy_impact": -1到1的小数,     // 政策/监管对该公司的影响方向与力度
  "confidence": 0到1的小数,         // 你对上述判断的把握（材料少/含糊就给低分）
  "stance": "偏多" 或 "偏空" 或 "中性",
  "key_drivers": ["最多3条最关键的多空驱动，每条不超过25字"],
  "risks": ["最多2条主要风险，每条不超过25字"],
  "summary": "不超过80字的中文总结，说明这些消息如何影响股价"
}}
注意：如果材料与该股关系很弱，sentiment 请靠近 0.5、confidence 给低分、stance 给"中性"。"""


def _fmt_news(groups: dict, limit: int) -> list[str]:
    label = {"stock": "个股", "policy": "政策", "market": "市场"}
    lines = []
    for kind in ("stock", "policy", "market"):
        for it in (groups.get(kind) or []):
            lines.append(f"[{label.get(kind, '其他')}] {it.brief(120)}")
            if len(lines) >= limit:
                return lines
    return lines


def _clamp(x, lo, hi, default):
    try:
        v = float(x)
    except Exception:
        return default
    if v != v:  # NaN
        return default
    return max(lo, min(hi, v))


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


def _cache_path(key: str) -> Path:
    return _CACHE_DIR / f"{key}.json"


def _local_fallback(groups: dict, reason: str) -> dict:
    """无大模型时的降级：snownlp/词典情绪（明确标注来源，不冒充 LLM）。"""
    from .sentiment import analyze_news  # 延迟导入，避免循环

    flat = []
    for kind in ("stock", "policy", "market"):
        flat.extend(groups.get(kind) or [])
    s = analyze_news(flat)
    return {
        "ok": False,
        "provider": "fallback",
        "model": "snownlp/词典",
        "sentiment": s["mean"],
        "policy_impact": 0.0,
        "confidence": 0.2,
        "stance": {"偏看涨": "偏多", "偏看跌": "偏空"}.get(s["label"], "中性"),
        "key_drivers": [f"本地情绪分 {s['mean']:.3f}（{s['count']} 条材料）"],
        "risks": ["未接入大模型，无法做事件级/政策级归因"],
        "summary": "未配置大模型密钥，本结论仅由本地词典情绪得到，颗粒度较粗。",
        "news_count": len(flat),
        "note": reason,
        "elapsed": 0.0,
    }


def analyze_news_llm(groups: dict, code: str = "", name: str = "", price: float = 0.0,
                     ret5: float = 0.0, ret20: float = 0.0, vol_expand: float = 1.0,
                     net_ratio: float = 0.0, override: dict | None = None) -> dict:
    """用大模型分析新闻/政策，返回结构化结论（含降级路径）。

    override: 网页端传入的自带密钥，非空时优先于环境变量 / secrets.json。
    """
    flat_count = sum(len(groups.get(k) or []) for k in ("stock", "policy", "market"))
    if flat_count == 0:
        return _local_fallback(groups, "没有抓到任何新闻")

    lines = _fmt_news(groups, getattr(config, "LLM_MAX_NEWS", 30))
    prompt = _PROMPT_TMPL.format(
        name=name or code, code=code, price=f"{price:.2f}", ret5=ret5, ret20=ret20,
        vol_expand=vol_expand, net_ratio=net_ratio, n=len(lines),
        news_block="\n".join(lines),
    )
    ov = _override_target(override) or _override_target(_env_override())
    providers = available_providers()
    if not providers and not ov:
        return _local_fallback(groups, "未找到可用的大模型密钥/本地服务（可用环境变量 "
                                      "DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY 配置）")

    # 缓存：同一批新闻 + 同一个模型端点 → 复用（自带密钥时把 base_url 也算进 key）
    tag = f"{ov[1]}|{ov[2]}" if ov else f"{providers[0]}|{_model_for(providers[0])}"
    h = hashlib.sha1((tag + "|" + prompt).encode("utf-8")).hexdigest()[:16]
    cp = _cache_path(h)
    if cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            if time.time() - cached.get("_ts", 0) < getattr(config, "LLM_CACHE_TTL", 1800):
                cached["cached"] = True
                return cached
        except Exception:
            pass

    res = chat(prompt, system=_SYSTEM, override=override)
    if not res["ok"]:
        out = _local_fallback(groups, f"大模型调用失败：{res['error']}")
        return out

    obj = _extract_json(res["text"])
    if obj is None:
        out = _local_fallback(groups, "大模型返回无法解析为 JSON，已降级到本地情绪")
        out["raw"] = res["text"][:400]
        return out

    sentiment = _clamp(obj.get("sentiment"), 0, 1, 0.5)
    policy = _clamp(obj.get("policy_impact"), -1, 1, 0.0)
    conf = _clamp(obj.get("confidence"), 0, 1, 0.3)
    stance = str(obj.get("stance") or "")
    if stance not in ("偏多", "偏空", "中性"):
        stance = "偏多" if sentiment > 0.55 else ("偏空" if sentiment < 0.45 else "中性")

    def _lst(key, n):
        v = obj.get(key) or []
        if isinstance(v, str):
            v = [v]
        return [str(x)[:60] for x in v if str(x).strip()][:n]

    out = {
        "ok": True,
        "provider": res["provider"],
        "model": res["model"],
        "sentiment": round(sentiment, 4),
        "policy_impact": round(policy, 3),
        "confidence": round(conf, 3),
        "stance": stance,
        "key_drivers": _lst("key_drivers", 3),
        "risks": _lst("risks", 2),
        "summary": str(obj.get("summary") or "")[:200],
        "news_count": len(lines),
        "note": "",
        "elapsed": res.get("elapsed", 0.0),
        "_ts": time.time(),
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass
    return out


# ------------------------- 决策层的二次解读 -------------------------
_DEC_SYSTEM = (
    "你是 A 股策略研究员。下面是模型算出的量化决策结果（含各信号权重与贡献），"
    "以及对应的新闻政策材料。你的任务是把这些**数字**翻译成人能读懂的判断，"
    "只基于给定数据推演，不要编造新的事实，不要给出确定性承诺。"
)

_DEC_PROMPT = """标的：{name}（{code}）

【量化决策结果】综合分 {score:+.3f}（-1 空 ↔ +1 多），系统建议「{advice}」，
信号一致度 {agreement:.0%}，适用周期 5~10 个交易日。

各信号（取值/权重/贡献）：
{signals}

【拐点概率】{turn_txt}

【新闻政策材料】（{n} 条）
{news_block}

请输出 JSON（不要任何多余文字）：
{{
  "conclusion": "不超过 120 字的中文研判：先讲哪些信号在主导，再讲结论",
  "agrees": ["最多 3 条：支撑该结论的证据，每条 ≤30 字"],
  "diverges": ["最多 3 条：与该结论矛盾或需要警惕的点，每条 ≤30 字"],
  "watch": ["最多 3 条：未来 5~10 日最需要盯的具体信号/价位/事件，每条 ≤30 字"],
  "final_stance": "偏多" 或 "偏空" 或 "中性"
}}"""


def analyze_decision_llm(decision: dict, groups: dict, name: str = "", code: str = "",
                         turn: dict | None = None, override: dict | None = None) -> dict:
    """把量化决策结果交给大模型做二次解读（解释层，**不增加新的信号权重**）。

    为什么单独做一次调用：新闻分析回答的是"消息面怎么样"，
    而这里要回答"当模型说涨、但资金在流出、且 10 日内顶拐点概率高于基线时，
    到底该怎么理解这个分歧"。这种多信号交叉解读正是大模型擅长的。

    输出里 final_stance 只是模型对已有数字的复述性判断，**不进入**联合决策加权
    （否则等于让 LLM 给自己打分），这一点必须在界面上写清楚。
    """
    if not decision:
        return {"ok": False, "note": "没有决策结果可解读"}
    if not available_providers() and not (_override_target(override) or _override_target(_env_override())):
        return {"ok": False, "note": "未配置大模型密钥，跳过决策解读"}

    sig_txt = "\n".join(
        f"  - {s['name']}：取值 {s['value']:+.3f}，权重 {s['weight']:.2f}，"
        f"贡献 {s['contrib']:+.3f}（{s['note']}）"
        for s in decision.get("signals", []))
    tt = []
    for h in sorted((turn or {}).get("probs", {}) or {}):
        p, b = (turn["probs"][h] or {}), ((turn.get("base") or {}).get(h) or {})
        tt.append(f"{h}日内顶 {p.get('top',0)*100:.0f}%（基线{b.get('top',0)*100:.0f}%）、"
                  f"底 {p.get('bot',0)*100:.0f}%（基线{b.get('bot',0)*100:.0f}%）")
    for kind, cn in (("top", "顶"), ("bot", "底")):
        d = (turn or {}).get("days", {}).get(kind)
        if d:
            tt.append(f"预计下一个{cn}拐点约 {d['point']:.0f} 个交易日后（区间 {d['lo']:.0f}~{d['hi']:.0f} 日）")
    if not tt:
        tt.append("拐点模型未训练成功")

    lines = _fmt_news(groups, getattr(config, "LLM_MAX_NEWS", 30))
    prompt = _DEC_PROMPT.format(
        name=name or code, code=code, score=float(decision.get("score", 0)),
        advice=decision.get("advice", ""), agreement=float(decision.get("agreement", 0)),
        signals=sig_txt, turn_txt="；".join(tt), n=len(lines),
        news_block="\n".join(lines))
    res = chat(prompt, system=_DEC_SYSTEM, max_tokens=700, override=override)
    if not res["ok"]:
        return {"ok": False, "note": f"大模型调用失败：{res['error']}"}
    obj = _extract_json(res["text"])
    if obj is None:
        return {"ok": False, "note": "大模型返回无法解析为 JSON", "raw": res["text"][:300]}
    stance = str(obj.get("final_stance") or "")
    if stance not in ("偏多", "偏空", "中性"):
        stance = "中性"
    return {
        "ok": True,
        "provider": res["provider"],
        "model": res["model"],
        "conclusion": str(obj.get("conclusion") or "")[:200],
        "agrees": [str(x)[:40] for x in (obj.get("agrees") or [])][:3],
        "diverges": [str(x)[:40] for x in (obj.get("diverges") or [])][:3],
        "watch": [str(x)[:40] for x in (obj.get("watch") or [])][:3],
        "final_stance": stance,
        "elapsed": res.get("elapsed", 0.0),
        "note": "本段是对量化决策的解释，不是独立信号，不参与加权",
    }


# ==========================================================================
# 大模型独立价格预测 + 条件规则预判
#
# 设计红线（务必保持）：
#   这两块是**独立第二意见**，**不进入联合决策加权**。
#   原因：联合决策的权重是系统内生的（config.SIGNAL_WEIGHTS），
#   若让 LLM 的预测或规则也参与加权，就等于"让大模型给自己打分"，
#   整个体系的可解释性立刻塌掉。
#
#   为让 LLM 的预测真正**独立**（而不是迎合量化结论），
#   喂给它的材料里**刻意不含量化模型的目标价/区间/综合分**，
#   只给原始行情 + 技术指标 + 市场环境 + 它自己上一轮的新闻研判结论。
#   这样两套结论的差异才是真信息量，分歧本身值得展示。
# ==========================================================================

_FC_SYSTEM = (
    "你是一名 A 股独立策略分析师。你会基于自己看到的行情与技术指标，"
    "独立给出未来几个交易周期的价格预测，并把结论落成可执行的条件规则。"
    "只根据给定数据推演，不编造事实，不使用未来信息，不给确定性承诺。"
    "严格输出 JSON，不要输出任何解释文字或 markdown 代码块。"
)

_FC_PROMPT = """标的：{name}（{code}）

【最近 {n_days} 个交易日收盘价与成交量】
{price_block}

【当前技术指标】
{tech_block}

【市场环境】
{mkt_block}

【我此前对新闻/政策的判断】（这是我自己上一轮的结论，请沿用这个口径）
{news_block}

【系统已给出的关键价位】（这些价位是客观计算结果，供你写规则时引用）
{levels_block}

请输出 JSON（不要任何多余文字）：
{{
  "direction": "偏多" 或 "偏空" 或 "中性",
  "confidence": 0到1的小数,
  "forecasts": [
    {{"horizon": 5, "pct": -3.2, "price": 26.80, "prob_up": 0.45, "reason": "不超过30字"}},
    {{"horizon": 10, "pct": 1.5, "price": 28.00, "prob_up": 0.55, "reason": "不超过30字"}},
    {{"horizon": 20, "pct": 0.0, "price": 27.00, "prob_up": 0.50, "reason": "不超过30字"}}
  ],
  "rules": [
    {{"trigger": "收盘跌破 26.50 且量比小于 0.8", "action": "减仓或离场", "why": "不超过30字"}},
    {{"trigger": "放量站上 28.20 并连续 2 日收于其上", "action": "可加仓至计划仓位", "why": "不超过30字"}}
  ],
  "key_reasons": ["最多3条，每条不超过30字"],
  "risks": ["最多3条，每条不超过30字"],
  "summary": "不超过120字的中文总结：先讲价格预测的核心逻辑，再讲最关键的那条规则"
}}

硬性要求：
1. forecasts 必须给出 5 / 10 / 20 三个周期，horizon 用整数天数；
   pct 是相对当前收盘价的涨跌幅（百分比，可正可负），price 是对应目标价（元）。
2. prob_up 是"该周期收涨的概率"（0到1）。没有把握就给 0.5 附近，别硬给极端值。
3. rules 必须是**条件式**的：trigger 写清"什么条件成立"（价位/量能/均线/形态），
   action 写清"满足后怎么做"。至少 2 条，最多 5 条。
4. 不要输出 markdown 代码块，只输出 JSON。"""


def _fmt_price_block(kline, n_days: int = 20) -> str:
    """把最近 N 日的收盘价 + 成交量拼成紧凑文本（日期:收盘/量万手）。"""
    if kline is None or len(kline) == 0:
        return "（无行情数据）"
    try:
        df = kline.tail(n_days)
        lines = []
        for _, r in df.iterrows():
            d = str(r["date"])[:10]
            c = float(r["close"])
            v = float(r.get("volume", 0) or 0) / 1e6     # 换算成百万股，更易读
            lines.append(f"  {d}  收 {c:.2f}  量 {v:.1f}")
        return "\n".join(lines)
    except Exception:
        return "（行情格式化失败）"


def _fmt_tech_block(tech) -> str:
    """当前技术指标快照（只取最后一行）。"""
    if tech is None or len(tech) == 0:
        return "（无技术指标）"
    try:
        r = tech.iloc[-1]

        def g(col, scale=1.0, fmt="{:.4f}", suffix=""):
            if col not in tech.columns:
                return None
            v = r[col]
            if v is None or (isinstance(v, float) and v != v):   # NaN
                return None
            return fmt.format(float(v) * scale) + suffix

        items = [
            ("MA5", g("ma5", 1 / 1, "{:.2f}")),
            ("MA10", g("ma10", 1 / 1, "{:.2f}")),
            ("MA20", g("ma20", 1 / 1, "{:.2f}")),
            ("RSI(14)", g("rsi", 1, "{:.1f}")),
            ("MACD柱", g("macd_hist", 1, "{:.4f}")),
            ("布林%B", g("boll_pctb", 1, "{:.3f}")),
            ("ATR%均值", g("atr_norm", 100, "{:.2f}", "%")),
            ("20日波动率", g("vol", 100, "{:.2f}", "%")),
            ("量比", g("vol_ratio", 1, "{:.2f}")),
            ("5日动量", g("mom5", 100, "{:.2f}", "%")),
            ("20日动量", g("mom20", 100, "{:.2f}", "%")),
            ("距20日高", g("dist_high20", 100, "{:.2f}", "%")),
            ("距20日低", g("dist_low20", 100, "{:.2f}", "%")),
            ("超额(5日,相对大盘)", g("rs5", 100, "{:.2f}", "%")),
        ]
        out = [f"  - {k}：{v}" for k, v in items if v is not None]
        return "\n".join(out) if out else "（技术指标缺失）"
    except Exception:
        return "（技术指标格式化失败）"


def _fmt_market_block(tech) -> str:
    """市场环境：指数动量 + 个股相对强弱。"""
    if tech is None or len(tech) == 0:
        return "  - 指数数据不可用"
    try:
        r = tech.iloc[-1]
        out = []
        for label, col, sc in (("上证5日", "idx_mom5", 100), ("上证20日", "idx_mom20", 100),
                               ("上证当日", "idx_ret", 100)):
            if col in tech.columns:
                v = r[col]
                if v == v:      # not NaN
                    out.append(f"  - {label}：{float(v) * sc:+.2f}%")
        out.append(f"  - 个股相对大盘超额(20日)：{float(r.get('rs20', 0) or 0) * 100:+.2f}%"
                   if "rs20" in tech.columns else "  - 相对强弱不可用")
        return "\n".join(out)
    except Exception:
        return "  - 市场环境不可用"


def analyze_forecast_llm(kline, tech, news: dict | None = None, levels: dict | None = None,
                          name: str = "", code: str = "", override: dict | None = None) -> dict:
    """让大模型给出**独立**的价格预测 + 条件式决策规则。

    与 analyze_decision_llm 的分工：
      - analyze_decision_llm：解读**已有**的量化结论（"这个结论该怎么理解"）
      - analyze_forecast_llm：自己**先出**价格预测和规则（"如果我来做，我会怎么判"）

    关键：喂进来的材料**不含**量化模型的目标价与综合分，保证它是独立第二意见。
    输出**不参与**联合决策加权（见文件上方设计红线）。
    """
    if not available_providers() and not (_override_target(override) or _override_target(_env_override())):
        return {"ok": False, "note": "未配置大模型密钥，跳过价格预测"}

    # 新闻研判：沿用大模型自己上一轮的结论（保持口径一致）
    nb = []
    if news and news.get("ok"):
        nb.append(f"  - 立场：{news.get('stance', '中性')}（情绪 {float(news.get('sentiment', 0.5)):.2f}，"
                  f"政策影响 {float(news.get('policy_impact', 0)):+.2f}，置信 {float(news.get('confidence', 0)):.2f}）")
        for d in (news.get("key_drivers") or [])[:3]:
            nb.append(f"  - 利好逻辑：{d}")
        for d in (news.get("risks") or [])[:3]:
            nb.append(f"  - 风险点：{d}")
    else:
        nb.append("  - （新闻研判不可用，本次预测主要依据行情与技术面）")

    # 关键价位：给 LLM 写规则时要引用的客观锚点
    lv = []
    for k, v in (levels or {}).items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            lv.append(f"  - {k}：{v[0]} ~ {v[1]}" if len(v) >= 2 else f"  - {k}：{v}")
        else:
            lv.append(f"  - {k}：{v}")
    levels_block = "\n".join(lv) if lv else "  - （无）"

    prompt = _FC_PROMPT.format(
        name=name or code, code=code,
        n_days=min(20, len(kline) if kline is not None else 0),
        price_block=_fmt_price_block(kline),
        tech_block=_fmt_tech_block(tech),
        mkt_block=_fmt_market_block(tech),
        news_block="\n".join(nb),
        levels_block=levels_block,
    )
    res = chat(prompt, system=_FC_SYSTEM, max_tokens=1400, override=override)
    if not res["ok"]:
        return {"ok": False, "note": f"大模型调用失败：{res['error']}"}
    obj = _extract_json(res["text"])
    if obj is None:
        return {"ok": False, "note": "大模型返回无法解析为 JSON", "raw": res["text"][:300]}

    # ---- 解析预测（逐条做数值校验与夹逼，LLM 偶尔会给出离谱数字）
    last_close = 0.0
    try:
        if kline is not None and len(kline):
            last_close = float(kline["close"].iloc[-1])
    except Exception:
        pass
    fc = []
    for it in (obj.get("forecasts") or [])[:5]:
        try:
            h = int(it.get("horizon") or 0)
            if h <= 0:
                continue
            pct = float(it.get("pct"))
            pct = max(-60.0, min(60.0, pct))            # 夹逼，别让 LLM 的离谱值进报告
            price = it.get("price")
            price = float(price) if price not in (None, "") else (
                last_close * (1 + pct / 100) if last_close else None)
            if price is not None and last_close:
                price = max(last_close * 0.4, min(last_close * 1.6, price))
            pu = float(it.get("prob_up") or 0.5)
            fc.append({
                "horizon": h,
                "pct": round(pct, 2),
                "price": round(price, 2) if price is not None else None,
                "prob_up": max(0.0, min(1.0, pu)),
                "reason": str(it.get("reason") or "")[:60],
            })
        except Exception:
            continue
    fc.sort(key=lambda x: x["horizon"])

    rules = []
    for it in (obj.get("rules") or [])[:5]:
        trig = str(it.get("trigger") or "").strip()
        act = str(it.get("action") or "").strip()
        if not trig or not act:
            continue
        rules.append({"trigger": trig[:70], "action": act[:50],
                      "why": str(it.get("why") or "")[:50]})

    direction = str(obj.get("direction") or "")
    if direction not in ("偏多", "偏空", "中性"):
        direction = "中性"
    conf = float(obj.get("confidence") or 0.0)
    if conf != conf:          # NaN
        conf = 0.0

    return {
        "ok": bool(fc),
        "provider": res["provider"],
        "model": res["model"],
        "direction": direction,
        "confidence": max(0.0, min(1.0, conf)),
        "forecasts": fc,
        "rules": rules,
        "key_reasons": [str(x)[:50] for x in (obj.get("key_reasons") or [])][:3],
        "risks": [str(x)[:50] for x in (obj.get("risks") or [])][:3],
        "summary": str(obj.get("summary") or "")[:200],
        "elapsed": res.get("elapsed", 0.0),
        "note": "独立第二意见，不参与联合决策加权",
    }


def status() -> dict:
    provs = available_providers()
    return {
        "enabled": bool(getattr(config, "LLM_ENABLED", True)),
        "providers": provs,
        "model": _model_for(provs[0]) if provs else "",
        "ready": bool(provs),
        "hint": "" if provs else "未检测到大模型密钥（设置 DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY 即可启用）",
    }
