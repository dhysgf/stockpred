"""检查 docs/ 静态站的站内链接是否都有对应文件（防止再出现 /report 这类 404）。

用法：
    python tools/check_docs_links.py

判定规则（确定性，无网络请求）：
- 只查 href / src 属性；
- 跳过外链（http://、https://、//）、锚点（#）、协议类（mailto:、data:、javascript:）；
- 相对链接按「所在文件目录」解析，去掉 ?query 与 #hash 后要求文件存在；
- 以 / 开头的站内绝对路径一律判为缺陷——GitHub Pages 项目站的根是 /<repo>/，
  绝对路径会绕过仓库前缀，必然 404。
退出码：0 = 全部通过；1 = 存在缺陷。
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse, unquote

ATTRIBUTES = ("href", "src")
SKIP_PREFIX = ("http://", "https://", "//", "#", "mailto:", "data:", "javascript:")


def iter_links(html: str):
    """极简属性抽取：够用且不引第三方依赖，HTML 由本项目自己生成，结构可控。"""
    for attr in ATTRIBUTES:
        marker = attr + '="'
        start = 0
        while True:
            i = html.find(marker, start)
            if i < 0:
                break
            j = html.find('"', i + len(marker))
            if j < 0:
                break
            yield attr, html[i + len(marker):j]
            start = j + 1


def check(docs_dir: Path):
    problems: list[tuple[str, str, str]] = []
    checked = 0
    for page in sorted(docs_dir.rglob("*.html")):
        html = page.read_text(encoding="utf-8", errors="replace")
        for attr, raw in iter_links(html):
            link = raw.strip()
            if not link or link.startswith(SKIP_PREFIX):
                continue
            if "${" in link or link.startswith("{{"):
                continue  # JS 模板占位符，真实路径运行时才拼出
            checked += 1
            rel = page.relative_to(docs_dir).as_posix()
            if link.startswith("/"):
                problems.append((rel, attr, link + "  ← 站内绝对路径，会绕过 /<repo>/ 前缀"))
                continue
            path = unquote(urlparse(link).path)
            if not path:
                continue
            target = (page.parent / path).resolve()
            if path.endswith("/") or target.is_dir():
                target = target / "index.html"
            if not target.exists():
                problems.append((rel, attr, link + "  ← 目标文件不存在"))
    return checked, problems


def main() -> int:
    docs = Path(__file__).resolve().parent.parent / "docs"
    checked, problems = check(docs)
    print(f"检查 {docs}")
    print(f"站内链接 {checked} 条，缺陷 {len(problems)} 条")
    for page, attr, msg in problems:
        print(f"  [FAIL] {page}  {attr}=\"{msg}\"")
    print("=> 全部通过" if not problems else "=> 存在缺陷，需修复")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
