#!/usr/bin/env python3
"""
AI 工具新闻流水线：采集 → 合并去重过滤 → Gemini 生成每日简报
用法: python run_digest.py [--date YYYY-MM-DD] [--config path/to/ai-tools.yml] [--repo-root .]
"""
import argparse
import html
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import feedparser
import requests
import yaml

# 关键词白名单（命中则保留）
KEYWORDS_WHITELIST = [
    "agent", "ai agent", "copilot", "coding", "ide", "workflow", "mcp",
    "rag", "llm", "inference", "eval", "fine-tune", "finetune",
    "prompt", "tool", "desktop", "browser", "voice", "meeting", "ppt", "design",
    "openai", "anthropic", "gemini", "claude", "cursor", "release", "launch",
]
# 黑名单（命中则丢弃）
KEYWORDS_BLACKLIST = ["job", "hiring", "careers", "招聘", "opinion", "软文"]


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def strip_html(desc: str) -> str:
    if not desc:
        return ""
    # 简单去标签
    text = re.sub(r"<[^>]+>", " ", desc)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()[:500]


def fetch_feed(url: str, source_name: str) -> list[dict]:
    """抓取单个 feed，返回条目列表。"""
    entries = []
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "AI-Tools-Digest/1.0"})
        resp.raise_for_status()
        fp = feedparser.parse(resp.content)
        for e in fp.entries:
            link = e.get("link") or ""
            if not link:
                continue
            title = (e.get("title") or "").strip()
            desc = e.get("summary") or e.get("description") or ""
            published = e.get("published") or e.get("updated") or ""
            entries.append({
                "title": title,
                "link": link,
                "description": strip_html(desc),
                "published": published,
                "source": source_name,
            })
    except Exception as err:
        print(f"[WARN] fetch failed {source_name} {url}: {err}", file=sys.stderr)
    return entries


def merge_and_filter(all_entries: list[dict]) -> list[dict]:
    """按 link 去重，再按关键词白名单/黑名单过滤。"""
    seen = set()
    unique = []
    for e in all_entries:
        link = (e.get("link") or "").strip()
        if not link or link in seen:
            continue
        seen.add(link)
        title_lower = (e.get("title") or "").lower()
        desc_lower = (e.get("description") or "").lower()
        combined = f"{title_lower} {desc_lower}"
        if any(b in combined for b in KEYWORDS_BLACKLIST):
            continue
        if any(w in combined for w in KEYWORDS_WHITELIST) or "release" in combined or "blog" in e.get("source", "").lower():
            unique.append(e)
    return unique


def build_context(entries: list[dict], max_items: int = 80) -> str:
    """生成交给 Gemini 的 context 文本。"""
    lines = []
    for i, e in enumerate(entries[:max_items], 1):
        lines.append(f"[{i}] 来源: {e.get('source', '')}\n标题: {e.get('title', '')}\n链接: {e.get('link', '')}\n摘要: {e.get('description', '')[:300]}\n")
    return "\n".join(lines)


def call_gemini(context: str, date_str: str, api_key: str) -> str:
    """调用 Gemini 生成每日简报 Markdown。"""
    prompt = f"""你是一位 AI 工具与开发生态编辑。根据下面抓取到的「AI 工具」相关资讯（标题、链接、摘要），生成一份**今日简报**（日期：{date_str}），要求：

1. **Top 5 必看**：选 5 条最重要的，每条一句话说明「发生了什么」+ 一句话「为什么重要/对读者有什么用」+ 链接。
2. **按分类要点**：其余条目按类别归纳（编程工具 / 办公与多模态 / Agent 与框架 / 模型与推理 / 开源项目 / 行业动态），每类 3–8 条简短要点，保留链接。
3. **可立即尝试**：列出 1–3 条「今天值得试的动作」（如：升级某工具、试用某功能）。
4. 全部使用中文，短句、可执行，链接必须保留。若某类无内容可写「无」或省略。

=== 原始资讯（供你筛选与总结）===
{context}
=== 结束 ===

请直接输出 Markdown，不要输出「好的，以下是…」等前缀。"""

    # v1beta 的 gemini-1.5-flash 已下线，改用 v1 + gemini-1.5-flash-latest
    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash:generateContent?key={api_key}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.4,
            "maxOutputTokens": 8192,
        },
    }
    resp = requests.post(url, json=body, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
    if not text:
        raise RuntimeError("Gemini returned empty content")
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.utcnow().strftime("%Y-%m-%d"), help="日期 YYYY-MM-DD")
    parser.add_argument("--config", default=".github/feeds/ai-tools.yml", help="feed 配置文件路径")
    parser.add_argument("--repo-root", default=".", help="仓库根目录")
    parser.add_argument("--skip-gemini", action="store_true", help="只采集与合并，不调 Gemini（调试用）")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    config_path = repo / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(config_path)
    sources = cfg.get("sources", [])
    date_str = args.date

    raw_dir = repo / "news" / "ai-tools" / "raw" / date_str
    digest_dir = repo / "news" / "ai-tools" / "digest"
    raw_dir.mkdir(parents=True, exist_ok=True)
    digest_dir.mkdir(parents=True, exist_ok=True)

    all_entries = []
    for s in sources:
        name = s.get("name", "unknown")
        url = s.get("url", "").strip()
        if not url:
            continue
        entries = fetch_feed(url, name)
        all_entries.extend(entries)
        # 按源保存 raw（文件名安全化）
        safe_name = re.sub(r"[^\w\-]", "-", name).strip("-") or "source"
        raw_file = raw_dir / f"{safe_name}.json"
        try:
            with open(raw_file, "w", encoding="utf-8") as f:
                json.dump(entries, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] write raw {raw_file}: {e}", file=sys.stderr)

    merged = merge_and_filter(all_entries)
    context = build_context(merged)
    context_path = raw_dir / "context.txt"
    context_path.write_text(context, encoding="utf-8")

    if args.skip_gemini:
        print(f"Collected {len(all_entries)} entries, merged {len(merged)}. context.txt written. Skipping Gemini.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not set, skipping digest generation. Raw + context saved.", file=sys.stderr)
        sys.exit(0)

    md = call_gemini(context, date_str, api_key)
    digest_path = digest_dir / f"{date_str}.md"
    digest_path.write_text(md, encoding="utf-8")
    print(f"Digest written: {digest_path}")


if __name__ == "__main__":
    main()
