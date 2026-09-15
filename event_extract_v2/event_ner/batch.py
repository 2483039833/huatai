# -*- coding: utf-8 -*-
"""
无标签数据批量抽取（路线 C：规则预筛 + LLM + 规则兜底）
========================================================
- 断点续跑：JSONL 增量写盘 + 按 article_file_id 去重（避免重复行）
- 并行：ThreadPoolExecutor（LLM 为网络 IO 密集，多线程有效）
- 输出：outputs_event/extracted_{source}.jsonl（每行一个 JSON）

用法：
  python -m event_ner.run --source unlabeled_test --limit 1000 --workers 4
  python -m event_ner.run --source unlabeled_test --rule-only   # 纯规则，秒级
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .extractor import EventExtractor
from .paths import cleaned_dir, output_dir

SOURCE_MAP = {
    "unlabeled_test": "unlabeled_test_cleaned.xlsx",
    "unlabeled_train": "unlabeled_train_cleaned.xlsx",
}


def _load_articles(source: str, limit: Optional[int]) -> List[Dict]:
    if source not in SOURCE_MAP:
        raise ValueError(f"source 必须是 {list(SOURCE_MAP.keys())} 之一")
    path = os.path.join(cleaned_dir(), SOURCE_MAP[source])
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 {path}")
    df = pd.read_excel(path)
    if limit:
        df = df.head(limit)
    return df.to_dict("records")


def _dedup_and_load(out_path: str, keep_llm_only: bool = False) -> Tuple[Set[str], int]:
    """
    读取已有 JSONL，按 article_file_id 去重后重写，返回 (seen_ids, 有效行数)。

    keep_llm_only=True 时只保留 method=llm 的行（用于重新处理规则兜底的部分）。
    """
    seen: Set[str] = set()
    unique_lines: List[str] = []
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    aid = obj.get("article_file_id", "")
                    method = obj.get("method", "")
                except Exception:
                    continue
                if keep_llm_only and method != "llm":
                    continue  # 丢弃规则兜底行，稍后重新处理
                if aid and aid not in seen:
                    seen.add(aid)
                    unique_lines.append(line)
        # 重写为去重/过滤后的文件
        with open(out_path, "w", encoding="utf-8") as f:
            if unique_lines:
                f.write("\n".join(unique_lines) + "\n")
    return seen, len(unique_lines)


def batch_extract(
    source: str,
    limit: Optional[int] = None,
    rule_only: bool = False,
    workers: int = 4,
    resume: bool = True,
    chunk: int = 200,
    redo_rule: bool = False,
) -> str:
    """
    批量抽取无标签数据，返回输出 JSONL 路径。

    redo_rule=True 时：丢弃已有文件中 method=rule 的行（规则兜底结果），
    让这些文章重新走 LLM 抽取（用于 API 恢复后补抽）。
    """
    articles = _load_articles(source, limit)
    total = len(articles)

    out = output_dir()
    out_path = os.path.join(out, f"extracted_{source}.jsonl")
    ckpt_path = os.path.join(out, f".{source}.ckpt.json")

    # 断点恢复：去重已有文件并重建 seen 集合，杜绝重复行
    if resume:
        seen, done = _dedup_and_load(out_path, keep_llm_only=redo_rule)
    else:
        if os.path.exists(out_path):
            open(out_path, "w", encoding="utf-8").close()
        seen, done = set(), 0

    # 待处理：跳过已存在的 article_file_id
    todo: List[Dict] = []
    for a in articles:
        if str(a.get("article_file_id", "")) not in seen:
            todo.append(a)

    if not todo:
        print(f"[批量抽取] {source} 已完成（{total} 条），无需重复运行。")
        return out_path

    extractor = EventExtractor(use_exemplars=False)
    mode = "规则模式(rule_only)" if rule_only else "混合模式(预筛+LLM+规则兜底)"

    def _process(article: Dict) -> Dict:
        return extractor.extract_for_batch(article, rule_only=rule_only)

    print(
        f"[批量抽取] {source} 共 {total} 条，已完成 {done} 条，待处理 {len(todo)} 条，"
        f"workers={workers}，{mode}"
    )

    method_count = {"llm": 0, "rule": 0}

    with open(out_path, "a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i in range(0, len(todo), chunk):
                batch = todo[i:i + chunk]
                results = list(pool.map(_process, batch))
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    m = r.get("method", "rule")
                    method_count[m] = method_count.get(m, 0) + 1
                f.flush()
                done += len(results)
                with open(ckpt_path, "w", encoding="utf-8") as c:
                    json.dump({"done": done}, c, ensure_ascii=False)
                print(
                    f"  [进度] {done}/{total}（本段 LLM {method_count['llm']} / 规则 {method_count['rule']}）",
                    flush=True,
                )

    # LLM 失败告警
    fail = getattr(extractor.llm, "fail_count", 0)
    if fail:
        print(f"[警告] LLM 调用失败 {fail} 次（可能是 API 欠费/限流），失败部分已用规则兜底。")

    print(f"[批量抽取] 完成 {total} 条 -> {out_path}")
    print(f"  本段方法统计：LLM {method_count['llm']} 条 / 规则 {method_count['rule']} 条")
    return out_path
