"""Evaluate a transparent same-event news aggregation baseline.

The pipeline uses character-shingle retrieval to avoid comparing every pair of
articles, calibrates a similarity threshold on the labelled duplicate set, and
turns accepted pairs into event clusters with union-find.
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "event_radar" / "data" / "processed"
DERIVED_DIR = ROOT / "event_radar" / "data" / "derived"
REPORT_DIR = ROOT / "event_radar" / "reports"
TRAIN_FILES = ("duplicates_train_company.jsonl.gz", "duplicates_train_industry.jsonl.gz")
TEST_FILES = ("duplicates_test_company.jsonl.gz", "duplicates_test_industry.jsonl.gz")
COMPACT = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def shingles(text: str, length: int, limit: int) -> set[str]:
    compact = COMPACT.sub("", text)[:limit]
    return {compact[index : index + length] for index in range(max(0, len(compact) - length + 1))}


@dataclass
class Article:
    article_id: str
    duplication_id: str
    title: str
    content: str
    published_at: str | None
    source: str | None
    title_features: set[str]
    body_features: set[str]


def make_articles(filenames: tuple[str, ...]) -> list[Article]:
    articles: list[Article] = []
    for filename in filenames:
        for row in iter_jsonl(DATA_DIR / filename):
            title = row.get("title") or ""
            content = row.get("content") or ""
            articles.append(
                Article(
                    article_id=row["article_id"],
                    duplication_id=row["duplication_id"],
                    title=title,
                    content=content,
                    published_at=row.get("published_at"),
                    source=row.get("source"),
                    title_features=shingles(title, 2, 240),
                    body_features=shingles(f"{title} {content}", 3, 900),
                )
            )
    return articles


def candidate_pairs(articles: list[Article]) -> set[tuple[int, int]]:
    """Retrieve likely duplicates using rare shared body shingles."""
    index: dict[str, list[int]] = defaultdict(list)
    for article_index, article in enumerate(articles):
        for feature in article.body_features:
            index[feature].append(article_index)

    overlap: Counter[tuple[int, int]] = Counter()
    for indices in index.values():
        # Common generic phrases create many irrelevant pairs and carry little signal.
        if len(indices) > 35:
            continue
        for left, right in combinations(indices, 2):
            overlap[(left, right)] += 1
    return {pair for pair, count in overlap.items() if count >= 2}


def containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def similarity(left: Article, right: Article) -> tuple[float, dict[str, float]]:
    title = containment(left.title_features, right.title_features)
    body = containment(left.body_features, right.body_features)
    # A near-identical title is a strong signal, while body overlap captures rewrites.
    score = 0.62 * title + 0.38 * body
    return score, {"title_containment": round(title, 4), "body_containment": round(body, 4)}


def positive_pairs(articles: list[Article]) -> set[tuple[int, int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, article in enumerate(articles):
        groups[article.duplication_id].append(index)
    return {pair for group in groups.values() for pair in combinations(group, 2)}


def choose_threshold(scored_pairs: list[tuple[tuple[int, int], float]], positives: set[tuple[int, int]]) -> tuple[float, dict[str, float]]:
    """Choose a recall-first threshold subject to a conservative precision floor."""
    precision_floor = 0.88
    best_threshold = 1.01
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    for threshold in [index / 100 for index in range(15, 101, 2)]:
        predicted = {pair for pair, score in scored_pairs if score >= threshold}
        metrics = pair_metrics(predicted, positives)
        if metrics["precision"] >= precision_floor and (
            metrics["recall"] > best_metrics["recall"]
            or (metrics["recall"] == best_metrics["recall"] and metrics["f1"] > best_metrics["f1"])
        ):
            best_threshold, best_metrics = threshold, metrics
    return best_threshold, best_metrics


def pair_metrics(predicted: set[tuple[int, int]], actual: set[tuple[int, int]]) -> dict[str, float]:
    true_positive = len(predicted & actual)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "true_positive": true_positive}


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def cluster_articles(size: int, accepted_pairs: set[tuple[int, int]]) -> list[list[int]]:
    union_find = UnionFind(size)
    for left, right in accepted_pairs:
        union_find.union(left, right)
    clusters: dict[int, list[int]] = defaultdict(list)
    for index in range(size):
        clusters[union_find.find(index)].append(index)
    return list(clusters.values())


def b_cubed(articles: list[Article], clusters: list[list[int]]) -> dict[str, float]:
    true_groups: dict[str, set[int]] = defaultdict(set)
    predicted_groups: dict[int, set[int]] = {}
    for index, article in enumerate(articles):
        true_groups[article.duplication_id].add(index)
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            predicted_groups[index] = set(cluster)
    precision = recall = 0.0
    for index, article in enumerate(articles):
        overlap = len(predicted_groups[index] & true_groups[article.duplication_id])
        precision += overlap / len(predicted_groups[index])
        recall += overlap / len(true_groups[article.duplication_id])
    precision /= len(articles)
    recall /= len(articles)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_articles = make_articles(TRAIN_FILES)
    train_candidates = candidate_pairs(train_articles)
    train_positives = positive_pairs(train_articles)
    train_scored = [(pair, similarity(train_articles[pair[0]], train_articles[pair[1]])[0]) for pair in train_candidates]
    threshold, train_metrics = choose_threshold(train_scored, train_positives)

    test_articles = make_articles(TEST_FILES)
    test_candidates = candidate_pairs(test_articles)
    test_positives = positive_pairs(test_articles)
    test_scored = [(pair, *similarity(test_articles[pair[0]], test_articles[pair[1]])) for pair in test_candidates]
    accepted_pairs = {pair for pair, score, _ in test_scored if score >= threshold}
    test_metrics = pair_metrics(accepted_pairs, test_positives)
    clusters = cluster_articles(len(test_articles), accepted_pairs)
    cluster_metrics = b_cubed(test_articles, clusters)

    edge_path = DERIVED_DIR / "deduplication_edges_test.jsonl.gz"
    with gzip.open(edge_path, "wt", encoding="utf-8") as target:
        for pair, score, components in sorted(test_scored, key=lambda item: item[1], reverse=True):
            left, right = (test_articles[index] for index in pair)
            target.write(json.dumps({
                "article_id_left": left.article_id,
                "article_id_right": right.article_id,
                "similarity": round(score, 4),
                "components": components,
                "same_event": score >= threshold,
                "threshold": threshold,
                "reference_same_event": left.duplication_id == right.duplication_id,
            }, ensure_ascii=False) + "\n")

    cluster_path = DERIVED_DIR / "deduplication_clusters_test.jsonl.gz"
    with gzip.open(cluster_path, "wt", encoding="utf-8") as target:
        for cluster_index, member_indices in enumerate(sorted(clusters, key=lambda value: (-len(value), value))):
            members = [test_articles[index] for index in member_indices]
            canonical = max(members, key=lambda item: (len(item.content), len(item.title)))
            target.write(json.dumps({
                "cluster_id": f"EVENT-{cluster_index + 1:04d}",
                "canonical_article_id": canonical.article_id,
                "canonical_title": canonical.title,
                "article_ids": [item.article_id for item in members],
                "article_count": len(members),
                "sources": sorted({item.source for item in members if item.source}),
                "first_published_at": min((item.published_at for item in members if item.published_at), default=None),
                "reference_duplication_ids": sorted({item.duplication_id for item in members}),
            }, ensure_ascii=False) + "\n")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "method": "rare character-shingle candidate retrieval + title/body containment similarity + recall-first threshold calibration (training precision floor 88%) + union-find clustering",
        "threshold": threshold,
        "training": {
            "records": len(train_articles),
            "positive_pairs": len(train_positives),
            "candidate_pairs": len(train_candidates),
            "candidate_recall": round(len(train_candidates & train_positives) / len(train_positives), 4),
            "pair_metrics_at_selected_threshold": {key: round(value, 4) if isinstance(value, float) else value for key, value in train_metrics.items()},
        },
        "test": {
            "records": len(test_articles),
            "positive_pairs": len(test_positives),
            "candidate_pairs": len(test_candidates),
            "candidate_recall": round(len(test_candidates & test_positives) / len(test_positives), 4),
            "pair_metrics": {key: round(value, 4) if isinstance(value, float) else value for key, value in test_metrics.items()},
            "b_cubed": {key: round(value, 4) for key, value in cluster_metrics.items()},
            "predicted_clusters": len(clusters),
            "multi_article_clusters": sum(len(cluster) > 1 for cluster in clusters),
            "edge_file": edge_path.name,
            "cluster_file": cluster_path.name,
        },
    }
    (REPORT_DIR / "deduplication_baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# 同源事件聚合基线报告", "",
        "方法：稀有字符 shingle 召回候选对，以标题和正文包含度计算相似度；在训练集保持至少 88% 对级 Precision 的前提下优先提高 Recall，再通过并查集生成事件簇。", "",
        f"- 选定相似度阈值：`{threshold:.2f}`",
        f"- 测试候选召回率：`{report['test']['candidate_recall']:.2%}`",
        f"- 测试对级 Precision / Recall / F1：`{test_metrics['precision']:.2%}` / `{test_metrics['recall']:.2%}` / `{test_metrics['f1']:.2%}`",
        f"- 测试簇级 B³ Precision / Recall / F1：`{cluster_metrics['precision']:.2%}` / `{cluster_metrics['recall']:.2%}` / `{cluster_metrics['f1']:.2%}`",
        f"- 输出事件簇：{len(clusters)} 个，其中多文章簇 {sum(len(cluster) > 1 for cluster in clusters)} 个。",
        "",
        "后续将事件类型、主体、时间与可信度融入重排序模型，可降低同主体不同事项的误聚合，并提升改写报道的召回。",
    ]
    (REPORT_DIR / "deduplication_baseline_report.md").write_text("\n".join(markdown), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
