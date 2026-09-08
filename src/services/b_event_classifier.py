"""Member-B event classifier adapted to the main system contract.

The classifier ports B's production event recognition idea without consuming
B's generated cluster files: taxonomy-constrained character NB, issuer-local
title nearest neighbor, and optional B Word2Vec semantic retrieval for company
events.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.api.schemas.cluster import EventGroupSchema
from src.services.cleaned_xlsx import iter_cleaned_rows


ROOT = Path(__file__).resolve().parents[2]
RETRAINED_WORD2VEC_PATH = ROOT / "data" / "models" / "b_company_title_char_w2v.model"
LEGACY_B_WORD2VEC_PATH = ROOT / "financial_event_radar_share(1)" / "financial_event_radar_share" / "models" / "company_title_char_w2v.model"
B_WORD2VEC_PATH = RETRAINED_WORD2VEC_PATH if RETRAINED_WORD2VEC_PATH.exists() else LEGACY_B_WORD2VEC_PATH
RAW_TRAINING_FILE = ROOT / "news_with_tags_train.xlsx"
CLEANED_COMPANY_TRAINING_FILE = ROOT / "cleaned_data" / "train_tagged_cleaned.xlsx"
DIRECTION_MAP = {
    "正面": "正面",
    "利好": "正面",
    "positive": "正面",
    "负面": "负面",
    "利空": "负面",
    "negative": "负面",
    "中性": "中性",
    "neutral": "中性",
}
POSITIVE_WORDS = ("增长", "突破", "中标", "签约", "获批", "盈利", "增持", "创新", "合作", "投产", "交付", "上调")
NEGATIVE_WORDS = ("下滑", "亏损", "减持", "处罚", "调查", "违约", "风险", "终止", "召回", "裁员", "下调", "澄清")


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _model_text(record: dict[str, Any]) -> str:
    title = record.get("title") or ""
    content = record.get("content") or ""
    return f"{title} {title} {content[:1200]}"


def _char_ngrams(text: str) -> Counter[str]:
    compact = re.sub(r"\s+", "", text or "")
    features: Counter[str] = Counter()
    for size in (2, 3):
        features.update(compact[index : index + size] for index in range(len(compact) - size + 1))
    return features


def _title_chars(title: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", title or "")


def _direction_from_keywords(text: str) -> str:
    positive = sum(text.count(word) for word in POSITIVE_WORDS)
    negative = sum(text.count(word) for word in NEGATIVE_WORDS)
    if positive > negative:
        return "正面"
    if negative > positive:
        return "负面"
    return "中性"


class TaxonomyNaiveBayes:
    """Small auditable multinomial NB over Chinese character n-grams."""

    def __init__(self) -> None:
        self.class_documents: Counter[str] = Counter()
        self.class_tokens: dict[str, Counter[str]] = defaultdict(Counter)
        self.class_totals: Counter[str] = Counter()
        self.vocabulary: set[str] = set()
        self.document_count = 0

    def fit(self, records: Iterable[dict[str, Any]]) -> None:
        raw_counts: Counter[str] = Counter()
        cached: list[tuple[str, Counter[str]]] = []
        for record in records:
            label = record.get("event_name")
            if not label:
                continue
            features = _char_ngrams(_model_text(record))
            cached.append((label, features))
            self.class_documents[label] += 1
            self.document_count += 1
            raw_counts.update(features)
        self.vocabulary = {feature for feature, count in raw_counts.items() if count >= 2}
        for label, features in cached:
            filtered = Counter({key: count for key, count in features.items() if key in self.vocabulary})
            self.class_tokens[label].update(filtered)
            self.class_totals[label] += sum(filtered.values())

    @property
    def labels(self) -> list[str]:
        return sorted(self.class_documents)

    def score(self, text: str, candidates: list[str]) -> dict[str, float]:
        candidates = [label for label in candidates if label in self.class_documents]
        if not candidates:
            return {}
        features = Counter({key: value for key, value in _char_ngrams(text).items() if key in self.vocabulary})
        vocabulary_size = max(len(self.vocabulary), 1)
        scores: dict[str, float] = {}
        for label in candidates:
            prior = (self.class_documents[label] + 1) / (self.document_count + len(self.labels))
            denominator = self.class_totals[label] + vocabulary_size
            score = math.log(prior)
            for feature, count in features.items():
                score += count * math.log((self.class_tokens[label][feature] + 1) / denominator)
            scores[label] = score
        return scores


class TitleNearestNeighbor:
    """Issuer-local nearest labelled title using B's character overlap signal."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.by_entity: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
        self.global_samples: list[tuple[str, set[str]]] = []
        for record in records:
            sample = (record["event_name"], set(_char_ngrams(record.get("title") or "")))
            self.global_samples.append(sample)
            code = record.get("security_code") or ""
            if code:
                self.by_entity[code].append(sample)

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        target = set(_char_ngrams(record.get("title") or ""))
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        scores = {label: 0.0 for label in candidates}
        for label, features in samples:
            if label in scores and target and features:
                scores[label] = max(scores[label], len(target & features) / min(len(target), len(features)))
        return scores


class SemanticTitleNearestNeighbor:
    """Issuer-local maximum cosine retrieval over B's Word2Vec title vectors."""

    def __init__(self, records: list[dict[str, Any]], model: Any) -> None:
        self.model = model
        by_entity: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        global_samples: list[tuple[str, np.ndarray]] = []
        for record in records:
            sample = (record["event_name"], self._sentence_vector(record.get("title") or ""))
            global_samples.append(sample)
            code = record.get("security_code") or ""
            if code:
                by_entity[code].append(sample)
        self.global_samples = self._pack(global_samples)
        self.by_entity = {code: self._pack(samples) for code, samples in by_entity.items()}

    def score(self, record: dict[str, Any], candidates: list[str]) -> dict[str, float]:
        target = self._sentence_vector(record.get("title") or "")
        samples = self.by_entity.get(record.get("security_code") or "", self.global_samples)
        scores = {label: 0.0 for label in candidates}
        labels, vectors = samples
        if not np.any(target) or not len(labels):
            return scores
        similarities = vectors @ target
        for label, similarity in zip(labels, similarities, strict=True):
            if label in scores:
                scores[label] = max(scores[label], float(similarity))
        return scores

    def _pack(self, samples: list[tuple[str, np.ndarray]]) -> tuple[list[str], np.ndarray]:
        if not samples:
            return [], np.empty((0, self.model.vector_size), dtype=np.float32)
        return [label for label, _ in samples], np.vstack([vector for _, vector in samples])

    def _sentence_vector(self, title: str) -> np.ndarray:
        tokens = [token for token in _title_chars(title) if token in self.model.wv]
        if not tokens:
            return np.zeros(self.model.vector_size, dtype=np.float32)
        vector = np.mean(self.model.wv[tokens], axis=0)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


@dataclass(frozen=True)
class BClassification:
    event_name: str
    impact_direction: str
    confidence: float
    scope: str
    classifier: str
    margin: float
    review_required: bool


class BEventClassifier:
    """B's event classification model exposed for our EventGroupSchema."""

    def __init__(
        self,
        company_records: list[dict[str, Any]],
        industry_records: list[dict[str, Any]],
        semantic_model: Any | None = None,
        semantic_error: str | None = None,
    ) -> None:
        self.company_model = TaxonomyNaiveBayes()
        self.company_model.fit(company_records)
        self.industry_model = TaxonomyNaiveBayes()
        self.industry_model.fit(industry_records)
        self.direction_model = TaxonomyNaiveBayes()
        direction_training = []
        for record in company_records + industry_records:
            direction = DIRECTION_MAP.get(record.get("event_emotion") or "")
            if direction:
                labelled = dict(record)
                labelled["event_name"] = direction
                direction_training.append(labelled)
        self.direction_model.fit(direction_training)
        self.company_title_nn = TitleNearestNeighbor(company_records)
        self.company_semantic_nn = SemanticTitleNearestNeighbor(company_records, semantic_model) if semantic_model else None
        self.semantic_error = semantic_error
        self.taxonomy = {
            "company": self._taxonomy(company_records, "security_code"),
            "industry": self._taxonomy(industry_records, "industry_code"),
        }

    @classmethod
    def from_default_assets(cls) -> "BEventClassifier":
        company, industry = _load_training_records()
        semantic_model = None
        semantic_error = None
        if B_WORD2VEC_PATH.exists():
            try:
                from gensim.models import Word2Vec

                semantic_model = Word2Vec.load(str(B_WORD2VEC_PATH))
            except Exception as exc:  # pragma: no cover - depends on local env
                semantic_error = f"{type(exc).__name__}: {exc}"
        else:
            semantic_error = f"missing model: {B_WORD2VEC_PATH}"
        return cls(company, industry, semantic_model=semantic_model, semantic_error=semantic_error)

    def classify_group(self, group: EventGroupSchema) -> BClassification | None:
        if not group.events:
            return None
        record = self._record_from_group(group)
        scope = "company" if record.get("security_code") else "industry"
        model = self.company_model if scope == "company" else self.industry_model
        candidates = self._allowed_labels(scope, record, model.labels)
        if not candidates:
            return None
        if scope == "company":
            char_scores = self.company_title_nn.score(record, candidates)
            if self.company_semantic_nn:
                semantic_scores = self.company_semantic_nn.score(record, candidates)
                scores = {label: 0.1 * char_scores[label] + 0.9 * semantic_scores[label] for label in candidates}
                classifier = "b_issuer_title_semantic_hybrid"
            else:
                scores = char_scores
                classifier = "b_issuer_title_character_nn"
        else:
            scores = model.score(_model_text(record), candidates)
            classifier = "b_body_character_nb"
        ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranking:
            return None
        event_name, top_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else top_score - 2.0
        margin = float(top_score - second_score)
        return BClassification(
            event_name=event_name,
            impact_direction=self._direction(record),
            confidence=self._confidence(scope, margin, classifier),
            scope=scope,
            classifier=classifier,
            margin=round(margin, 4),
            review_required=margin < (0.01 if classifier == "b_issuer_title_semantic_hybrid" else 0.03),
        )

    def apply_to_group(self, group: EventGroupSchema) -> EventGroupSchema:
        classification = self.classify_group(group)
        if classification is None:
            return group
        return self.apply_classification(group, classification)

    @staticmethod
    def apply_classification(group: EventGroupSchema, classification: BClassification) -> EventGroupSchema:
        updated = group.model_copy(deep=True)
        updated.event_name = classification.event_name
        for event in updated.events:
            event.event_name = classification.event_name
            event.event_impact_direction = classification.impact_direction
        return updated

    @staticmethod
    def _taxonomy(records: list[dict[str, Any]], code_field: str) -> dict[str, set[str]]:
        values: dict[str, set[str]] = defaultdict(set)
        for record in records:
            code = record.get(code_field) or ""
            event_name = record.get("event_name") or ""
            if code and event_name:
                values[code].add(event_name)
        return values

    def _allowed_labels(self, scope: str, record: dict[str, Any], labels: list[str]) -> list[str]:
        code_field = "security_code" if scope == "company" else "industry_code"
        constrained = sorted(self.taxonomy[scope].get(record.get(code_field) or "", set()) & set(labels))
        return constrained or labels

    def _direction(self, record: dict[str, Any]) -> str:
        keyword_direction = _direction_from_keywords(record.get("text") or "")
        scores = self.direction_model.score(_model_text(record), self.direction_model.labels)
        if not scores:
            return keyword_direction
        return max(scores.items(), key=lambda item: item[1])[0] or keyword_direction

    @staticmethod
    def _confidence(scope: str, margin: float, classifier: str) -> float:
        if classifier == "b_issuer_title_semantic_hybrid":
            if margin >= 0.08:
                return 0.95
            if margin >= 0.05:
                return 0.94
            if margin >= 0.03:
                return 0.92
            if margin >= 0.02:
                return 0.90
            if margin >= 0.01:
                return 0.88
            if margin >= 0.005:
                return 0.80
            return 0.60
        if scope == "company":
            return 0.95 if margin >= 0.15 else 0.90 if margin >= 0.10 else 0.82 if margin >= 0.05 else 0.55
        return round(max(0.35, min(0.98, 0.55 + 0.12 * margin)), 3)

    @staticmethod
    def _record_from_group(group: EventGroupSchema) -> dict[str, Any]:
        representative = next(
            (event for event in group.events if event.article_file_id == group.representative_article_id),
            group.events[0],
        )
        text = f"{group.merged_title} {group.merged_content} {representative.article_title} {representative.content}"
        return {
            "article_id": representative.article_file_id,
            "dataset": "main_system_group",
            "title": group.merged_title or representative.article_title,
            "content": group.merged_content or representative.content,
            "text": text,
            "published_at": representative.article_publish_time.isoformat() if representative.article_publish_time else "",
            "security_code": (group.trading_codes[0] if group.trading_codes else representative.trading_code) or "",
            "security_name": (group.secu_names[0] if group.secu_names else representative.secu_abbr) or "",
            "industry_code": representative.industry_code or "",
            "industry_name": representative.industry_name or "",
        }


@lru_cache(maxsize=1)
def get_default_b_event_classifier() -> BEventClassifier:
    return BEventClassifier.from_default_assets()


@lru_cache(maxsize=1)
def _load_training_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # A 的清洗文件当前只包含原工作簿的公司事件页。公司模型优先使用
    # 去重后的清洗训练集；行业模型仍从原始训练工作簿的行业页读取。
    # 两者都严格排除官方测试集，保证后续评测是独立留出评测。
    if CLEANED_COMPANY_TRAINING_FILE.exists():
        company = _records_from_cleaned_excel(CLEANED_COMPANY_TRAINING_FILE, "company")
    else:
        company = _records_from_excel(RAW_TRAINING_FILE, "个股新闻", "company")
    industry = _records_from_excel(RAW_TRAINING_FILE, "行业新闻", "industry")
    if not company and not industry:
        raise FileNotFoundError("No labelled training files found for B event classifier")
    return company, industry


def _records_from_excel(path: Path, sheet: str | int, scope: str) -> list[dict[str, Any]]:
    try:
        frame = pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        return []
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        event_name = _clean(row.get("event_name"))
        if not event_name:
            continue
        title = _clean(row.get("article_title"))
        content = _clean(row.get("content"))
        records.append(
            {
                "article_id": _clean(row.get("article_file_id")),
                "dataset": path.name,
                "title": title,
                "content": content,
                "text": f"{title} {content}",
                "published_at": _clean(row.get("article_publish_time")),
                "security_code": _clean(row.get("trading_code")) if scope == "company" else "",
                "security_name": _clean(row.get("secu_abbr")) if scope == "company" else "",
                "industry_code": _clean(row.get("industry_code")),
                "industry_name": _clean(row.get("industry_name")),
                "event_name": event_name,
                "event_emotion": _clean(row.get("event_emotion")),
            }
        )
    return records


def _records_from_cleaned_excel(path: Path, scope: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in iter_cleaned_rows(path):
        event_name = _clean(row.get("event_name"))
        if not event_name:
            continue
        title = _clean(row.get("article_title"))
        content = _clean(row.get("content"))
        records.append(
            {
                "article_id": _clean(row.get("article_file_id")),
                "dataset": path.name,
                "title": title,
                "content": content,
                "text": f"{title} {content}",
                "published_at": _clean(row.get("article_publish_time")),
                "security_code": _clean(row.get("trading_code")) if scope == "company" else "",
                "security_name": _clean(row.get("secu_abbr")) if scope == "company" else "",
                "industry_code": _clean(row.get("industry_code")),
                "industry_name": _clean(row.get("industry_name")),
                "event_name": event_name,
                "event_emotion": _clean(row.get("event_emotion")),
            }
        )
    return records
