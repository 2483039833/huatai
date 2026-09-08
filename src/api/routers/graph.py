"""
关系图谱接口 — 从 alerts.json 构建「公司-事件」关联图谱
节点：公司（secu_abbr）+ 事件组；边：事件 → 关联公司
"""
import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "results"


def _load_scored_groups():
    path = RESULTS_DIR / "alerts.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _build_graph(groups: list[dict]) -> dict:
    """构建公司-事件二分图"""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for g in groups:
        event_id = g["group_id"]
        nodes[event_id] = {
            "id": event_id,
            "label": (g.get("merged_title", "") or "")[:30],
            "type": "event",
            "event_name": g.get("event_name"),
            "article_count": g.get("article_count", 0),
            "alert_level": g.get("alert_level"),
            "impact_score": g.get("impact_score"),
            "cluster_confidence": g.get("cluster_confidence"),
            "representative_article_id": g.get("representative_article_id"),
        }
        for code, name in zip(g.get("trading_codes", []), g.get("secu_names", [])):
            nodes[code] = {
                "id": code,
                "label": name or code,
                "type": "company",
                "trading_code": code,
            }
            edges.append({
                "source": event_id,
                "target": code,
                "event_name": g.get("event_name"),
                "alert_level": g.get("alert_level"),
                "weight": g.get("impact_score") or g.get("article_count", 1),
            })
        event_type = g.get("event_name") or "unknown"
        type_node = f"type_{event_type}"
        nodes[type_node] = {
            "id": type_node,
            "label": event_type,
            "type": "event_type",
        }
        edges.append({
            "source": event_id,
            "target": type_node,
            "relation": "has_type",
            "weight": 1,
        })

        for event in g.get("events", []):
            source = event.get("article_source")
            if not source:
                continue
            source_node = f"source_{source}"
            nodes[source_node] = {
                "id": source_node,
                "label": source,
                "type": "source",
            }
            edges.append({
                "source": event_id,
                "target": source_node,
                "relation": "reported_by",
                "weight": 1,
            })
    return {"nodes": list(nodes.values()), "edges": edges}


@router.get("/")
def get_graph(
    only_alerted: bool = Query(True, description="仅包含有预警的事件组（默认）"),
    limit: int = Query(100, ge=1, le=500, description="事件组数量上限"),
):
    """获取事件-公司关系图谱"""
    groups = _load_scored_groups()
    if only_alerted:
        groups = [g for g in groups if g.get("alert_level") not in (None, "P3")]
    # 按报道数降序取前 limit 个事件组
    groups.sort(key=lambda g: g.get("article_count", 0), reverse=True)
    graph = _build_graph(groups[:limit])
    return {
        "total_events": len(groups),
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
        **graph,
    }


@router.get("/company/{trading_code}")
def get_company_graph(trading_code: str):
    """获取某公司的关联事件图谱"""
    groups = [g for g in _load_scored_groups() if trading_code in (g.get("trading_codes") or [])]
    graph = _build_graph(groups)
    return {
        "trading_code": trading_code,
        "total_events": len(groups),
        **graph,
    }

