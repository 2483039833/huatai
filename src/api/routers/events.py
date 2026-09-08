"""
事件组查询接口 — 读取 C 聚类产物（event_groups.json）返回真实数据
"""
import json
from pathlib import Path

from fastapi import APIRouter, Query

router = APIRouter()

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "results"


def _load_groups():
    path = RESULTS_DIR / "event_groups.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/")
def list_events(
    limit: int = Query(50, ge=1, le=500, description="返回条数"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """获取事件组列表（真实聚类结果）"""
    groups = _load_groups()
    return {
        "total": len(groups),
        "events": groups[offset: offset + limit],
    }


@router.get("/{group_id}")
def get_event(group_id: str):
    """获取单个事件组详情"""
    for g in _load_groups():
        if g.get("group_id") == group_id:
            return g
    return {"error": f"事件组 {group_id} 不存在"}
