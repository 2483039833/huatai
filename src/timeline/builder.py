"""
时间线构建：按事件组 + 时间轴聚合
"""
from collections import defaultdict
import pandas as pd


class TimelineBuilder:
    """事件时间线构建器"""

    @staticmethod
    def build(events: list[dict]) -> list[dict]:
        """
        从合并后的事件组列表构建时间线

        Args:
            events: merge_all_groups() 的输出

        Returns:
            时间线列表，每条包含时间、事件、详情
        """
        timeline = []
        for event in events:
            timeline.append({
                "time": event.get("first_publish_time", ""),
                "title": event.get("merged_title", ""),
                "event_name": event.get("event_name", ""),
                "article_count": event.get("article_count", 0),
                "sources": event.get("sources", []),
                "group_id": event.get("group_id"),
                "trading_codes": event.get("trading_codes", []),
            })

        timeline.sort(key=lambda x: x["time"])
        return timeline

    @staticmethod
    def build_by_company(events: list[dict]) -> dict[str, list[dict]]:
        """按公司分组构建时间线"""
        company_events = defaultdict(list)
        for event in events:
            for code in event.get("trading_codes", []):
                company_events[code].append(event)
        return dict(company_events)

    @staticmethod
    def build_by_event_type(events: list[dict]) -> dict[str, list[dict]]:
        """按事件类型分组构建时间线"""
        type_events = defaultdict(list)
        for event in events:
            type_events[event.get("event_name", "unknown")].append(event)
        return dict(type_events)
