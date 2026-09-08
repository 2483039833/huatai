"""
同源事件合并策略
将聚类到同一组的事件合并为一条"事件摘要"
"""
from datetime import datetime
from collections import Counter
import pandas as pd


class EventMerger:
    """事件合并器"""

    @staticmethod
    def merge_group(articles: pd.DataFrame, group_id: int) -> dict:
        """
        将一个聚类组的文章合并为一条结构化事件

        Args:
            articles: 组内文章 DataFrame
            group_id: 聚类组 ID

        Returns:
            dict: 合并后的事件
        """
        if articles.empty:
            return {"group_id": group_id, "article_count": 0}

        # 按时间排序
        articles_sorted = articles.sort_values("article_publish_time")

        # 统计来源多样性
        sources = articles["article_source"].dropna().tolist()

        # 统计事件类型分布（取众数）
        event_names = articles.get("event_name", pd.Series()).dropna().tolist()
        event_name = Counter(event_names).most_common(1)[0][0] if event_names else ""

        # 取最早的发布时间作为事件时间
        first_pub = articles_sorted.iloc[0].get("article_publish_time", "")
        last_pub = articles_sorted.iloc[-1].get("article_publish_time", "")

        # 标题合并：取最长/最完整标题
        titles = articles["article_title"].dropna().tolist()
        merged_title = max(titles, key=len) if titles else ""

        # 内容摘要：取最早文章的前200字
        first_content = str(articles_sorted.iloc[0].get("content", ""))
        merged_content = first_content[:300] if first_content else ""

        # 公司信息取众数
        codes = articles.get("trading_code", pd.Series()).dropna().tolist()
        secu_names = articles.get("secu_abbr", pd.Series()).dropna().tolist()

        return {
            "group_id": group_id,
            "event_name": event_name,
            "merged_title": merged_title,
            "merged_content": merged_content,
            "article_count": len(articles),
            "article_ids": articles["article_file_id"].tolist(),
            "sources": list(set(sources)),
            "source_count": len(set(sources)),
            "first_publish_time": str(first_pub),
            "last_publish_time": str(last_pub),
            "trading_codes": list(set(codes)),
            "secu_names": list(set(secu_names)),
        }

    @staticmethod
    def merge_all_groups(df: pd.DataFrame, label_column: str = "cluster_label") -> list[dict]:
        """
        将聚类后的 DataFrame 按标签合并为事件列表

        Args:
            df: 包含聚类标签的 DataFrame
            label_column: 标签列名

        Returns:
            合并后的事件列表（按文章数量降序）
        """
        # 排除噪声点（-1）
        clustered = df[df[label_column] != -1].copy()
        if clustered.empty:
            return []

        events = []
        for group_id, group_df in clustered.groupby(label_column):
            event = EventMerger.merge_group(group_df, int(group_id))
            events.append(event)

        # 按文章数量降序排列
        events.sort(key=lambda x: x["article_count"], reverse=True)
        return events
