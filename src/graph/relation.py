"""
关系图谱：事件-公司-行业关联
"""
from collections import defaultdict
import networkx as nx


class RelationGraph:
    """事件关系图谱"""

    def __init__(self):
        self.graph = nx.Graph()

    def build_from_events(self, events: list[dict]) -> None:
        """
        从事件列表构建图谱

        节点类型: Company, Industry, Event, News
        """
        for event in events:
            group_id = str(event.get("group_id", ""))
            event_name = event.get("event_name", "")

            # 事件节点
            event_node = f"event_{group_id}"
            self.graph.add_node(event_node, type="event", label=event_name)

            # 关联公司
            for code in event.get("trading_codes", []):
                company_node = f"company_{code}"
                self.graph.add_node(company_node, type="company", label=code)
                self.graph.add_edge(event_node, company_node, relation="involves")

            # 关联新闻
            for aid in event.get("article_ids", []):
                news_node = f"news_{aid}"
                self.graph.add_node(news_node, type="news", label=aid[:20])
                self.graph.add_edge(event_node, news_node, relation="has_article")

    def get_company_events(self, trading_code: str) -> list[str]:
        """获取公司的关联事件"""
        company_node = f"company_{trading_code}"
        if company_node not in self.graph:
            return []
        neighbors = list(self.graph.neighbors(company_node))
        return [n for n in neighbors if n.startswith("event_")]

    def get_event_news(self, group_id: str) -> list[str]:
        """获取事件的关联新闻"""
        event_node = f"event_{group_id}"
        if event_node not in self.graph:
            return []
        neighbors = list(self.graph.neighbors(event_node))
        return [n for n in neighbors if n.startswith("news_")]

    def stats(self) -> dict:
        """图谱统计"""
        node_types = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            node_types[data.get("type", "unknown")] += 1
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "node_types": dict(node_types),
        }
