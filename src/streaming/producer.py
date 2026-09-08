"""事件流生产者：支持本地数据回放与 Kafka 实时发布。"""
import json
import time
import random
from datetime import datetime
from typing import Optional
import pandas as pd


class ReplayProducer:
    """按固定间隔回放 DataFrame 中的历史事件。"""

    def __init__(self, df: pd.DataFrame, interval: float = 0.5):
        """
        Args:
            df: 事件 DataFrame
            interval: 回放发送间隔（秒）
        """
        self.df = df
        self.interval = interval
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self) -> dict:
        if self.index >= len(self.df):
            raise StopIteration
        row = self.df.iloc[self.index].to_dict()
        self.index += 1
        time.sleep(self.interval)
        return {
            "data": row,
            "timestamp": datetime.now().isoformat(),
            "seq": self.index,
        }

    def reset(self):
        self.index = 0


class KafkaProducer:
    """Kafka 生产者（需要 Kafka 集群时启用）"""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        from kafka import KafkaProducer as KProducer
        self.producer = KProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )

    def send(self, topic: str, event: dict):
        self.producer.send(topic, value=event)
