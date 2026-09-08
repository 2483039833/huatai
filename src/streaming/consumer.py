"""
Kafka 消费者：实时消费事件并进行聚类处理
"""
import json
from typing import Callable, Optional


class KafkaConsumer:
    """Kafka 消费者"""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "raw-events",
        group_id: str = "event-cluster-group",
    ):
        from kafka import KafkaConsumer as KConsumer
        self.consumer = KConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
        )
        self.running = False

    def start(self, callback: Callable[[dict], None]):
        """开始消费，每条消息调用 callback"""
        self.running = True
        for msg in self.consumer:
            if not self.running:
                break
            callback(msg.value)

    def stop(self):
        self.running = False
        self.consumer.close()
