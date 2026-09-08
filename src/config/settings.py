"""
pydantic-settings 全局配置管理
优先级：环境变量 > config.yaml > 默认值
"""
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml_config() -> dict:
    yaml_path = PROJECT_ROOT / "src" / "config" / "config.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


_yaml_config = load_yaml_config()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HUATAI_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ── Data ──
    data_raw_dir: str = str(PROJECT_ROOT)
    data_processed_dir: str = str(PROJECT_ROOT / "data" / "processed")
    data_embeddings_dir: str = str(PROJECT_ROOT / "data" / "embeddings")
    data_results_dir: str = str(PROJECT_ROOT / "data" / "results")

    # ── Embedding ──
    embedding_model_name: str = _yaml_config["embedding"]["model_name"]
    embedding_device: str = _yaml_config["embedding"]["device"]
    embedding_batch_size: int = _yaml_config["embedding"]["batch_size"]
    embedding_max_seq_length: int = _yaml_config["embedding"]["max_seq_length"]
    embedding_dimension: int = _yaml_config["embedding"]["dimension"]

    # ── Clustering ──
    clustering_method: Literal["graph", "hdbscan", "dbscan"] = _yaml_config["clustering"]["method"]
    hdbscan_min_cluster_size: int = _yaml_config["clustering"]["hdbscan"]["min_cluster_size"]
    hdbscan_min_samples: int = _yaml_config["clustering"]["hdbscan"]["min_samples"]
    dbscan_eps: float = _yaml_config["clustering"]["dbscan"]["eps"]
    dbscan_min_samples: int = _yaml_config["clustering"]["dbscan"]["min_samples"]

    # ── Retrieval ──
    retrieval_top_k: int = _yaml_config["retrieval"]["top_k"]
    similarity_threshold: float = _yaml_config["retrieval"]["similarity_threshold"]

    # ── Alert thresholds ──
    alert_p0_min_impact: float = _yaml_config.get("alert", {}).get("P0", {}).get("min_impact", 0.80)
    alert_p0_min_credibility: float = _yaml_config.get("alert", {}).get("P0", {}).get("min_credibility", 0.80)
    alert_p1_min_impact: float = _yaml_config.get("alert", {}).get("P1", {}).get("min_impact", 0.65)
    alert_p1_min_credibility: float = _yaml_config.get("alert", {}).get("P1", {}).get("min_credibility", 0.70)
    alert_p2_min_impact: float = _yaml_config.get("alert", {}).get("P2", {}).get("min_impact", 0.52)
    alert_p2_min_credibility: float = _yaml_config.get("alert", {}).get("P2", {}).get("min_credibility", 0.55)
    alert_p3_min_impact: float = _yaml_config.get("alert", {}).get("P3", {}).get("min_impact", 0.00)
    alert_p3_min_credibility: float = _yaml_config.get("alert", {}).get("P3", {}).get("min_credibility", 0.00)

    # ── API ──
    api_host: str = _yaml_config["api"]["host"]
    api_port: int = _yaml_config["api"]["port"]
    api_debug: bool = _yaml_config["api"]["debug"]

    # ── Logging ──
    log_level: str = _yaml_config["logging"]["level"]

    @property
    def company_train_path(self) -> str:
        return f"{self.data_raw_dir}/{_yaml_config['data']['raw']['company_train']}"

    @property
    def company_schema_path(self) -> str:
        return f"{self.data_raw_dir}/{_yaml_config['data']['raw']['event_schema_company']}"

    @property
    def industry_schema_path(self) -> str:
        return f"{self.data_raw_dir}/{_yaml_config['data']['raw']['event_schema_industry']}"

    @property
    def unlabeled_train_path(self) -> str:
        return f"{self.data_raw_dir}/{_yaml_config['data']['raw']['unlabeled_train']}"


settings = Settings()

