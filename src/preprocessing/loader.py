"""
数据加载器：读取 xlsx 四个sheet + JSON 事件体系
"""
from pathlib import Path
from typing import Literal
import pandas as pd
import json

from src.config import settings


class DataLoader:
    """统一数据加载入口"""

    # Sheet 索引映射
    SHEET_MAP = {
        "company_data": 0,   # 公司事件标注数据
        "industry_data": 1,  # 行业事件标注数据
        "company_dup": 2,    # 公司重复/谣言匹配
        "industry_dup": 3,   # 行业重复/谣言匹配
    }

    @staticmethod
    def load_tagged_company_data() -> pd.DataFrame:
        """加载有标签的公司事件训练数据（Sheet1）"""
        df = pd.read_excel(
            settings.company_train_path,
            sheet_name=DataLoader.SHEET_MAP["company_data"],
            engine="openpyxl",
        )
        return df

    @staticmethod
    def load_tagged_industry_data() -> pd.DataFrame:
        """加载有标签的行业事件训练数据（Sheet2）"""
        df = pd.read_excel(
            settings.company_train_path,
            sheet_name=DataLoader.SHEET_MAP["industry_data"],
            engine="openpyxl",
        )
        return df

    @staticmethod
    def load_company_duplication() -> pd.DataFrame:
        """加载公司重复数据（Sheet3：谣言/转载 → 原文映射）"""
        df = pd.read_excel(
            settings.company_train_path,
            sheet_name=DataLoader.SHEET_MAP["company_dup"],
            engine="openpyxl",
        )
        return df

    @staticmethod
    def load_industry_duplication() -> pd.DataFrame:
        """加载行业重复数据（Sheet4）"""
        df = pd.read_excel(
            settings.company_train_path,
            sheet_name=DataLoader.SHEET_MAP["industry_dup"],
            engine="openpyxl",
        )
        return df

    @staticmethod
    def load_company_event_schema() -> dict:
        """加载公司事件体系 JSON"""
        with open(settings.company_schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_industry_event_schema() -> dict:
        """加载行业事件体系 JSON"""
        with open(settings.industry_schema_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_all_tagged() -> dict[str, pd.DataFrame]:
        """一次加载所有有标签数据"""
        return {
            "company_data": DataLoader.load_tagged_company_data(),
            "industry_data": DataLoader.load_tagged_industry_data(),
            "company_dup": DataLoader.load_company_duplication(),
            "industry_dup": DataLoader.load_industry_duplication(),
        }

    @staticmethod
    def parse_duplication_id(dup_id_str: str) -> list[str]:
        """
        解析 duplication_id 字段。
        格式可能是："ID_A,ID_B,ID_C" 或单个 "ID_A"
        返回 ID 列表。
        """
        if pd.isna(dup_id_str) or not dup_id_str:
            return []
        parts = str(dup_id_str).split(",")
        return [p.strip() for p in parts if p.strip()]
