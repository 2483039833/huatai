"""
文本清洗工具
"""
import re
from typing import Optional


class TextCleaner:
    """文本清洗：去除HTML、特殊字符、统一格式"""

    HTML_TAG_RE = re.compile(r"<[^>]+>")
    URL_RE = re.compile(r"https?://[^\s]+")
    WHITESPACE_RE = re.compile(r"\s+")
    SPECIAL_CHARS_RE = re.compile(r"[^一-鿿　-〿＀-￯\w\s]")
    SERIAL_NUM_RE = re.compile(r"[\d一二三四五六七八九十]+[、．\.\s]+")  # 序号如 "1、", "一、"

    @classmethod
    def clean_html(cls, text: str) -> str:
        """去除 HTML 标签"""
        return cls.HTML_TAG_RE.sub("", text)

    @classmethod
    def clean_urls(cls, text: str) -> str:
        """去除 URL"""
        return cls.URL_RE.sub("", text)

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """合并多余空白符"""
        return cls.WHITESPACE_RE.sub(" ", text).strip()

    @classmethod
    def clean_text(cls, text: Optional[str]) -> str:
        """
        全流程清洗
        1. 空值处理
        2. 去除HTML
        3. 去除URL
        4. 合并空白
        """
        if not text:
            return ""
        text = cls.clean_html(text)
        text = cls.clean_urls(text)
        text = cls.normalize_whitespace(text)
        return text

    @staticmethod
    def truncate(text: str, max_length: int = 512) -> str:
        """截断到最大长度（按字符）"""
        if len(text) <= max_length:
            return text
        return text[:max_length]

    @staticmethod
    def parse_excel_time(serial: float) -> str:
        """
        将 Excel 日期序列号转为 datetime 字符串。
        Excel 中 46024.4497916667 ≈ 2026-01-01 10:47
        """
        from datetime import datetime, timedelta
        base = datetime(1899, 12, 30)  # Excel 日期起点
        return (base + timedelta(days=serial)).strftime("%Y-%m-%d %H:%M:%S")
