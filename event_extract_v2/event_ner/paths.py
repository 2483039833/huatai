# -*- coding: utf-8 -*-
"""
数据路径解析
============
数据（原始 data/ 与清洗后 cleaned_data/）为团队共享，可能位于：
  - event_extract/ 同级目录（父目录 ../data、../cleaned_data）
  - 或 event_extract/ 内部（./data、./cleaned_data）

本模块自动探测，优先父目录（共享数据），其次当前目录。
"""

from __future__ import annotations

import os

# event_extract 目录（本文件在 event_extract/event_ner/ 下）
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_dir(name: str, marker: str = "") -> str:
    """在候选位置中查找含 marker 文件的目录，返回第一个存在的目录。"""
    candidates = [
        os.path.join(PACKAGE_DIR, "..", name),   # 父目录（共享数据）
        os.path.join(PACKAGE_DIR, name),          # 当前目录
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if marker:
            if os.path.exists(os.path.join(c, marker)):
                return c
        elif os.path.isdir(c):
            return c
    # 兜底返回父目录路径（即使不存在，让调用方报清晰错误）
    return os.path.normpath(candidates[0])


def data_dir() -> str:
    """原始数据目录（含 事件类型_*.json 与 xlsx）。"""
    return find_dir("data", marker="事件类型_标的.json")


def cleaned_dir() -> str:
    """清洗后数据目录。"""
    return find_dir("cleaned_data", marker="train_tagged_cleaned.xlsx")


def output_dir() -> str:
    """本模块输出目录。"""
    d = os.path.join(PACKAGE_DIR, "outputs_event")
    os.makedirs(d, exist_ok=True)
    return d
