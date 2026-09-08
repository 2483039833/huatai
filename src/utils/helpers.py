"""
通用工具函数
"""
import os
import time
import functools
from datetime import datetime, timedelta
from typing import Optional


def ensure_dir(path: str) -> str:
    """确保目录存在，返回路径"""
    os.makedirs(path, exist_ok=True)
    return path


def timer(func):
    """函数执行时间装饰器"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"[Timer] {func.__name__} 耗时 {elapsed:.2f}s")
        return result
    return wrapper


def batch_iter(data: list, batch_size: int = 64):
    """将列表分批 yield"""
    for i in range(0, len(data), batch_size):
        yield data[i: i + batch_size]


def format_excel_time(serial: Optional[float]) -> Optional[str]:
    """
    Excel 日期序列号 → 可读字符串
    示例: 46024.4497916667 → 2026-01-01 10:47
    """
    if serial is None:
        return None
    try:
        base = datetime(1899, 12, 30)
        return (base + timedelta(days=float(serial))).strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return str(serial)
