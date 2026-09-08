"""
日志配置
"""
import sys
from pathlib import Path
from loguru import logger
from src.config import settings


def get_logger(name: str = "huatai"):
    """获取配置好的 logger 实例"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger.remove()  # 移除默认 handler

    # 控制台输出
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<5}</level> | {message}",
        colorize=True,
    )

    # 文件输出
    logger.add(
        log_dir / "app.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
        format="{time} | {level:<5} | {message}",
    )

    return logger
