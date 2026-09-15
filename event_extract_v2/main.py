# -*- coding: utf-8 -*-
"""
多源事件识别与要素抽取 —— 服务入口
====================================
本模块只负责：事件识别 + 要素抽取（其他模块由各自组员实现）。

启动：
  uvicorn main:app --host 0.0.0.0 --port 8000

接口文档：
  http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_event import router as event_router

app = FastAPI(
    title="多源事件识别与要素抽取",
    description="赛题二任务1：事件识别、主体/时间/影响对象/影响方向抽取、跨文本补全与标准化",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(event_router)


@app.get("/health")
async def health_check():
    return {"status": "ok", "module": "event_extraction"}
