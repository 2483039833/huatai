"""
FastAPI 应用入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings

app = FastAPI(
    title="Huatai Event Intelligence API",
    description="上市公司事件识别、聚类、可信度评估与预警系统",
    version="0.1.0",
)

# CORS 配置（允许前端跨域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "huatai-event-intelligence"}


@app.get("/")
def root():
    return {
        "name": "Huatai Event Intelligence API",
        "version": "0.1.0",
        "docs": "/docs",
    }


# 路由注册：四类接口均读取端到端流水线生成的真实结果文件。
from src.api.routers import events, timeline, graph, alert

app.include_router(events.router, prefix="/api/v1/events", tags=["events"])
app.include_router(timeline.router, prefix="/api/v1/timeline", tags=["timeline"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["graph"])
app.include_router(alert.router, prefix="/api/v1/alert", tags=["alert"])
