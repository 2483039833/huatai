# -*- coding: utf-8 -*-
"""
LLM 客户端 (Qwen / DeepSeek / GLM)
===================================
三家厂商均提供 OpenAI 兼容的 /chat/completions 接口，因此统一封装。

配置优先级：
  1. 项目根目录 config.json（见 config.example.json）
  2. 环境变量 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
  3. 都没有 -> 进入离线规则模式（返回 None，触发规则回退）

用法：
  from event_ner.llm_client import LLMClient
  client = LLMClient()
  reply = client.chat([{"role":"user","content":"..."}])
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import requests

# 厂商预设（OpenAI 兼容 base_url）
PROVIDER_PRESETS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
}

# config.json 位置（event_extract 目录）
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        # 1. 读 config.json
        cfg: Dict = {}
        if os.path.exists(_CONFIG_PATH):
            try:
                with open(_CONFIG_PATH, encoding="utf-8") as f:
                    cfg = json.load(f).get("llm", {})
            except Exception:
                cfg = {}

        provider = provider or cfg.get("provider") or os.environ.get("LLM_PROVIDER") or "qwen"
        api_key = api_key or cfg.get("api_key") or os.environ.get("LLM_API_KEY") or ""
        preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["qwen"])
        base_url = base_url or cfg.get("base_url") or os.environ.get("LLM_BASE_URL") or preset["base_url"]
        model = model or cfg.get("model") or os.environ.get("LLM_MODEL") or preset["model"]

        self.provider = provider
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.available = bool(self.api_key)

        if not self.available:
            print("[LLM] 未配置 api_key，将使用离线规则回退")

    # ------------------------------------------------------------------
    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.1,
        json_mode: bool = True,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """
        调用 LLM，返回文本内容；离线规则模式返回 None。
        """
        if not self.available:
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # OpenAI 兼容的 JSON 输出约束（部分模型支持）
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            print(f"[LLM] 调用失败: {e}")
            return None
