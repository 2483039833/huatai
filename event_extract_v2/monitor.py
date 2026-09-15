# -*- coding: utf-8 -*-
"""
批量抽取进度监控（Windows 友好，无需 tail/grep）

用法（在 event_extract 目录下）：
    python monitor.py
"""
import json
import os

SOURCES = [
    ("无标签测试集", "unlabeled_test", 106387),
    ("无标签训练集", "unlabeled_train", 197577),
]

LOG = "full_run_all.log"


def _done(source: str) -> int:
    ckpt = os.path.join("outputs_event", f".{source}.ckpt.json")
    if not os.path.exists(ckpt):
        return 0
    try:
        return int(json.load(open(ckpt, encoding="utf-8")).get("done", 0))
    except Exception:
        return 0


def main() -> None:
    fails = 0
    if os.path.exists(LOG):
        try:
            fails = sum(1 for l in open(LOG, encoding="utf-8", errors="ignore") if "调用失败" in l)
        except Exception:
            fails = -1

    print("=" * 44)
    print("批量抽取进度监控")
    print("=" * 44)
    for name, src, total in SOURCES:
        d = _done(src)
        pct = d / total * 100 if total else 0
        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
        print(f"{name}：{d}/{total}  ({pct:5.2f}%)")
        print(f"  {bar}")
    print("-" * 44)
    print(f"LLM 调用失败：{fails} 次")
    print("=" * 44)
    if fails:
        print("⚠️  有 LLM 调用失败，注意检查 API 余额/网络")
    else:
        print("✅ 状态正常")


if __name__ == "__main__":
    main()
