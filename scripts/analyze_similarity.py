"""
重复组相似度分布分析
====================
分析 ground truth 组内文章对（正例）vs 随机文章对（负例）的 BGE 相似度分布，
为阈值法/图聚类提供依据。

用法：
    PYTHONIOENCODING=utf-8 python scripts/analyze_similarity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.preprocessing.loader import DataLoader
from src.clustering.evaluator import ClusterEvaluator


def main():
    print("=" * 60)
    print("重复组相似度分布分析")

    # 加载数据
    df = DataLoader.load_tagged_company_data()
    gt = ClusterEvaluator._ground_truth_from_dup_ids(df["duplication_id"])
    df["gt_label"] = gt

    # 加载缓存的 embedding
    emb = np.load("data/embeddings/company_content_bge_large.npy")
    print(f"Embedding: {emb.shape}")

    # 归一化（应为单位向量）
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / (norms + 1e-9)

    # 构造正例（组内对）与负例（跨组随机对）
    groups = {}
    for idx, label in enumerate(gt):
        if label != -1:
            groups.setdefault(label, []).append(idx)

    pos_pairs = []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pos_pairs.append((members[i], members[j]))

    # 随机负例（数量与正例相当）
    rng = np.random.default_rng(42)
    all_idx = np.where(gt == -1)[0]
    neg_pairs = []
    while len(neg_pairs) < len(pos_pairs):
        a, b = rng.choice(all_idx, size=2, replace=False)
        neg_pairs.append((int(a), int(b)))

    def compute_sims(pairs):
        sims = []
        for a, b in pairs:
            sim = float(np.dot(emb[a], emb[b]))
            sims.append(sim)
        return np.array(sims)

    pos_sims = compute_sims(pos_pairs)
    neg_sims = compute_sims(neg_pairs)

    print(f"\n组内正例对: {len(pos_sims)}, 组间负例对: {len(neg_sims)}")
    print(f"\n正例(同源)相似度分布: min={pos_sims.min():.4f}, "
          f"median={np.median(pos_sims):.4f}, max={pos_sims.max():.4f}")
    print(f"负例(非同源)相似度分布: min={neg_sims.min():.4f}, "
          f"median={np.median(neg_sims):.4f}, max={neg_sims.max():.4f}")

    # 不同阈值下的查准/查全
    print("\n阈值分析:")
    for threshold in [0.90, 0.92, 0.94, 0.95, 0.96, 0.97]:
        tp = int(np.sum(pos_sims >= threshold))
        fp = int(np.sum(neg_sims >= threshold))
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / len(pos_sims) if pos_sims.size else 0
        print(f"  threshold={threshold:.2f}: precision={precision:.4f}, "
              f"recall={recall:.4f} (TP={tp}, FP={fp})")

    # 检查重复组内公司代码一致性
    print("\n重复组公司代码一致性:")
    same_company = 0
    total = 0
    for members in groups.values():
        codes = df.iloc[members]["trading_code"].dropna().unique()
        if len(members) >= 2:
            total += 1
            if len(codes) <= 1:
                same_company += 1
    print(f"  同公司占比: {same_company}/{total} = {same_company/total:.2%}")


if __name__ == "__main__":
    main()
