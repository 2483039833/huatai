"""
聚类评估工具
以数据的 duplication_id 为 ground truth，评估聚类效果
"""
from collections import defaultdict
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class ClusterEvaluator:
    """聚类评估器"""

    @staticmethod
    def _ground_truth_from_dup_ids(dup_id_series: "pd.Series") -> np.ndarray:
        """
        从 duplication_id 列构造 ground truth 标签。
        同一组 ID 的文章赋予相同标签。

        Returns:
            labels: shape (n,)，-1 表示无重复标签
        """
        from src.preprocessing.loader import DataLoader
        id_to_label = {}
        labels = []
        label_counter = 0
        for dup_str in dup_id_series:
            ids = DataLoader.parse_duplication_id(dup_str)
            if not ids:
                labels.append(-1)
            else:
                key = tuple(sorted(ids))
                if key not in id_to_label:
                    id_to_label[key] = label_counter
                    label_counter += 1
                labels.append(id_to_label[key])
        return np.array(labels, dtype=np.int32)

    @staticmethod
    def purity(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        计算聚类纯度 Purity
        purity = (1/N) * sum_k max_j |C_k ∩ L_j|
        """
        # sklearn>=1.5 中 contingency_matrix 移至 metrics.cluster 子模块
        try:
            from sklearn.metrics.cluster import contingency_matrix
        except ImportError:
            from sklearn.metrics import contingency_matrix
        contingency = contingency_matrix(true_labels, pred_labels)
        return float(np.sum(np.max(contingency, axis=0))) / np.sum(contingency)

    @staticmethod
    def evaluate(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> dict:
        """
        综合评估聚类效果

        Args:
            true_labels: ground truth 标签（-1 表示无标签）
            pred_labels: 聚类预测标签（-1 表示噪声点）

        Returns:
            包含各项指标的 dict
        """
        # 只评估有 ground truth 且被聚类的样本
        mask = (true_labels != -1) & (pred_labels != -1)
        if mask.sum() == 0:
            return {"error": "无有效样本用于评估"}

        true_valid = true_labels[mask]
        pred_valid = pred_labels[mask]

        metrics = {
            "n_samples_evaluated": int(mask.sum()),
            "purity": round(ClusterEvaluator.purity(true_valid, pred_valid), 4),
            "ari": round(adjusted_rand_score(true_valid, pred_valid), 4),
            "nmi": round(normalized_mutual_info_score(true_valid, pred_valid), 4),
        }
        return metrics

    @staticmethod
    def evaluate_group_detection(
        true_labels: np.ndarray,
        pred_labels: np.ndarray,
    ) -> dict:
        """
        评估"同源事件组"检测能力（放宽到组级别匹配）
        只要同一组内多数文章被聚到一起就算正确
        """
        from sklearn.metrics import adjusted_rand_score

        mask = true_labels != -1
        true_valid = true_labels[mask]
        pred_valid = pred_labels[mask]

        ari = adjusted_rand_score(true_valid, pred_valid) if len(set(true_valid)) > 1 else 0.0

        return {
            "ari_all": round(ari, 4),
            "n_true_groups": int(len(set(true_valid))),
            "n_pred_groups": int(len(set(pred_valid))),
        }
