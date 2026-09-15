import pandas as pd
import numpy as np
import os

# ============================================================
# 数据路径：优先父目录（团队共享数据），回退当前目录
# ============================================================
def _find(name):
    for c in (os.path.join("..", name), name):
        if os.path.exists(c):
            return c
    return os.path.join("..", name)

TRAIN_TAGGED_PATH = _find("data") + "/news_with_tags_train.xlsx"
TEST_TAGGED_PATH = _find("data") + "/news_with_tags_test.xlsx"
UNLABELED_TRAIN_PATH = _find("data") + "/news_without_tags_train.xlsx"
UNLABELED_TEST_PATH = _find("data") + "/news_without_tags_test.xlsx"

# ============================================================
# 1. 重点探查：带标签的训练集（这是建模的基础）
# ============================================================
print("=" * 60)
print("【1】带标签训练集探查 (news_with_tags_train.xlsx)")
print("=" * 60)

df_tagged = pd.read_excel(TRAIN_TAGGED_PATH)

# 1.1 基本结构
print(f"\n>>> 数据形状: {df_tagged.shape}")
print(f"\n>>> 列名列表:\n{df_tagged.columns.tolist()}")

# 1.2 前3行预览（查看内容样例）
print(f"\n>>> 前3行数据预览:\n{df_tagged.head(3).to_string()}")

# 1.3 数据类型与空值
print(f"\n>>> 数据信息:")
print(df_tagged.info())

print(f"\n>>> 各列空值数量:")
print(df_tagged.isnull().sum())

# 1.4 关键标签分布（这是最重要的）
print(f"\n>>> 事件类型(event_name)分布:")
print(df_tagged['event_name'].value_counts())

print(f"\n>>> 情感极性(event_emotion)分布:")
print(df_tagged['event_emotion'].value_counts())

# 1.5 文本长度统计（评估正文质量）
df_tagged['content_len'] = df_tagged['content'].astype(str).str.len()
df_tagged['title_len'] = df_tagged['article_title'].astype(str).str.len()

print(f"\n>>> 正文长度统计 (字符数):")
print(df_tagged['content_len'].describe())

print(f"\n>>> 标题长度统计 (字符数):")
print(df_tagged['title_len'].describe())

# 1.6 去重情况
print(f"\n>>> 文章ID去重后数量: {df_tagged['article_file_id'].nunique()}")
print(f"\n>>> duplication_id 非空数量: {df_tagged['duplication_id'].notna().sum()}")

# ============================================================
# 2. 快速探查：无标签数据（仅看结构和规模）
# ============================================================
print("\n" + "=" * 60)
print("【2】无标签训练集探查 (news_without_tags_train.xlsx)")
print("=" * 60)

try:
    df_unlabeled_train = pd.read_excel(UNLABELED_TRAIN_PATH)
    print(f"\n>>> 数据形状: {df_unlabeled_train.shape}")
    print(f"\n>>> 列名: {df_unlabeled_train.columns.tolist()}")
    print(f"\n>>> 空值情况:\n{df_unlabeled_train.isnull().sum()}")
except FileNotFoundError:
    print(">>> 文件未找到，请确认路径后手动补充探查")

print("\n" + "=" * 60)
print("【3】无标签测试集探查 (news_without_tags_test.xlsx)")
print("=" * 60)

try:
    df_unlabeled_test = pd.read_excel(UNLABELED_TEST_PATH)
    print(f"\n>>> 数据形状: {df_unlabeled_test.shape}")
    print(f"\n>>> 列名: {df_unlabeled_test.columns.tolist()}")
except FileNotFoundError:
    print(">>> 文件未找到，请确认路径后手动补充探查")

print("\n" + "=" * 60)
print("【4】有标签测试集探查 (news_with_tags_test.xlsx)")
print("=" * 60)

df_labeled_test = pd.read_excel(TEST_TAGGED_PATH)


print(f"数据形状: {df_labeled_test.shape}")
print(f"\n列名:\n{df_labeled_test.columns.tolist()}")

# 确认标签列存在
print(f"\n标签列是否存在 event_name: {'event_name' in df_labeled_test.columns}")
print(f"标签列是否存在 event_emotion: {'event_emotion' in df_labeled_test.columns}")

if 'event_name' in df_labeled_test.columns:
    print(f"\n事件类型分布:\n{df_labeled_test['event_name'].value_counts()}")

if 'event_emotion' in df_labeled_test.columns:
    print(f"\n情感分布:\n{df_labeled_test['event_emotion'].value_counts()}")

# 空值情况
print(f"\n各列空值数量:\n{df_labeled_test.isnull().sum()}")

# 去重情况
print(f"\narticle_file_id 去重后: {df_labeled_test['article_file_id'].nunique()}")

print("\n" + "=" * 60)
print("数据探查完成！")
print("=" * 60)