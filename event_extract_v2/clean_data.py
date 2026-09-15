import pandas as pd
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

OUTPUT_DIR = _find("cleaned_data") + "/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    # 去除多余空白和特殊字符
    return str(text).strip()

def basic_clean(df, is_tagged=True, min_len=30):
    """
    is_tagged=True: 不删短文本（只删真正空/乱码）
    is_tagged=False: 可适当删极短文本（无标签数据量大）
    """
    df = df.copy()

    # 去重
    df = df.drop_duplicates(subset=['article_file_id'], keep='first')
    
    # 填充空来源
    if 'article_source' in df.columns:
        df['article_source'] = df['article_source'].fillna('未知')
    
    # 清理content
    df['content'] = df['content'].fillna('').astype(str)
    df['content'] = df['content'].apply(clean_text)
    
    if is_tagged:
        # 带标签数据：只删除content为空或纯乱码（<10字符且无中文）
        df = df[~(df['content'].str.len() < 10)]
    else:
        # 无标签数据：删除content小于30字符的行（节省存储）
        df = df[df['content'].str.len() >= min_len]
    
    # 统一时间
    if 'article_publish_time' in df.columns:
        df['article_publish_time'] = pd.to_datetime(df['article_publish_time'], errors='coerce')
    
    # 填充event_impact_analysis（若有）
    if 'event_impact_analysis' in df.columns:
        df['event_impact_analysis'] = df['event_impact_analysis'].fillna('')
    
    return df

print("=" * 60)
print("开始清洗全部4份数据（带标签数据保留短文本）")
print("=" * 60)

# 1. 带标签训练集
df1 = pd.read_excel(TRAIN_TAGGED_PATH)
df1 = basic_clean(df1, is_tagged=True)
df1.to_excel(f"{OUTPUT_DIR}/train_tagged_cleaned.xlsx", index=False)
print(f"1. train_tagged: {len(df1)} 条 (原始1525)")

# 2. 带标签测试集
df2 = pd.read_excel(TEST_TAGGED_PATH)
df2 = basic_clean(df2, is_tagged=True)
df2.to_excel(f"{OUTPUT_DIR}/test_tagged_cleaned.xlsx", index=False)
print(f"2. test_tagged:  {len(df2)} 条")

# 3. 无标签训练集（20万条）
df3 = pd.read_excel(UNLABELED_TRAIN_PATH)
df3 = basic_clean(df3, is_tagged=False, min_len=30)
df3.to_excel(f"{OUTPUT_DIR}/unlabeled_train_cleaned.xlsx", index=False)
print(f"3. unlabeled_train: {len(df3)} 条")

# 4. 无标签测试集（10.7万条）
df4 = pd.read_excel(UNLABELED_TEST_PATH)
df4 = basic_clean(df4, is_tagged=False, min_len=30)
df4.to_excel(f"{OUTPUT_DIR}/unlabeled_test_cleaned.xlsx", index=False)
print(f"4. unlabeled_test:  {len(df4)} 条")

print("\n✅ 全部清洗完成！")

# 汇总
print("\n【数据汇总】")
print(f"带标签训练集: {len(df1)} 条")
print(f"带标签测试集: {len(df2)} 条")
print(f"无标签训练集: {len(df3)} 条")
print(f"无标签测试集: {len(df4)} 条")