"""
数据加载模块测试
"""
import sys
sys.path.insert(0, ".")

from src.preprocessing.loader import DataLoader


def test_load_company_data():
    df = DataLoader.load_tagged_company_data()
    assert len(df) > 0, "公司数据为空"
    assert "article_file_id" in df.columns, "缺少 article_file_id 列"
    print(f"[OK] 公司数据: {len(df)} 行, 列: {df.columns.tolist()}")


def test_load_industry_data():
    df = DataLoader.load_tagged_industry_data()
    assert len(df) > 0, "行业数据为空"
    print(f"[OK] 行业数据: {len(df)} 行")


def test_load_duplication_data():
    mapping = {"company_data": "load_tagged_company_data", "industry_data": "load_tagged_industry_data",
               "company_dup": "load_company_duplication", "industry_dup": "load_industry_duplication"}
    for name, method_name in mapping.items():
        df = getattr(DataLoader, method_name)()
        assert len(df) > 0, f"{name} 数据为空"
        print(f"[OK] {name}: {len(df)} 行")


def test_parse_dup_id():
    cases = [
        ("ID_A,ID_B", ["ID_A", "ID_B"]),
        ("ID_A", ["ID_A"]),
        ("", []),
    ]
    for input_str, expected in cases:
        result = DataLoader.parse_duplication_id(input_str)
        assert result == expected, f"解析失败: {input_str} → {result}"
    print("[OK] duplication_id 解析正常")


if __name__ == "__main__":
    test_load_company_data()
    test_load_industry_data()
    test_load_duplication_data()
    test_parse_dup_id()
    print("\n✅ 所有测试通过")
