"""
文本清洗模块测试
"""
import sys
sys.path.insert(0, ".")

from src.preprocessing.cleaner import TextCleaner


def test_clean_html():
    assert TextCleaner.clean_html("<p>正文</p>") == "正文"
    assert TextCleaner.clean_html("无标签") == "无标签"
    print("[OK] clean_html")


def test_clean_urls():
    result = TextCleaner.clean_urls("详情见 https://example.com/news")
    assert "https" not in result
    print("[OK] clean_urls")


def test_full_clean():
    raw = "<p>新闻内容，详见 https://x.com  </p>"
    result = TextCleaner.clean_text(raw)
    assert "<p>" not in result
    assert "https" not in result
    assert result == result.strip()
    print(f"[OK] full_clean: {result[:30]}")


def test_parse_excel_time():
    # 46024.4497916667 ≈ 2026-01-01
    result = TextCleaner.parse_excel_time(46024.4497916667)
    assert "2026" in result
    print(f"[OK] parse_excel_time: {result}")


if __name__ == "__main__":
    test_clean_html()
    test_clean_urls()
    test_full_clean()
    test_parse_excel_time()
    print("\n✅ 所有测试通过")
