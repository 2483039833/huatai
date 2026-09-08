import json
from datetime import datetime

from src.services.a_event_adapter import AEventAdapter


def test_a_event_adapter_maps_member_a_output(tmp_path):
    source = tmp_path / "a_output.json"
    source.write_text(
        json.dumps(
            [
                {
                    "article_file_id": "NEWS-1",
                    "article_title": "公司重大项目中标",
                    "article_source": "证券时报",
                    "content": "测试公司公告称已中标新能源项目。",
                    "publish_time": "2026-05-28 21:15:00",
                    "event_name": "重大项目中标",
                    "event_emotion": "正面",
                    "impact_direction": "正面",
                    "subjects": [{"secu_abbr": "测试公司", "trading_code": "600000"}],
                    "impact_objects": [{"name": "新能源项目", "type": "项目"}],
                    "industry": {"induname": "电力设备", "inducode": "280000"},
                    "event_time": "2026-05-28",
                    "confidence": 0.8,
                    "method": "llm",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    events = AEventAdapter.from_json(source)

    assert len(events) == 1
    event = events[0]
    assert event.article_file_id == "NEWS-1"
    assert event.article_publish_time == datetime(2026, 5, 28, 21, 15)
    assert event.event_time == datetime(2026, 5, 28)
    assert event.event_name == "重大项目中标"
    assert event.event_impact_direction == "正面"
    assert event.trading_code == "600000"
    assert event.secu_abbr == "测试公司"
    assert event.industry_code == "280000"
    assert event.industry_name == "电力设备"
    assert event.content == "测试公司公告称已中标新能源项目。"


def test_a_event_adapter_handles_empty_subjects_with_title_content():
    events = AEventAdapter.from_records(
        [
            {
                "article_file_id": "NEWS-2",
                "article_title": "行业政策发布",
                "event_name": "行业政策",
                "impact_direction": "中性",
                "subjects": [],
                "industry": {"induname": "银行"},
            }
        ]
    )

    event = events[0]
    assert event.trading_code is None
    assert event.secu_abbr is None
    assert event.industry_name == "银行"
    assert event.content == "行业政策发布 行业政策 中性 银行"
