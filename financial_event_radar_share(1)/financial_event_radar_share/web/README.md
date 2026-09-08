# 事件雷达演示界面

先生成浏览器数据包：

```powershell
python event_radar/scripts/build_dashboard_data.py
```

随后在 `event_radar` 目录启动本地静态服务：

```powershell
python -m http.server 8080
```

在浏览器打开 `http://localhost:8080/web/`。

界面包含预警总览、优先级和状态筛选、事件详情、评分构成、推送理由以及事件时间线。
