param(
    [string]$OutputPath = ""
)

$ErrorActionPreference = "Stop"

function Rgb([int]$R, [int]$G, [int]$B) {
    return $R + ($G -shl 8) + ($B -shl 16)
}

$C = @{
    Navy = Rgb 15 42 88
    Blue = Rgb 37 99 235
    Cyan = Rgb 14 165 233
    Teal = Rgb 5 150 105
    Green = Rgb 22 163 74
    Orange = Rgb 234 88 12
    Red = Rgb 220 38 38
    Ink = Rgb 22 32 51
    Muted = Rgb 101 115 138
    Light = Rgb 245 247 251
    Line = Rgb 226 232 240
    White = Rgb 255 255 255
    PaleBlue = Rgb 239 246 255
    PaleGreen = Rgb 236 253 245
    PaleOrange = Rgb 255 247 237
    PaleRed = Rgb 254 242 242
}

$msoFalse = 0
$msoTrue = -1
$msoShapeRectangle = 1
$msoShapeRoundedRectangle = 5
$msoShapeOval = 9
$msoTextOrientationHorizontal = 1
$ppLayoutBlank = 12
$ppAlignLeft = 1
$ppAlignCenter = 2
$ppAlignRight = 3
$ppSaveAsOpenXMLPresentation = 24
$Scale = 0.75 # PowerPoint's screen16x9 canvas is 720 × 405 points.

function Add-Rect($Slide, [double]$X, [double]$Y, [double]$W, [double]$H, [int]$Fill, [int]$Line = -1, [int]$Type = 1) {
    $shape = $Slide.Shapes.AddShape($Type, ($X * $Scale), ($Y * $Scale), ($W * $Scale), ($H * $Scale))
    $shape.Fill.Visible = $msoTrue
    $shape.Fill.ForeColor.RGB = $Fill
    if ($Line -lt 0) {
        $shape.Line.Visible = $msoFalse
    } else {
        $shape.Line.Visible = $msoTrue
        $shape.Line.ForeColor.RGB = $Line
        $shape.Line.Weight = 0.8
    }
    return $shape
}

function Add-Line($Slide, [double]$X1, [double]$Y1, [double]$X2, [double]$Y2, [int]$Color, [double]$Weight = 1.5) {
    $line = $Slide.Shapes.AddLine(($X1 * $Scale), ($Y1 * $Scale), ($X2 * $Scale), ($Y2 * $Scale))
    $line.Line.ForeColor.RGB = $Color
    $line.Line.Weight = $Weight
    return $line
}

function Add-Text($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [double]$H, [double]$Size = 16, [int]$Color = 0, [int]$Bold = 0, [int]$Align = 1) {
    $shape = $Slide.Shapes.AddTextbox($msoTextOrientationHorizontal, ($X * $Scale), ($Y * $Scale), ($W * $Scale), ($H * $Scale))
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = $msoTrue
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
    $shape.TextFrame.TextRange.Font.Size = ($Size * $Scale)
    $shape.TextFrame.TextRange.Font.Bold = $Bold
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Label($Slide, [string]$Text, [double]$X, [double]$Y, [double]$W, [int]$Fill, [int]$Color) {
    $box = Add-Rect $Slide $X $Y $W 22 $Fill $Fill $msoShapeRoundedRectangle
    Add-Text $Slide $Text ($X + 7) ($Y + 4) ($W - 14) 15 9.5 $Color $msoTrue $ppAlignCenter | Out-Null
    return $box
}

function Add-Card($Slide, [double]$X, [double]$Y, [double]$W, [double]$H, [string]$Label, [string]$Value, [string]$Note, [int]$Accent) {
    Add-Rect $Slide $X $Y $W $H $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Rect $Slide $X $Y 5 $H $Accent $Accent $msoShapeRoundedRectangle | Out-Null
    $compact = $H -lt 90
    $valueY = if ($compact) { $Y + 32 } else { $Y + 38 }
    $valueHeight = if ($compact) { 24 } else { 32 }
    $valueSize = if ($compact) { 18 } else { 24 }
    $noteY = if ($compact) { $Y + $H - 16 } else { $Y + $H - 22 }
    $noteSize = if ($compact) { 8 } else { 9.5 }
    Add-Text $Slide $Label ($X + 18) ($Y + 15) ($W - 30) 18 11 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Text $Slide $Value ($X + 18) $valueY ($W - 30) $valueHeight $valueSize $C.Ink $msoTrue $ppAlignLeft | Out-Null
    Add-Text $Slide $Note ($X + 18) $noteY ($W - 30) 14 $noteSize $C.Muted $msoFalse $ppAlignLeft | Out-Null
}

function Add-Bullets($Slide, [string[]]$Items, [double]$X, [double]$Y, [double]$W, [double]$LineHeight = 27, [double]$Size = 13.5, [int]$Color = 0) {
    $text = ($Items | ForEach-Object { "• " + $_ }) -join "`r`n"
    Add-Text $Slide $text $X $Y $W (($Items.Count * $LineHeight) + 6) $Size $Color $msoFalse $ppAlignLeft | Out-Null
}

function Add-Bar($Slide, [string]$Label, [double]$Value, [double]$Max, [double]$X, [double]$Y, [double]$W, [int]$Color, [string]$Text = "") {
    Add-Text $Slide $Label $X ($Y - 1) 170 17 11.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    Add-Rect $Slide ($X + 175) $Y $W 14 $C.Line $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Rect $Slide ($X + 175) $Y ([Math]::Max(8, $W * $Value / $Max)) 14 $Color $Color $msoShapeRoundedRectangle | Out-Null
    $display = if ($Text) { $Text } else { "{0:N2}%" -f $Value }
    Add-Text $Slide $display ($X + 183 + $W) ($Y - 1) 60 17 11.5 $C.Ink $msoTrue $ppAlignRight | Out-Null
}

function Add-Header($Slide, [string]$Title, [string]$Kicker, [int]$Page) {
    Add-Text $Slide $Kicker 48 26 480 16 10 $C.Blue $msoTrue $ppAlignLeft | Out-Null
    Add-Text $Slide $Title 48 48 820 40 27 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Line $Slide 48 94 912 94 $C.Line 1 | Out-Null
    Add-Text $Slide "金融事件雷达 · 赛题二答辩" 48 516 320 14 8.5 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Text $Slide ("{0:00}" -f $Page) 886 516 26 14 8.5 $C.Muted $msoFalse $ppAlignRight | Out-Null
}

function New-ContentSlide($Presentation, [int]$Page, [string]$Title, [string]$Kicker) {
    $slide = $Presentation.Slides.Add($Page, $ppLayoutBlank)
    Add-Rect $slide 0 0 960 540 $C.Light $C.Light | Out-Null
    Add-Header $slide $Title $Kicker $Page
    return $slide
}

$scriptDirectory = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { Join-Path (Get-Location) "scripts" } else { $PSScriptRoot }
$projectRoot = Split-Path -Parent $scriptDirectory
$deliverables = Join-Path $projectRoot "deliverables"
New-Item -ItemType Directory -Force -Path $deliverables | Out-Null
if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $OutputPath = Join-Path $deliverables "金融事件雷达_赛题二答辩.pptx"
}
if ([IO.Path]::GetExtension($OutputPath).ToLowerInvariant() -ne ".pptx") {
    throw "OutputPath must end with .pptx"
}

$app = $null
$presentation = $null
try {
    $app = New-Object -ComObject PowerPoint.Application
    $app.Visible = $msoTrue
    try { $app.WindowState = 2 } catch { }
    $presentation = $app.Presentations.Add()
    $presentation.PageSetup.SlideSize = 15 # 16:9

    # 01 Cover
    $slide = $presentation.Slides.Add(1, $ppLayoutBlank)
    Add-Rect $slide 0 0 960 540 $C.Navy $C.Navy | Out-Null
    Add-Rect $slide 560 0 400 540 (Rgb 24 65 132) (Rgb 24 65 132) | Out-Null
    Add-Rect $slide 650 70 260 260 (Rgb 37 99 235) (Rgb 37 99 235) $msoShapeOval | Out-Null
    Add-Rect $slide 720 135 170 170 (Rgb 14 165 233) (Rgb 14 165 233) $msoShapeOval | Out-Null
    Add-Label $slide "2026 泰证券·淮泰大学 AI+创新大赛" 54 56 246 (Rgb 30 80 156) $C.White | Out-Null
    Add-Text $slide "金融事件雷达" 54 142 510 58 36 $C.White $msoTrue $ppAlignLeft | Out-Null
    Add-Text $slide "赛题二：面向上市公司的事件驱动智能识别、可信评估与脉络追踪" 54 213 465 58 18 $C.White $msoFalse $ppAlignLeft | Out-Null
    Add-Line $slide 54 291 440 291 (Rgb 103 151 222) 1.5 | Out-Null
    Add-Text $slide "识别 · 证据评估 · 多源聚合 · 时间线追踪 · 分级预警" 54 314 500 28 15 (Rgb 203 222 255) $msoFalse $ppAlignLeft | Out-Null
    Add-Text $slide "可复现 · 可解释 · 人机协同" 54 454 360 24 14 (Rgb 203 222 255) $msoTrue $ppAlignLeft | Out-Null
    Add-Text $slide "项目答辩材料" 54 487 180 16 10 (Rgb 156 196 255) $msoFalse $ppAlignLeft | Out-Null

    # 02 Problem
    $slide = New-ContentSlide $presentation 2 "从资讯洪流到可行动事件" "赛题背景与项目目标"
    Add-Text $slide "金融研究与投研场景的难点，不是缺少新闻，而是缺少对【同一事件】的统一理解和可追溯判断。" 48 112 830 28 14 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    $problems = @("多源异构：公告、媒体、互动平台与行业资讯表达不同", "重复传播：同一事项多次转载，容易重复推送", "要素缺失：主体、时间、方向和证据分散在文本中", "单篇孤立：缺少事件确认、澄清与影响更新的过程", "优先级不一：高影响与低证据事件需要不同处置")
    for ($i = 0; $i -lt $problems.Count; $i++) {
        $y = 156 + ($i * 55)
        Add-Rect $slide 48 $y 400 42 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
        Add-Rect $slide 60 ($y + 9) 24 24 $C.PaleBlue $C.PaleBlue $msoShapeOval | Out-Null
        Add-Text $slide ("{0}" -f ($i + 1)) 60 ($y + 13) 24 14 10 $C.Blue $msoTrue $ppAlignCenter | Out-Null
        Add-Text $slide $problems[$i] 98 ($y + 11) 330 20 11.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    }
    Add-Rect $slide 506 156 400 294 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "项目目标：构建事件闭环" 532 181 330 24 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    $goals = @("识别：事件类型 + 主体 + 时间 + 方向 + 证据", "评估：来源、交叉印证、官方线索与传闻风险", "聚合：将同源资讯归并为一个事件簇", "追踪：维护报道、印证、确认、澄清等状态", "行动：将影响、可信度与时效转为分级预警")
    Add-Bullets $slide $goals 532 224 330 36 13 $C.Ink
    Add-Label $slide "不是新闻列表，而是可复核事件资产" 532 396 302 $C.PaleGreen $C.Teal | Out-Null

    # 03 Requirement mapping
    $slide = New-ContentSlide $presentation 3 "赛题要求—系统实现—交付物映射" "需求闭环"
    $rows = @(
        @{Title="多源事件识别"; Impl="事件体系约束 + 公司语义检索 / 行业文本分类"; Out="结构化事件预测：类型、主体、时间、方向、证据"; Color=$C.Blue},
        @{Title="可信评估 / 辨伪"; Impl="来源权威性、交叉印证、官方线索、具体性、传闻风险"; Out="证据可信度与理由；不等同于事实真伪判定"; Color=$C.Teal},
        @{Title="同源聚合与脉络追踪"; Impl="稀有 shingle 召回 + 相似度确认 + 并查集聚类 + 状态机"; Out="事件簇、时间线、确认/澄清等里程碑"; Color=$C.Orange},
        @{Title="影响分级与预警"; Impl="证据、重要性、信息广度、状态、时效、分类置信度"; Out="P0–P3、推送理由、人工复核与看板"; Color=$C.Red}
    )
    for ($i = 0; $i -lt $rows.Count; $i++) {
        $y = 124 + ($i * 88)
        $row = $rows[$i]
        Add-Rect $slide 48 $y 864 70 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
        Add-Rect $slide 48 $y 9 70 $row.Color $row.Color $msoShapeRoundedRectangle | Out-Null
        Add-Text $slide $row.Title 74 ($y + 15) 175 22 15 $C.Navy $msoTrue $ppAlignLeft | Out-Null
        Add-Text $slide $row.Impl 258 ($y + 13) 330 40 11.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
        Add-Line $slide 608 ($y + 10) 608 ($y + 60) $C.Line 1 | Out-Null
        Add-Text $slide $row.Out 630 ($y + 13) 250 40 11 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    }
    Add-Label $slide "评审口径：有标签任务报告指标；无真值模块报告运行覆盖与可解释机制" 48 477 660 $C.PaleOrange $C.Orange | Out-Null

    # 04 Data
    $slide = New-ContentSlide $presentation 4 "数据资产与验证治理" "数据与评测边界"
    Add-Card $slide 48 126 200 106 "事件体系" "120 条定义" "44 类公司事件 · 20 类行业事件" $C.Blue
    Add-Card $slide 270 126 200 106 "有标签事件新闻" "2,081 / 1,046" "训练 / 留出诊断" $C.Teal
    Add-Card $slide 492 126 200 106 "重复事件样本" "1,499 / 753" "训练校准 / 测试诊断" $C.Orange
    Add-Card $slide 714 126 198 106 "无标签新闻" "307,888 条" "用于本地语义表示学习" $C.Green
    Add-Rect $slide 48 262 864 184 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "严格的评测隔离" 74 286 280 24 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Rect $slide 74 334 190 52 $C.PaleBlue $C.PaleBlue $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "带标签训练集" 94 346 150 17 14 $C.Blue $msoTrue $ppAlignCenter | Out-Null
    Add-Text $slide "分层切分" 281 349 70 14 11 $C.Muted $msoFalse $ppAlignCenter | Out-Null
    Add-Rect $slide 366 334 190 52 $C.PaleGreen $C.PaleGreen $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "内部验证集" 386 346 150 17 14 $C.Teal $msoTrue $ppAlignCenter | Out-Null
    Add-Text $slide "仅用于选择模型/阈值" 367 395 188 15 9.5 $C.Muted $msoFalse $ppAlignCenter | Out-Null
    Add-Text $slide "最终诊断" 588 349 70 14 11 $C.Muted $msoFalse $ppAlignCenter | Out-Null
    Add-Rect $slide 674 334 190 52 $C.PaleOrange $C.PaleOrange $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "带标签留出集" 694 346 150 17 14 $C.Orange $msoTrue $ppAlignCenter | Out-Null
    Add-Text $slide "不参与反复调参" 676 395 186 15 9.5 $C.Muted $msoFalse $ppAlignCenter | Out-Null
    Add-Text $slide "→" 293 344 45 24 22 $C.Blue $msoTrue $ppAlignCenter | Out-Null
    Add-Text $slide "→" 590 344 45 24 22 $C.Orange $msoTrue $ppAlignCenter | Out-Null
    Add-Label $slide "避免测试标签泄漏，优先优化 Macro-F1 以兼顾长尾事件类别" 48 468 540 $C.PaleBlue $C.Blue | Out-Null

    # 05 architecture
    $slide = New-ContentSlide $presentation 5 "端到端系统架构" "从原始资讯到可执行预警"
    $steps = @(
        @{t="多源新闻"; s="公告 · 媒体 · 行业资讯"; c=$C.Blue},
        @{t="事件识别"; s="体系约束 + 要素抽取"; c=$C.Cyan},
        @{t="事件聚合"; s="去重 · 同源归并"; c=$C.Teal},
        @{t="可信与追踪"; s="证据链 · 时间线"; c=$C.Orange},
        @{t="分级预警"; s="影响评分 · 复核"; c=$C.Red}
    )
    for ($i = 0; $i -lt $steps.Count; $i++) {
        $x = 48 + ($i * 178)
        $step = $steps[$i]
        Add-Rect $slide $x 170 148 100 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
        Add-Rect $slide ($x + 49) 145 50 50 $step.c $step.c $msoShapeOval | Out-Null
        Add-Text $slide ("{0}" -f ($i + 1)) ($x + 49) 160 50 18 14 $C.White $msoTrue $ppAlignCenter | Out-Null
        Add-Text $slide $step.t ($x + 12) 205 124 19 15 $C.Navy $msoTrue $ppAlignCenter | Out-Null
        Add-Text $slide $step.s ($x + 12) 232 124 30 10.5 $C.Muted $msoFalse $ppAlignCenter | Out-Null
        if ($i -lt ($steps.Count - 1)) { Add-Text $slide "→" ($x + 151) 202 24 22 19 $C.Muted $msoTrue $ppAlignCenter | Out-Null }
    }
    Add-Rect $slide 48 332 864 112 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "统一事件对象（贯穿全链路）" 74 354 310 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Label $slide "事件类型" 74 393 95 $C.PaleBlue $C.Blue | Out-Null
    Add-Label $slide "主体 / 行业" 183 393 104 $C.PaleGreen $C.Teal | Out-Null
    Add-Label $slide "时间 / 方向" 301 393 104 $C.PaleOrange $C.Orange | Out-Null
    Add-Label $slide "证据 / 来源" 419 393 104 $C.PaleRed $C.Red | Out-Null
    Add-Label $slide "置信度 / 复核" 537 393 120 $C.PaleBlue $C.Blue | Out-Null
    Add-Label $slide "簇 / 时间线 / 告警" 671 393 170 $C.PaleGreen $C.Teal | Out-Null

    # 06 event recognition
    $slide = New-ContentSlide $presentation 6 "事件识别：体系约束 + 语义检索" "核心 AI 模块"
    Add-Rect $slide 48 124 490 320 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "公司事件识别" 74 148 210 24 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    $companySteps = @("证券代码 / 公司名定位候选事件体系", "在同公司历史标题中检索相近事件", "融合：10% 字符相似度 + 90% 本地语义相似度", "输出事件类型、置信度与人工复核标记")
    for ($i = 0; $i -lt $companySteps.Count; $i++) {
        $y = 193 + ($i * 52)
        Add-Rect $slide 76 $y 28 28 $C.PaleBlue $C.PaleBlue $msoShapeOval | Out-Null
        Add-Text $slide ("{0}" -f ($i + 1)) 76 ($y + 7) 28 14 10 $C.Blue $msoTrue $ppAlignCenter | Out-Null
        Add-Text $slide $companySteps[$i] 120 ($y + 5) 376 24 12.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    }
    Add-Rect $slide 562 124 350 150 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "行业事件识别" 588 148 210 24 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("行业体系约束候选标签", "正文字符 n-gram 朴素贝叶斯", "事件方向采用监督字符模型") 590 193 280 30 12.5 $C.Ink
    Add-Rect $slide 562 294 350 150 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "结构化输出" 588 318 210 24 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("主体、事件时间、正/负/中性方向", "原文证据片段与候选数量", "低边际分数标记 review_required") 590 360 280 26 11.5 $C.Ink
    Add-Label $slide "分类与抽取同时可解释、可回溯" 48 467 276 $C.PaleGreen $C.Teal | Out-Null

    # 07 optimisation
    $slide = New-ContentSlide $presentation 7 "优化：只接入通过验证的模型" "模型选择过程"
    Add-Text $slide "候选模型在训练集的分层验证上比较 Macro-F1，留出集仅用于最终诊断。" 48 118 760 22 13 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    Add-Rect $slide 48 155 478 248 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "个股事件 · 训练集内部验证" 74 177 310 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bar $slide "同公司字符近邻" 76.00 100 74 226 275 $C.Blue "76.00%"
    Add-Bar $slide "纯语义近邻" 78.87 100 74 274 275 $C.Cyan "78.87%"
    Add-Bar $slide "语义融合（选中）" 79.75 100 74 322 275 $C.Teal "79.75%"
    Add-Label $slide "选择规则：Macro-F1 优先，其次 Accuracy" 74 363 295 $C.PaleGreen $C.Teal | Out-Null
    Add-Rect $slide 550 155 362 248 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "未接入的候选" 576 177 220 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("标题 / 导语 TF-IDF + LinearSVC：整体 Accuracy 有提升，但验证 Macro-F1 未超过融合方案", "SVM + 近邻置信度融合：验证选择仍是纯近邻/融合主策略", "轻量 TextCNN：小样本且无外部预训练时显著落后", "原则：复杂度不是收益，未通过验证不进入线上链路") 576 222 300 36 11.3 $C.Ink
    Add-Label $slide "本地 Word2Vec：仅使用训练侧有标签与无标签标题" 48 438 388 $C.PaleBlue $C.Blue | Out-Null

    # 08 aggregation credibility timeline
    $slide = New-ContentSlide $presentation 8 "事件聚合、证据评估与时间线追踪" "把单篇资讯组织为事件资产"
    Add-Rect $slide 48 126 272 308 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "同源聚合 / 去重" 74 149 200 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("稀有字符 shingle 召回候选", "标题 / 正文包含与相似度确认", "并查集连接成事件簇", "753 条测试新闻 → 429 个事件簇") 74 193 215 32 12 $C.Ink
    Add-Label $slide "Pair F1 91.21%" 74 348 142 $C.PaleBlue $C.Blue | Out-Null
    Add-Label $slide "B3 F1 97.94%" 74 381 142 $C.PaleGreen $C.Teal | Out-Null
    Add-Rect $slide 344 126 272 308 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "证据可信度" 370 149 170 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("来源权威性", "多源交叉印证", "官方公告/监管线索", "文本具体性、簇内一致性", "传闻风险扣分") 370 193 205 29 12 $C.Ink
    Add-Text $slide "高 7 · 中 60 · 低 195 · 很低 167" 370 365 205 22 10.5 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Rect $slide 640 126 272 308 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "事件时间线" 666 149 170 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    $timelineStates = @(@("reported", $C.Blue), @("corroborated", $C.Teal), @("confirmed", $C.Green), @("clarified", $C.Orange))
    for ($i = 0; $i -lt $timelineStates.Count; $i++) {
        $y = 199 + ($i * 38)
        Add-Rect $slide 671 $y 15 15 $timelineStates[$i][1] $timelineStates[$i][1] $msoShapeOval | Out-Null
        Add-Text $slide $timelineStates[$i][0] 699 ($y - 2) 132 18 12 $C.Ink $msoTrue $ppAlignLeft | Out-Null
        if ($i -lt ($timelineStates.Count - 1)) { Add-Line $slide 678 ($y + 16) 678 ($y + 36) $C.Line 1.5 | Out-Null }
    }
    Add-Text $slide "429 条时间线 · 官方确认节点 262" 666 365 210 24 10.5 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Label $slide "可信度 = 证据充分程度，不是事实真伪自动判定" 48 466 430 $C.PaleOrange $C.Orange | Out-Null

    # 09 alert
    $slide = New-ContentSlide $presentation 9 "影响分级、预警与人机协同" "从信息到行动建议"
    Add-Rect $slide 48 128 414 294 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "影响评分框架" 74 151 230 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Text $slide "Impact Score =" 74 194 130 20 15 $C.Blue $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("证据可信度", "事件重要性", "信息广度", "状态确认", "时效性", "事件分类置信度") 212 188 165 28 12.5 $C.Ink
    Add-Label $slide "澄清事件 → 撤回 / 更新" 74 365 184 $C.PaleOrange $C.Orange | Out-Null
    Add-Label $slide "低置信高影响 → 人工复核" 272 365 165 $C.PaleRed $C.Red | Out-Null
    Add-Rect $slide 490 128 422 294 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "当前告警产出（429 个事件）" 516 151 300 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bar $slide "P3 即时推送" 4 49 516 207 190 $C.Red "4"
    Add-Bar $slide "P2 优先推送" 49 49 516 247 190 $C.Orange "49"
    Add-Bar $slide "P1 观察 / 复核" 228 228 516 287 190 $C.Blue "228"
    Add-Bar $slide "P0 归档 / 更新" 148 228 516 327 190 $C.Teal "148"
    Add-Text $slide "人工复核 1 条 · 澄清更新 9 条" 516 375 280 20 11 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Label $slide "金融场景原则：高影响不等于高可信，自动化必须保留复核出口" 48 464 545 $C.PaleBlue $C.Blue | Out-Null

    # 10 evaluation protocol
    $slide = New-ContentSlide $presentation 10 "评测设计与报告口径" "结果可解释，结论可复核"
    Add-Rect $slide 48 126 420 296 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "有标签任务：报告性能指标" 74 150 300 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("个股 / 行业事件识别：Accuracy、Macro-F1、Weighted-F1", "同源去重：Pair Precision、Recall、F1 与簇级 B3 F1", "类别级 F1、主要混淆对、置信度分层质量", "模型选择：训练集内部的分层验证") 74 195 340 34 12.2 $C.Ink
    Add-Label $slide "留出集只做最终诊断" 74 363 212 $C.PaleGreen $C.Teal | Out-Null
    Add-Rect $slide 492 126 420 296 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "无真值任务：报告运行覆盖" 518 150 300 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("可信度：证据级别、来源层级与多源覆盖，不宣称事实判定准确率", "时间线：状态、里程碑和官方确认覆盖，不虚构状态 F1", "告警：优先级与动作分布，不虚构预警命中率", "业务规则可审计，后续需增加人工审核标签校准") 518 195 345 34 12.2 $C.Ink
    Add-Label $slide "不把规则输出包装为已验证模型性能" 518 363 276 $C.PaleOrange $C.Orange | Out-Null
    Add-Text $slide "这也是金融场景中【严谨可信】的必要条件。" 48 462 620 22 15 $C.Navy $msoTrue $ppAlignLeft | Out-Null

    # 11 results
    $slide = New-ContentSlide $presentation 11 "核心评测结果：优化有效，边界清晰" "有标签诊断集表现"
    Add-Rect $slide 48 126 418 292 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "个股事件：语义融合带来稳定增益" 74 150 340 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Text $slide "Macro-F1" 74 201 100 18 12 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Bar $slide "原字符近邻" 77.26 100 74 229 130 $C.Blue "77.26%"
    Add-Bar $slide "当前语义融合" 80.09 100 74 272 130 $C.Teal "80.09%"
    Add-Text $slide "Accuracy：79.32% → 82.33%" 74 327 240 20 12.5 $C.Ink $msoTrue $ppAlignLeft | Out-Null
    Add-Label $slide "Macro-F1 +2.83 个百分点" 74 363 190 $C.PaleGreen $C.Teal | Out-Null
    Add-Rect $slide 492 126 420 292 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "行业识别与同源去重" 518 150 290 22 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bar $slide "行业事件 Macro-F1" 96.82 100 518 205 140 $C.Teal "96.82%"
    Add-Bar $slide "行业事件 Accuracy" 97.52 100 518 247 140 $C.Blue "97.52%"
    Add-Bar $slide "去重 Pair F1" 91.21 100 518 289 140 $C.Orange "91.21%"
    Add-Bar $slide "去重 B3 F1" 97.94 100 518 331 140 $C.Green "97.94%"
    Add-Label $slide "完整指标与误差分析已写入评测报告" 48 463 332 $C.PaleBlue $C.Blue | Out-Null

    # 12 risks and errors
    $slide = New-ContentSlide $presentation 12 "误差分析、局限与下一轮优化" "客观说明当前边界"
    Add-Rect $slide 48 126 410 294 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "当前误差集中点" 74 150 250 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("长尾类别样本少：海外产能进展、新车出海、国际化布局等 F1 较低", "相近语义混淆：公司战略调整 ↔ 产品技术创新", "关键技术突破 ↔ 产能布局调整 / 新业务拓展", "去重策略偏查准率：Pair Recall 为 85.79%") 74 198 340 36 12 $C.Ink
    Add-Rect $slide 492 126 420 294 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "可执行的优化路径" 518 150 250 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Bullets $slide @("补充相近类别的对比样本与人工标注示例", "对去重边界样本引入语义重排序，保持高 Precision", "采集事件簇级人工审核标签，校准可信度与告警规则", "引入金融领域预训练模型前，先建立不泄漏的对照实验") 518 198 340 36 12 $C.Ink
    Add-Label $slide "结论：现有结果可用，但不将无真值规则模块误报为【准确率】" 48 462 560 $C.PaleOrange $C.Orange | Out-Null

    # 13 demo
    $slide = New-ContentSlide $presentation 13 "在线演示与全类别案例覆盖" "交付可见、可查、可讲"
    Add-Rect $slide 48 126 562 310 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Rect $slide 48 126 562 39 $C.Navy $C.Navy $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "金融事件雷达 · 本地看板" 68 138 270 18 13 $C.White $msoTrue $ppAlignLeft | Out-Null
    Add-Rect $slide 66 184 225 51 $C.PaleRed $C.PaleRed $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "P3 即时推送  4" 82 196 185 17 13 $C.Red $msoTrue $ppAlignLeft | Out-Null
    Add-Rect $slide 66 247 225 51 $C.PaleOrange $C.PaleOrange $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "P2 优先推送  49" 82 259 185 17 13 $C.Orange $msoTrue $ppAlignLeft | Out-Null
    Add-Rect $slide 66 310 225 51 $C.PaleBlue $C.PaleBlue $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "P1 观察 / 复核  228" 82 322 190 17 13 $C.Blue $msoTrue $ppAlignLeft | Out-Null
    Add-Rect $slide 314 184 270 206 $C.Light $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "鼎龙股份：CMP 抛光液产品取得重大进展" 334 205 225 38 14 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Label $slide "21 篇报道" 334 257 88 $C.PaleBlue $C.Blue | Out-Null
    Add-Label $slide "12 个来源" 431 257 88 $C.PaleGreen $C.Teal | Out-Null
    Add-Text $slide "多源聚合 → 官方确认 → 证据链 → 预警理由" 334 309 215 38 11.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    Add-Text $slide "现场演示：搜索、按优先级/状态筛选、查看评分构成与完整时间线" 66 406 485 15 10 $C.Muted $msoFalse $ppAlignLeft | Out-Null
    Add-Rect $slide 638 126 274 310 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
    Add-Text $slide "演示案例库" 664 151 170 22 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    Add-Card $slide 664 193 220 72 "公司事件类别" "44 / 44" "43 条留出集 + 1 条补充" $C.Blue
    Add-Card $slide 664 278 220 72 "行业事件类别" "20 / 20" "全部为留出集正确预测" $C.Teal
    Add-Text $slide "演示路线覆盖：识别、去重、可信度、5 类时间线状态、P0–P3 与人工复核" 664 370 205 45 10.5 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    Add-Label $slide "http://localhost:8080/web/" 48 466 248 $C.PaleGreen $C.Teal | Out-Null

    # 14 conclusion
    $slide = New-ContentSlide $presentation 14 "总结：以评测驱动的金融事件闭环" "项目价值与答辩结论"
    Add-Text $slide "本项目已形成从多源资讯到可行动预警的可复现闭环。" 48 120 720 30 18 $C.Navy $msoTrue $ppAlignLeft | Out-Null
    $summaryCards = @(
        @{t="完整性"; d="覆盖识别、可信评估、去重、追踪与预警五个赛题能力"; c=$C.Blue},
        @{t="有效性"; d="语义融合使个股事件 Macro-F1 提升至 80.09%"; c=$C.Teal},
        @{t="严谨性"; d="训练验证隔离；无真值模块只报告覆盖，不虚构指标"; c=$C.Orange},
        @{t="可演示性"; d="429 个事件簇、看板、全类型案例库与逐页讲稿"; c=$C.Red}
    )
    for ($i = 0; $i -lt $summaryCards.Count; $i++) {
        $x = 48 + (($i % 2) * 430)
        $y = 183 + ([Math]::Floor($i / 2) * 116)
        $item = $summaryCards[$i]
        Add-Rect $slide $x $y 404 92 $C.White $C.Line $msoShapeRoundedRectangle | Out-Null
        Add-Rect $slide ($x + 18) ($y + 21) 48 48 $item.c $item.c $msoShapeOval | Out-Null
        Add-Text $slide ("{0}" -f ($i + 1)) ($x + 18) ($y + 35) 48 15 12 $C.White $msoTrue $ppAlignCenter | Out-Null
        Add-Text $slide $item.t ($x + 84) ($y + 19) 160 22 16 $C.Navy $msoTrue $ppAlignLeft | Out-Null
        Add-Text $slide $item.d ($x + 84) ($y + 46) 290 29 11 $C.Ink $msoFalse $ppAlignLeft | Out-Null
    }
    Add-Label $slide "交付物：PPT · 评测报告 · 运行 README · 演示案例库 · 本地看板" 48 442 450 $C.PaleBlue $C.Blue | Out-Null
    Add-Text $slide "谢谢各位老师，欢迎提问。" 48 486 480 24 17 $C.Navy $msoTrue $ppAlignLeft | Out-Null

    if (Test-Path $OutputPath) { Remove-Item -LiteralPath $OutputPath -Force }
    $presentation.SaveAs($OutputPath, $ppSaveAsOpenXMLPresentation)
    $presentation.Close()
    $app.Quit()
    Write-Output "PPT generated: $OutputPath"
}
finally {
    if ($presentation -ne $null) { try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) } catch {} }
    if ($app -ne $null) { try { [void][Runtime.InteropServices.Marshal]::ReleaseComObject($app) } catch {} }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
