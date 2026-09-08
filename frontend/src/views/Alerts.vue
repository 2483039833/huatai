<template>
  <div class="alerts-page">
    <section class="alert-kpis">
      <div class="alert-kpi p0"><span>P0 立即推送</span><strong>{{ stat.P0 || 0 }}</strong></div>
      <div class="alert-kpi p1"><span>P1 优先推送</span><strong>{{ stat.P1 || 0 }}</strong></div>
      <div class="alert-kpi p2"><span>P2 观察列表</span><strong>{{ stat.P2 || 0 }}</strong></div>
      <div class="alert-kpi p3"><span>P3 归档</span><strong>{{ stat.P3 || 0 }}</strong></div>
    </section>

    <el-row :gutter="18">
      <el-col :xs="24" :lg="17">
        <el-card shadow="never">
          <template #header>
            <div class="toolbar">
              <span class="panel-title">预警事件列表</span>
              <el-radio-group v-model="level" size="small" @change="reloadFromFirstPage">
                <el-radio-button label="all">全部</el-radio-button>
                <el-radio-button label="P0">P0</el-radio-button>
                <el-radio-button label="P1">P1</el-radio-button>
                <el-radio-button label="P2">P2</el-radio-button>
                <el-radio-button label="P3">P3</el-radio-button>
              </el-radio-group>
            </div>
          </template>

          <el-table :data="alerts" v-loading="loading" stripe height="calc(100vh - 310px)">
            <el-table-column label="级别" width="90">
              <template #default="{ row }"><el-tag :type="alertType(row.alert_level)">{{ alertLabel(row.alert_level) }}</el-tag></template>
            </el-table-column>
            <el-table-column label="事件" min-width="300">
              <template #default="{ row }">
                <div class="table-title">{{ row.merged_title || '未命名事件' }}</div>
                <div class="table-subtitle">{{ row.alert_reason || row.cluster_reason }}</div>
              </template>
            </el-table-column>
            <el-table-column prop="event_name" label="类型" width="130" show-overflow-tooltip />
            <el-table-column label="影响力" width="130">
              <template #default="{ row }"><el-progress :percentage="percent(row.impact_score)" :color="scoreColor(row.impact_score)" /></template>
            </el-table-column>
            <el-table-column label="可信度" width="130">
              <template #default="{ row }"><el-progress :percentage="percent(row.credibility_score)" :color="scoreColor(row.credibility_score)" /></template>
            </el-table-column>
            <el-table-column prop="article_count" label="报道" width="76" />
            <el-table-column label="公司" min-width="140">
              <template #default="{ row }"><div class="tag-row"><el-tag v-for="c in (row.trading_codes || []).slice(0, 3)" :key="c" size="small" effect="plain">{{ c }}</el-tag></div></template>
            </el-table-column>
            <el-table-column label="操作" width="86">
              <template #default="{ row }"><el-button :icon="View" text @click="open(row)">详情</el-button></template>
            </el-table-column>
          </el-table>

          <el-pagination layout="prev, pager, next, total" :total="total" :page-size="pageSize" :current-page="page" @current-change="onPage" class="pager" />
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="7">
        <el-card shadow="never">
          <template #header><span class="panel-title">当前阈值</span></template>
          <div class="rule-list">
            <div v-for="name in ['P0', 'P1', 'P2']" :key="name" class="rule-item">
              <el-tag :type="alertType(name)">{{ alertLabel(name) }}</el-tag>
              <span>综合分 ≥ {{ thresholdText(name, 'min_impact') }}</span>
              <span>{{ actionLabel(name) }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-drawer v-model="drawer" size="44%" :title="current?.merged_title || '预警详情'">
      <div v-if="current" class="detail-stack">
        <el-alert :title="current.alert_reason || '暂无预警原因'" :type="alertType(current.alert_level)" show-icon :closable="false" />
        <el-descriptions :column="2" border>
          <el-descriptions-item label="预警级别">{{ alertLabel(current.alert_level) }}</el-descriptions-item>
          <el-descriptions-item label="影响等级">{{ current.impact_level || '-' }}</el-descriptions-item>
          <el-descriptions-item label="可信度">{{ percent(current.credibility_score) }}%</el-descriptions-item>
          <el-descriptions-item label="影响力">{{ percent(current.impact_score) }}%</el-descriptions-item>
          <el-descriptions-item label="情绪方向">{{ current.alert_factors?.sentiment_label || '-' }}</el-descriptions-item>
          <el-descriptions-item label="聚类置信度">{{ percent(current.cluster_confidence) }}%</el-descriptions-item>
          <el-descriptions-item label="来源多样性">{{ percent(current.source_diversity) }}%</el-descriptions-item>
          <el-descriptions-item label="事件组" :span="2">{{ current.group_id }} · {{ current.article_count || 0 }} 篇报道</el-descriptions-item>
        </el-descriptions>
        <div>
          <h3>解释因子</h3>
          <div class="factor-grid">
            <div v-for="(value, key) in current.alert_factors || {}" :key="key" class="factor"><span>{{ factorLabel(String(key)) }}</span><strong>{{ factorValue(value) }}</strong></div>
          </div>
        </div>
        <div>
          <h3>聚类证据</h3>
          <p class="muted">{{ current.cluster_reason || '暂无聚类解释' }}</p>
        </div>
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { View } from '@element-plus/icons-vue'
import { fetchAlertConfig, fetchAlerts, fetchAlertStats } from '../api'

const alerts = ref<any[]>([])
const stat = ref<Record<string, number>>({ P0: 0, P1: 0, P2: 0, P3: 0 })
const config = ref<any>({})
const level = ref('all')
const page = ref(1)
const pageSize = 20
const total = ref(0)
const loading = ref(false)
const drawer = ref(false)
const current = ref<any>(null)

function alertType(l: string) { return { P0: 'danger', P1: 'warning', P2: 'warning', P3: 'info' }[l] || 'info' }
function alertLabel(l: string) { return { P0: 'P0 立即推送', P1: 'P1 优先推送', P2: 'P2 观察列表', P3: 'P3 归档' }[l] || l || '未知' }
function actionLabel(l: string) { return { P0: '立即推送', P1: '优先推送', P2: '观察列表', P3: '归档' }[l] || '-' }
function percent(v: number) { return Math.max(0, Math.min(100, Math.round((v || 0) * 100))) }
function scoreColor(s: number) { return (s || 0) >= 0.8 ? '#d94b4b' : (s || 0) >= 0.6 ? '#e89b3c' : '#2f80ed' }
function thresholdText(name: string, key: string) { return config.value?.[name]?.[key] == null ? '-' : `${Math.round(config.value[name][key] * 100)}%` }
function factorLabel(key: string) {
  return {
    credibility_score: '可信度',
    impact_score: '影响力',
    alert_score: '综合预警分',
    alert_policy: '预警策略',
    sentiment_score: '情绪分',
    sentiment_label: '情绪方向',
    article_count: '报道数',
    source_diversity: '来源多样性',
    cluster_confidence: '聚类置信度',
    impact_level: '影响等级',
    b_alert_level_hint: '预警建议',
    b_classification_event_name: '事件分类',
    b_classification_scope: '分类范围',
    b_classification_classifier: '分类模型',
    b_classification_confidence: '分类置信度',
    b_classification_margin: '分类边际',
    b_classification_review_required: '是否需复核',
    b_credibility_source_authority: '来源权威性',
    b_credibility_cross_source_corroboration: '多源印证',
    b_credibility_official_evidence: '官方证据',
    b_credibility_content_specificity: '内容具体性',
    b_credibility_cluster_agreement: '聚类一致性',
    b_credibility_rumour_penalty: '传闻惩罚',
    b_impact_evidence_confidence: '证据可信度',
    b_impact_event_materiality: '事件重要性',
    b_impact_information_breadth: '信息广度',
    b_impact_state_confirmation: '状态确认',
    b_impact_freshness: '新鲜度',
  }[key] || key
}
function factorValue(value: any) {
  if (value === true) return '是'
  if (value === false) return '否'
  if (value === 'b_issuer_title_semantic_hybrid') return '标题语义融合分类'
  if (value === 'b_issuer_title_character_nn') return '标题近邻分类'
  if (value === 'b_body_character_nb') return '正文字符朴素贝叶斯'
  if (value === 'company') return '公司事件'
  if (value === 'industry') return '行业事件'
  if (value === 'b_style_comprehensive_score') return '综合评分'
  if (typeof value === 'number') return value <= 1 ? `${Math.round(value * 100)}%` : value.toFixed(2)
  return value == null || value === '' ? '-' : String(value)
}

async function load() {
  loading.value = true
  try {
    const params: any = { limit: pageSize, offset: (page.value - 1) * pageSize }
    if (level.value !== 'all') params.level = level.value
    const res = await fetchAlerts(params)
    alerts.value = res.alerts || []
    total.value = res.total || 0
  } finally {
    loading.value = false
  }
}
function reloadFromFirstPage() { page.value = 1; load() }
function onPage(p: number) { page.value = p; load() }
function open(row: any) { current.value = row; drawer.value = true }

async function loadStats() {
  try {
    const s = await fetchAlertStats()
    stat.value = { P0: 0, P1: 0, P2: 0, P3: 0, ...(s.distribution || {}) }
  } catch (e) {
    const res = await fetchAlerts({ limit: 500 })
    const dist: Record<string, number> = { P0: 0, P1: 0, P2: 0, P3: 0 }
    ;(res.alerts || []).forEach((item: any) => {
      const key = item.alert_level || 'P3'
      dist[key] = (dist[key] || 0) + 1
    })
    stat.value = dist
  }
}

async function loadConfig() {
  try {
    config.value = await fetchAlertConfig()
  } catch (e) {
    config.value = {
      P0: { min_impact: 0.8, min_credibility: 0.8 },
      P1: { min_impact: 0.65, min_credibility: 0.7 },
      P2: { min_impact: 0.52, min_credibility: 0.55 },
      P3: { min_impact: 0, min_credibility: 0 },
    }
  }
}

onMounted(async () => {
  await Promise.all([loadStats(), loadConfig()])
  load()
})
</script>

<style scoped>
.alerts-page { display: grid; gap: 18px; }
.alert-kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.alert-kpi { border-radius: 8px; padding: 14px 16px; border: 1px solid #dfe5ec; background: #fff; }
.alert-kpi span { display: block; color: #687589; font-size: 13px; }
.alert-kpi strong { display: block; margin-top: 8px; font-size: 30px; color: #202734; }
.alert-kpi.p0 { border-left: 4px solid #d94b4b; }
.alert-kpi.p1 { border-left: 4px solid #e89b3c; }
.alert-kpi.p2 { border-left: 4px solid #d8bc2e; }
.alert-kpi.p3 { border-left: 4px solid #8a96a6; }
.toolbar { display: flex; justify-content: space-between; align-items: center; }
.table-subtitle { margin-top: 4px; font-size: 12px; color: #7b8797; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.pager { margin-top: 14px; justify-content: flex-end; }
.rule-list { display: grid; gap: 10px; }
.rule-item { display: grid; grid-template-columns: 72px 1fr; gap: 8px 10px; align-items: center; padding: 10px; border: 1px solid #e3e8ef; border-radius: 8px; color: #596579; font-size: 13px; }
.rule-item span:last-child { grid-column: 2; }
.detail-stack { display: grid; gap: 18px; }
.detail-stack h3 { margin: 0 0 8px; font-size: 15px; }
.factor-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
.factor { border: 1px solid #e3e8ef; border-radius: 8px; padding: 10px; }
.factor span { display: block; font-size: 12px; color: #7b8797; }
.factor strong { display: block; margin-top: 4px; color: #263241; }
@media (max-width: 900px) { .alert-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .alert-kpis, .factor-grid { grid-template-columns: 1fr; } }
</style>


