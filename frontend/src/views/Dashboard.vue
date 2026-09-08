<template>
  <div class="dashboard page-grid">
    <section class="kpi-grid">
      <div class="kpi" v-for="item in kpis" :key="item.label">
        <div class="kpi-label">{{ item.label }}</div>
        <div class="kpi-value">{{ item.value }}</div>
        <div class="kpi-note">{{ item.note }}</div>
      </div>
    </section>

    <el-row :gutter="18">
      <el-col :xs="24" :lg="15">
        <el-card shadow="never">
          <template #header>
            <div class="section-head">
              <span class="panel-title">事件类型分布</span>
              <el-tag size="small" effect="plain">Top 10</el-tag>
            </div>
          </template>
          <v-chart :option="typeChart" class="chart" autoresize />
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="9">
        <el-card shadow="never">
          <template #header>
            <div class="section-head">
              <span class="panel-title">预警分布</span>
              <el-button :icon="Refresh" text @click="load" />
            </div>
          </template>
          <v-chart :option="alertChart" class="chart" autoresize />
        </el-card>
      </el-col>
    </el-row>

    <el-row :gutter="18">
      <el-col :xs="24" :lg="14">
        <el-card shadow="never">
          <template #header>
            <span class="panel-title">高风险事件</span>
          </template>
          <div class="risk-list" v-loading="loading">
            <div v-for="item in topAlerts" :key="item.group_id" class="risk-item">
              <div class="risk-main">
                <el-tag :type="alertType(item.alert_level)" size="small">{{ alertLabel(item.alert_level) }}</el-tag>
                <span class="risk-title">{{ item.merged_title || '未命名事件' }}</span>
              </div>
              <div class="risk-meta">
                <span>{{ item.event_name || '未知类型' }}</span>
                <span>报道 {{ item.article_count || 0 }} 篇</span>
                <span>可信度 {{ pct(item.credibility_score) }}</span>
                <span>影响力 {{ pct(item.impact_score) }}</span>
              </div>
              <div class="risk-reason">{{ item.alert_reason || item.cluster_reason }}</div>
            </div>
            <el-empty v-if="!topAlerts.length && !loading" description="暂无活跃预警" />
          </div>
        </el-card>
      </el-col>
      <el-col :xs="24" :lg="10">
        <el-card shadow="never">
          <template #header>
            <span class="panel-title">系统能力覆盖</span>
          </template>
          <div class="capability-list">
            <div class="capability" v-for="c in capabilities" :key="c.title">
              <el-icon><component :is="c.icon" /></el-icon>
              <div>
                <strong>{{ c.title }}</strong>
                <p>{{ c.desc }}</p>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart, PieChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Connection, DataAnalysis, Refresh, Stopwatch, Warning } from '@element-plus/icons-vue'
import { fetchAlerts, fetchAlertStats, fetchEvents, fetchTimelineStats } from '../api'

use([BarChart, PieChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

const loading = ref(false)
const totalEvents = ref(0)
const totalCompanies = ref(0)
const totalTypes = ref(0)
const alertDist = ref<Record<string, number>>({ P0: 0, P1: 0, P2: 0, P3: 0 })
const topAlerts = ref<any[]>([])
const typeChart = ref<any>({})
const alertChart = ref<any>({})

const activeAlerts = computed(() => (alertDist.value.P0 || 0) + (alertDist.value.P1 || 0) + (alertDist.value.P2 || 0))
const kpis = computed(() => [
  { label: '聚合事件组', value: totalEvents.value, note: '由多篇新闻归并后的事件单元' },
  { label: '覆盖公司', value: totalCompanies.value, note: '按事件组关联标的统计' },
  { label: '活跃预警', value: activeAlerts.value, note: 'P0 / P1 / P2 活跃预警' },
  { label: '事件类型', value: totalTypes.value, note: '覆盖公司、行业、产业链事件' },
])

const capabilities = [
  { title: '同源聚合', desc: 'BGE 语义向量、FAISS 召回、相似度图归并。', icon: Connection },
  { title: '解释证据', desc: '代表文章、相似度证据边、组内置信度。', icon: DataAnalysis },
  { title: '脉络追踪', desc: '首发、跟进、官方确认、澄清与影响更新。', icon: Stopwatch },
  { title: '分层预警', desc: '可信度、影响力、来源多样性共同触发。', icon: Warning },
]

function alertType(level: string) {
  return { P0: 'danger', P1: 'warning', P2: 'warning', P3: 'info' }[level] || 'info'
}
function alertLabel(level: string) {
  return { P3: 'P3', P2: 'P2', P1: 'P1', P0: 'P0' }[level] || level || '未知'
}
function pct(value: number) {
  return value == null ? '-' : `${Math.round(value * 100)}%`
}

async function load() {
  loading.value = true
  try {
    const [ev, aStat, tStat, all] = await Promise.all([
      fetchEvents({ limit: 1 }),
      fetchAlertStats(),
      fetchTimelineStats(),
      fetchAlerts({ limit: 500 }),
    ])
    const alerts = all.alerts || []
    const companies = new Set<string>()
    alerts.forEach((a: any) => (a.trading_codes || []).forEach((c: string) => companies.add(c)))

    totalEvents.value = ev.total || 0
    totalCompanies.value = companies.size
    totalTypes.value = Object.keys(tStat.by_event_type || {}).length
    alertDist.value = { P0: 0, P1: 0, P2: 0, P3: 0, ...(aStat.distribution || {}) }
    topAlerts.value = alerts
      .filter((a: any) => a.alert_level && a.alert_level !== 'P3')
      .sort((a: any, b: any) => (b.impact_score || 0) - (a.impact_score || 0))
      .slice(0, 6)

    const entries = Object.entries(tStat.by_event_type || {})
      .sort((a: any, b: any) => b[1] - a[1])
      .slice(0, 10)
    typeChart.value = {
      tooltip: { trigger: 'axis' },
      grid: { left: 118, right: 24, top: 12, bottom: 28 },
      xAxis: { type: 'value', splitLine: { lineStyle: { color: '#edf1f6' } } },
      yAxis: { type: 'category', data: entries.map((e: any) => e[0]), inverse: true, axisLabel: { width: 100, overflow: 'truncate' } },
      series: [{ type: 'bar', data: entries.map((e: any) => e[1]), barWidth: 14, itemStyle: { color: '#2f80ed', borderRadius: 3 } }],
    }
    alertChart.value = {
      tooltip: { trigger: 'item' },
      legend: { bottom: 0, itemWidth: 10, itemHeight: 10 },
      series: [{
        type: 'pie',
        radius: ['52%', '74%'],
        center: ['50%', '44%'],
        label: { formatter: '{b}: {c}' },
        data: [
          { name: 'P0 立即推送', value: alertDist.value.P0 || 0, itemStyle: { color: '#d94b4b' } },
          { name: 'P1 优先推送', value: alertDist.value.P1 || 0, itemStyle: { color: '#e89b3c' } },
          { name: 'P2 观察列表', value: alertDist.value.P2 || 0, itemStyle: { color: '#d8bc2e' } },
          { name: 'P3 归档', value: alertDist.value.P3 || 0, itemStyle: { color: '#8a96a6' } },
        ],
      }],
    }
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<style scoped>
.page-grid { display: grid; gap: 18px; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.kpi { background: #fff; border: 1px solid #dfe5ec; border-radius: 8px; padding: 16px; }
.kpi-label { font-size: 13px; color: #687589; }
.kpi-value { margin-top: 8px; font-size: 31px; font-weight: 800; color: #202734; }
.kpi-note { margin-top: 5px; font-size: 12px; color: #8a96a6; }
.section-head { display: flex; justify-content: space-between; align-items: center; }
.chart { height: 344px; }
.risk-list { display: grid; gap: 12px; min-height: 280px; }
.risk-item { border: 1px solid #e3e8ef; border-radius: 8px; padding: 12px; background: #fbfcfe; }
.risk-main { display: flex; align-items: center; gap: 8px; min-width: 0; }
.risk-title { font-weight: 700; color: #263241; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.risk-meta { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; font-size: 12px; color: #687589; }
.risk-reason { margin-top: 8px; color: #596579; font-size: 13px; line-height: 1.5; }
.capability-list { display: grid; gap: 14px; }
.capability { display: flex; gap: 12px; align-items: flex-start; padding: 12px; border: 1px solid #e3e8ef; border-radius: 8px; }
.capability .el-icon { width: 32px; height: 32px; border-radius: 6px; background: #eaf4f1; color: #2c8a71; font-size: 18px; }
.capability strong { color: #263241; }
.capability p { margin: 5px 0 0; color: #687589; line-height: 1.5; font-size: 13px; }
@media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 640px) { .kpi-grid { grid-template-columns: 1fr; } }
</style>

