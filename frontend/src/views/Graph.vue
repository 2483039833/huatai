<template>
  <div class="graph-page">
    <el-card shadow="never">
      <template #header>
        <div class="toolbar">
          <div>
            <span class="panel-title">事件关系图谱</span>
            <span class="muted total-text">{{ summary }}</span>
          </div>
          <div class="actions">
            <el-checkbox v-model="onlyAlerted" @change="load">仅预警事件</el-checkbox>
            <el-button :icon="Refresh" @click="load" />
          </div>
        </div>
      </template>
      <v-chart :option="chart" class="graph-chart" autoresize />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { GraphChart } from 'echarts/charts'
import { LegendComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Refresh } from '@element-plus/icons-vue'
import { fetchGraph } from '../api'

use([GraphChart, TooltipComponent, LegendComponent, CanvasRenderer])

const onlyAlerted = ref(true)
const chart = ref<any>({})
const summary = ref('')

const palette: Record<string, string> = {
  event: '#2f80ed',
  company: '#2fa67a',
  event_type: '#9a6dd7',
  source: '#e89b3c',
}
const alertColor: Record<string, string> = {
  P0: '#d94b4b',
  P1: '#e89b3c',
  P2: '#d8bc2e',
}

function nodeSize(n: any) {
  if (n.type === 'company') return 42
  if (n.type === 'event_type') return 34
  if (n.type === 'source') return 24
  return Math.max(28, Math.min(58, 28 + (n.article_count || 0) * 2))
}
function nodeColor(n: any) {
  if (n.type === 'event' && n.alert_level && n.alert_level !== 'P3') return alertColor[n.alert_level] || palette.event
  return palette[n.type] || '#8a96a6'
}

async function load() {
  const res = await fetchGraph({ only_alerted: onlyAlerted.value ? 'true' : 'false', limit: 180 })
  summary.value = `${res.total_events || 0} 个事件 · ${res.total_nodes || 0} 个节点 · ${res.total_edges || 0} 条关系`
  const nodes = (res.nodes || []).map((n: any) => ({
    ...n,
    id: String(n.id),
    name: n.label || n.id,
    category: n.type || 'event',
    symbolSize: nodeSize(n),
    itemStyle: { color: nodeColor(n), borderColor: '#fff', borderWidth: 1 },
    label: { show: n.type !== 'source', fontSize: 10, overflow: 'truncate', width: 100 },
  }))
  const links = (res.edges || []).map((e: any) => ({
    source: String(e.source),
    target: String(e.target),
    value: e.relation || e.event_name || '',
    lineStyle: { opacity: 0.36, width: Math.max(1, Math.min(4, Number(e.weight || 1))) },
  }))
  const types = [...new Set(nodes.map((n: any) => n.category))]
  chart.value = {
    tooltip: {
      trigger: 'item',
      formatter: (p: any) => {
        const d = p.data || {}
        if (p.dataType === 'edge') return `${p.data.source} → ${p.data.target}<br/>${p.data.value || '关联'}`
        return `${d.name}<br/>类型：${d.category}<br/>预警：${d.alert_level || '-'}<br/>报道：${d.article_count || '-'}`
      },
    },
    legend: { top: 0, data: types, formatter: (name: string) => ({ event: '事件', company: '公司', event_type: '事件类型', source: '来源' }[name] || name) },
    series: [{
      type: 'graph',
      layout: 'force',
      top: 36,
      data: nodes,
      links,
      categories: types.map((t: any) => ({ name: t })),
      roam: true,
      draggable: true,
      force: { repulsion: 210, edgeLength: [46, 110], gravity: 0.08 },
      label: { show: true, color: '#263241' },
      lineStyle: { color: '#9aa6b2', curveness: 0.08 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 3, opacity: 0.8 } },
    }],
  }
}

onMounted(load)
</script>

<style scoped>
.toolbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.total-text { margin-left: 10px; font-size: 13px; font-weight: 400; }
.actions { display: flex; align-items: center; gap: 10px; }
.graph-chart { height: calc(100vh - 176px); min-height: 560px; }
@media (max-width: 760px) { .toolbar { align-items: flex-start; flex-direction: column; } .graph-chart { height: 620px; } }
</style>
