<template>
  <div class="timeline-page">
    <el-card shadow="never">
      <template #header>
        <div class="toolbar">
          <span class="panel-title">事件脉络追踪</span>
          <div class="filters">
            <el-select v-model="company" filterable clearable placeholder="按公司筛选" size="small" @change="load">
              <el-option v-for="c in companies" :key="c" :label="c" :value="c" />
            </el-select>
            <el-select v-model="eventType" filterable clearable placeholder="按事件类型筛选" size="small" @change="load">
              <el-option v-for="t in eventTypes" :key="t" :label="t" :value="t" />
            </el-select>
            <el-button :icon="Refresh" @click="load" />
          </div>
        </div>
      </template>

      <el-timeline v-if="items.length" class="timeline-list">
        <el-timeline-item v-for="item in items" :key="item.group_id" :timestamp="item.time || '-'" placement="top" :type="timelineType(item.alert_level)">
          <div class="event-block">
            <div class="event-head">
              <div>
                <div class="event-title">{{ item.title || '未命名事件' }}</div>
                <div class="event-meta">
                  <el-tag v-if="item.event_name" size="small" effect="plain">{{ item.event_name }}</el-tag>
                  <span>报道 {{ item.article_count || 0 }} 篇</span>
                  <span>可信度 {{ pct(item.credibility_score) }}</span>
                  <span>影响力 {{ pct(item.impact_score) }}</span>
                </div>
              </div>
              <el-tag :type="alertType(item.alert_level)">{{ alertLabel(item.alert_level) }}</el-tag>
            </div>
            <p v-if="item.summary" class="summary">{{ item.summary }}</p>
            <div class="tag-row"><el-tag v-for="c in item.trading_codes || []" :key="c" size="small" type="info">{{ c }}</el-tag></div>

            <div v-if="item.timeline_stages && item.timeline_stages.length" class="stage-strip">
              <div v-for="stage in item.timeline_stages.slice(0, 6)" :key="stage.article_id" class="stage-card">
                <el-tag size="small" :type="stageType(stage.stage)">{{ stageLabel(stage.stage) }}</el-tag>
                <div class="stage-title">{{ stage.title || stage.article_id }}</div>
                <div class="stage-meta">{{ stage.source || '-' }}</div>
              </div>
            </div>
          </div>
        </el-timeline-item>
      </el-timeline>
      <el-empty v-else description="暂无事件" />
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Refresh } from '@element-plus/icons-vue'
import { fetchTimeline, fetchTimelineStats } from '../api'

const items = ref<any[]>([])
const company = ref('')
const eventType = ref('')
const companies = ref<string[]>([])
const eventTypes = ref<string[]>([])

function alertType(l: string) { return { P0: 'danger', P1: 'warning', P2: 'warning', P3: 'info' }[l] || 'info' }
function alertLabel(l: string) { return { P0: 'P0 立即推送', P1: 'P1 优先推送', P2: 'P2 观察列表', P3: 'P3 归档' }[l] || l || '未知' }
function timelineType(l: string) { return { P0: 'danger', P1: 'warning', P2: 'warning', P3: 'primary' }[l] || 'primary' }
function stageType(stage: string) { return { official_confirmation: 'success', clarification: 'danger', impact_update: 'warning', initial_report: 'primary', follow_up: 'info' }[stage] || 'info' }
function stageLabel(stage: string) {
  return { initial_report: '首发', follow_up: '跟进', official_confirmation: '官方确认', clarification: '澄清', impact_update: '影响更新' }[stage] || stage || '未知'
}
function pct(value: number) { return value == null ? '-' : `${Math.round(value * 100)}%` }

async function load() {
  const params: any = { limit: 200 }
  if (company.value) params.company = company.value
  if (eventType.value) params.event_type = eventType.value
  const res = await fetchTimeline(params)
  items.value = res.timeline || []
}

onMounted(async () => {
  const stats = await fetchTimelineStats()
  companies.value = Object.keys(stats.by_company || {})
  eventTypes.value = Object.keys(stats.by_event_type || {})
  load()
})
</script>

<style scoped>
.toolbar { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.filters { display: flex; gap: 10px; align-items: center; }
.filters .el-select { width: 190px; }
.timeline-list { padding: 10px 8px; }
.event-block { border: 1px solid #e3e8ef; border-radius: 8px; padding: 14px; background: #fff; }
.event-head { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; }
.event-title { font-size: 15px; font-weight: 750; color: #263241; line-height: 1.5; }
.event-meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-top: 7px; color: #687589; font-size: 12px; }
.summary { margin: 10px 0; color: #596579; line-height: 1.65; }
.stage-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin-top: 12px; }
.stage-card { border: 1px solid #e3e8ef; border-radius: 8px; padding: 10px; background: #fbfcfe; min-width: 0; }
.stage-title { margin-top: 7px; color: #263241; font-size: 13px; line-height: 1.45; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.stage-meta { margin-top: 5px; color: #7b8797; font-size: 12px; }
@media (max-width: 760px) { .toolbar, .filters { align-items: stretch; flex-direction: column; } .filters .el-select { width: 100%; } }
</style>
