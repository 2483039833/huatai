<template>
  <div class="events-page">
    <el-card shadow="never">
      <template #header>
        <div class="toolbar">
          <div>
            <span class="panel-title">事件聚合结果</span>
            <span class="muted total-text">共 {{ total }} 个事件组</span>
          </div>
          <el-button :icon="Refresh" @click="load" />
        </div>
      </template>

      <el-table :data="events" v-loading="loading" stripe height="calc(100vh - 230px)">
        <el-table-column prop="group_id" label="组ID" width="104" fixed />
        <el-table-column label="事件标题" min-width="310" fixed>
          <template #default="{ row }">
            <div class="table-title">{{ row.merged_title || '未命名事件' }}</div>
            <div class="table-subtitle">{{ row.cluster_reason || '暂无聚类解释' }}</div>
          </template>
        </el-table-column>
        <el-table-column prop="event_name" label="事件类型" width="140" show-overflow-tooltip />
        <el-table-column label="聚类置信度" width="132">
          <template #default="{ row }">
            <div class="score-cell">
              <el-progress :percentage="percent(row.cluster_confidence)" :show-text="false" :stroke-width="7" />
              <span>{{ percent(row.cluster_confidence) }}%</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="article_count" label="报道数" width="86" />
        <el-table-column label="来源多样性" width="120">
          <template #default="{ row }">{{ percent(row.source_diversity) }}%</template>
        </el-table-column>
        <el-table-column label="证据边" width="90">
          <template #default="{ row }">{{ (row.evidence_edges || []).length }}</template>
        </el-table-column>
        <el-table-column label="涉及公司" min-width="160">
          <template #default="{ row }">
            <div class="tag-row">
              <el-tag v-for="c in (row.trading_codes || []).slice(0, 4)" :key="c" size="small" effect="plain">{{ c }}</el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="timeline_start" label="首发时间" width="170" show-overflow-tooltip />
        <el-table-column label="操作" width="92" fixed="right">
          <template #default="{ row }">
            <el-button :icon="View" text @click="openDetail(row.group_id)">查看</el-button>
          </template>
        </el-table-column>
      </el-table>

      <el-pagination
        layout="prev, pager, next, total"
        :total="total"
        :page-size="pageSize"
        :current-page="page"
        @current-change="onPage"
        class="pager"
      />
    </el-card>

    <el-drawer v-model="drawer" size="52%" :title="detail?.merged_title || '事件详情'">
      <div v-if="detail" class="detail-grid">
        <section class="detail-section">
          <h3>聚合解释</h3>
          <el-descriptions :column="2" border>
            <el-descriptions-item label="组ID">{{ detail.group_id }}</el-descriptions-item>
            <el-descriptions-item label="代表文章">{{ detail.representative_article_id || '-' }}</el-descriptions-item>
            <el-descriptions-item label="事件类型">{{ detail.event_name || '-' }}</el-descriptions-item>
            <el-descriptions-item label="报道数">{{ detail.article_count || 0 }}</el-descriptions-item>
            <el-descriptions-item label="聚类置信度">{{ percent(detail.cluster_confidence) }}%</el-descriptions-item>
            <el-descriptions-item label="来源多样性">{{ percent(detail.source_diversity) }}%</el-descriptions-item>
            <el-descriptions-item label="时间跨度" :span="2">{{ detail.timeline_start || '-' }} 至 {{ detail.timeline_end || '-' }}</el-descriptions-item>
            <el-descriptions-item label="聚类原因" :span="2">{{ detail.cluster_reason || '-' }}</el-descriptions-item>
          </el-descriptions>
        </section>

        <section class="detail-section">
          <h3>相似度证据</h3>
          <el-table :data="detail.evidence_edges || []" size="small" max-height="220" border>
            <el-table-column prop="source_article_id" label="文章A" min-width="150" />
            <el-table-column prop="target_article_id" label="文章B" min-width="150" />
            <el-table-column label="相似度" width="100">
              <template #default="{ row }">{{ percent(row.similarity) }}%</template>
            </el-table-column>
          </el-table>
        </section>

        <section class="detail-section">
          <h3>事件阶段</h3>
          <el-timeline>
            <el-timeline-item v-for="s in detail.timeline_stages || []" :key="s.article_id" :timestamp="s.time" placement="top">
              <strong>{{ stageLabel(s.stage) }}</strong>
              <div class="stage-title">{{ s.title }}</div>
              <div class="muted">{{ s.source }} · {{ s.reason }}</div>
            </el-timeline-item>
          </el-timeline>
        </section>

        <section class="detail-section">
          <h3>组内报道</h3>
          <div class="article-list">
            <div v-for="e in detail.events || []" :key="e.article_file_id" class="article-item">
              <div class="article-title">{{ e.article_title || e.article_file_id }}</div>
              <div class="muted">{{ e.article_source || '-' }} · {{ e.article_publish_time || '-' }} · {{ directionLabel(e.event_impact_direction) }}</div>
            </div>
          </div>
        </section>
      </div>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { Refresh, View } from '@element-plus/icons-vue'
import { fetchEventDetail, fetchEvents } from '../api'

const events = ref<any[]>([])
const total = ref(0)
const page = ref(1)
const pageSize = 20
const loading = ref(false)
const drawer = ref(false)
const detail = ref<any>(null)

function percent(value: number) {
  return Math.max(0, Math.min(100, Math.round((value || 0) * 100)))
}
function stageLabel(stage: string) {
  return {
    initial_report: '首发报道',
    follow_up: '后续跟进',
    official_confirmation: '官方确认',
    clarification: '澄清/辟谣',
    impact_update: '影响更新',
  }[stage] || stage || '未知阶段'
}
function directionLabel(value: string) {
  return { positive: '正面', negative: '负面', neutral: '中性', 正面: '正面', 负面: '负面', 中性: '中性' }[value] || value || '中性'
}
async function load() {
  loading.value = true
  try {
    const res = await fetchEvents({ limit: pageSize, offset: (page.value - 1) * pageSize })
    events.value = res.events || []
    total.value = res.total || 0
  } finally {
    loading.value = false
  }
}
function onPage(p: number) {
  page.value = p
  load()
}
async function openDetail(id: string) {
  detail.value = await fetchEventDetail(id)
  drawer.value = true
}
onMounted(load)
</script>

<style scoped>
.events-page { min-width: 0; }
.toolbar { display: flex; align-items: center; justify-content: space-between; }
.total-text { margin-left: 10px; font-size: 13px; font-weight: 400; }
.table-subtitle { margin-top: 4px; color: #7b8797; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.score-cell .el-progress { width: 70px; }
.pager { margin-top: 14px; justify-content: flex-end; }
.detail-grid { display: grid; gap: 20px; }
.detail-section h3 { margin: 0 0 10px; font-size: 15px; color: #263241; }
.stage-title { margin: 4px 0; color: #263241; }
.article-list { display: grid; gap: 10px; }
.article-item { border: 1px solid #e3e8ef; border-radius: 8px; padding: 10px 12px; }
.article-title { color: #263241; font-weight: 650; margin-bottom: 5px; }
</style>
