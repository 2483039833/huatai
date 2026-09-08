<template>
  <el-container class="app-shell">
    <el-aside width="232px" class="sidebar">
      <div class="brand">
        <div class="brand-mark">HT</div>
        <div>
          <div class="brand-title">金融事件雷达</div>
          <div class="brand-subtitle">Event Intelligence</div>
        </div>
      </div>

      <el-menu router :default-active="route.path" class="nav-menu">
        <el-menu-item index="/">
          <el-icon><DataBoard /></el-icon>
          <span>总览看板</span>
        </el-menu-item>
        <el-menu-item index="/alerts">
          <el-icon><Warning /></el-icon>
          <span>预警中心</span>
        </el-menu-item>
        <el-menu-item index="/events">
          <el-icon><Files /></el-icon>
          <span>事件聚合</span>
        </el-menu-item>
        <el-menu-item index="/timeline">
          <el-icon><Clock /></el-icon>
          <span>脉络追踪</span>
        </el-menu-item>
        <el-menu-item index="/graph">
          <el-icon><Share /></el-icon>
          <span>关系图谱</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header class="topbar">
        <div>
          <h1>{{ pageTitle }}</h1>
          <p>同源聚合、增量追踪、可信预警的一体化工作台</p>
        </div>
        <div class="topbar-actions">
          <el-tag effect="plain" type="success">API /api/v1</el-tag>
          <el-tag effect="plain">BGE + FAISS</el-tag>
        </div>
      </el-header>
      <el-main class="main-panel">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import { Clock, DataBoard, Files, Share, Warning } from '@element-plus/icons-vue'

const route = useRoute()
const titles: Record<string, string> = {
  '/': '总览看板',
  '/events': '事件聚合',
  '/timeline': '脉络追踪',
  '/graph': '关系图谱',
  '/alerts': '预警中心',
}
const pageTitle = computed(() => titles[route.path] || '金融事件雷达')
</script>

<style>
:root {
  color: #202734;
  background: #eef1f5;
  font-family: Inter, "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
}

* { box-sizing: border-box; }
body { margin: 0; background: #eef1f5; }

.app-shell { min-height: 100vh; background: #eef1f5; }
.sidebar {
  background: #17202b;
  border-right: 1px solid #263241;
  color: #fff;
}
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  height: 72px;
  padding: 0 18px;
  border-bottom: 1px solid rgba(255,255,255,.08);
}
.brand-mark {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border-radius: 8px;
  background: #2fb28f;
  color: #fff;
  font-weight: 800;
}
.brand-title { font-size: 17px; font-weight: 700; letter-spacing: 0; }
.brand-subtitle { margin-top: 3px; font-size: 12px; color: #97a4b5; }
.nav-menu { border-right: 0; background: transparent; padding: 12px; }
.nav-menu .el-menu-item {
  height: 44px;
  margin-bottom: 6px;
  border-radius: 6px;
  color: #bfccd9;
}
.nav-menu .el-menu-item:hover,
.nav-menu .el-menu-item.is-active {
  background: #243142;
  color: #fff;
}
.topbar {
  height: 72px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 28px;
  background: #fff;
  border-bottom: 1px solid #dde3ea;
}
.topbar h1 { margin: 0; font-size: 22px; font-weight: 760; color: #202734; }
.topbar p { margin: 6px 0 0; color: #687589; font-size: 13px; }
.topbar-actions { display: flex; align-items: center; gap: 8px; }
.main-panel { padding: 22px 28px 30px; }
.el-card { border-radius: 8px; border-color: #dfe5ec; }
.el-card__header { padding: 13px 16px; font-weight: 700; color: #263241; }
.panel-title { font-size: 15px; font-weight: 750; color: #263241; }
.muted { color: #7b8797; }
.score-cell { display: flex; align-items: center; gap: 8px; }
.table-title { font-weight: 650; color: #263241; }
.tag-row { display: flex; flex-wrap: wrap; gap: 5px; }
@media (max-width: 900px) {
  .sidebar { display: none; }
  .topbar { height: auto; min-height: 72px; align-items: flex-start; flex-direction: column; gap: 10px; padding: 16px; }
  .main-panel { padding: 16px; }
}
</style>
