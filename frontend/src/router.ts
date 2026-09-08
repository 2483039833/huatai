import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  { path: '/', name: 'Dashboard', component: () => import('./views/Dashboard.vue') },
  { path: '/events', name: 'Events', component: () => import('./views/Events.vue') },
  { path: '/timeline', name: 'Timeline', component: () => import('./views/Timeline.vue') },
  { path: '/graph', name: 'Graph', component: () => import('./views/Graph.vue') },
  { path: '/alerts', name: 'Alerts', component: () => import('./views/Alerts.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
