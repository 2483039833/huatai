import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 10000,
})

export async function fetchEvents(params?: any) {
  const res = await api.get('/events/', { params })
  return res.data
}

export async function fetchEventDetail(groupId: string) {
  const res = await api.get(`/events/${groupId}`)
  return res.data
}

export async function fetchTimeline(params?: any) {
  const res = await api.get('/timeline/', { params })
  return res.data
}

export async function fetchTimelineStats() {
  const res = await api.get('/timeline/stats')
  return res.data
}

export async function fetchGraph(params?: any) {
  const res = await api.get('/graph/', { params })
  return res.data
}

export async function fetchCompanyGraph(code: string) {
  const res = await api.get(`/graph/company/${code}`)
  return res.data
}

export async function fetchAlerts(params?: any) {
  const res = await api.get('/alert/', { params })
  return res.data
}

export async function fetchAlertStats() {
  const res = await api.get('/alert/stats')
  return res.data
}

export async function fetchAlertConfig() {
  const res = await api.get('/alert/config')
  return res.data
}

export default api
