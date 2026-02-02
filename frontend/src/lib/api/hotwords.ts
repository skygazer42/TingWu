import apiClient from './client'
import type { HotwordsListResponse, HotwordsUpdateResponse } from './types'

/**
 * 获取当前热词列表
 */
export async function getHotwords(): Promise<HotwordsListResponse> {
  const response = await apiClient.get<HotwordsListResponse>('/api/v1/hotwords')
  return response.data
}

/**
 * 更新热词列表（替换全部）
 */
export async function updateHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords',
    { hotwords }
  )
  return response.data
}

/**
 * 追加热词（保留现有）
 */
export async function appendHotwords(hotwords: string[]): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>(
    '/api/v1/hotwords/append',
    { hotwords }
  )
  return response.data
}

/**
 * 从文件重新加载热词
 */
export async function reloadHotwords(): Promise<HotwordsUpdateResponse> {
  const response = await apiClient.post<HotwordsUpdateResponse>('/api/v1/hotwords/reload')
  return response.data
}
