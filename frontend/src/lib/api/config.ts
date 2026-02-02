import apiClient from './client'
import type { ConfigResponse, ConfigAllResponse } from './types'

/**
 * 获取可变配置
 */
export async function getConfig(): Promise<ConfigResponse> {
  const response = await apiClient.get<ConfigResponse>('/config')
  return response.data
}

/**
 * 获取完整配置（包括只读项）
 */
export async function getAllConfig(): Promise<ConfigAllResponse> {
  const response = await apiClient.get<ConfigAllResponse>('/config/all')
  return response.data
}

/**
 * 更新配置
 */
export async function updateConfig(updates: Record<string, unknown>): Promise<ConfigResponse> {
  const response = await apiClient.post<ConfigResponse>('/config', { updates })
  return response.data
}

/**
 * 重新加载引擎
 */
export async function reloadEngine(): Promise<{ status: string; message: string }> {
  const response = await apiClient.post<{ status: string; message: string }>('/config/reload')
  return response.data
}
