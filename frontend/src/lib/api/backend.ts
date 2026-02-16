import apiClient from './client'
import type { BackendInfoResponse } from './types'

export async function getBackendInfo(): Promise<BackendInfoResponse> {
  const response = await apiClient.get<BackendInfoResponse>('/api/v1/backend')
  return response.data
}

