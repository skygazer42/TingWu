import axios, { type AxiosInstance, type AxiosError } from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || ''

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 分钟超时（转写可能需要较长时间）
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    // 可以在这里添加认证 token 等
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    return response
  },
  (error: AxiosError) => {
    // 统一错误处理
    if (error.response) {
      const status = error.response.status
      const data = error.response.data as { detail?: string }

      switch (status) {
        case 400:
          console.error('请求参数错误:', data?.detail)
          break
        case 404:
          console.error('请求的资源不存在')
          break
        case 422:
          console.error('请求数据验证失败:', data?.detail)
          break
        case 500:
          console.error('服务器内部错误:', data?.detail)
          break
        default:
          console.error('请求失败:', status, data?.detail)
      }
    } else if (error.request) {
      console.error('网络错误: 无法连接到服务器')
    } else {
      console.error('请求配置错误:', error.message)
    }

    return Promise.reject(error)
  }
)

export default apiClient
