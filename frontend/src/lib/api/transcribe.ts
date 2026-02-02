import apiClient from './client'
import type {
  TranscribeResponse,
  BatchTranscribeResponse,
  TranscribeOptions,
  UrlTranscribeResponse,
  TaskResultResponse,
  VideoTranscribeResponse,
} from './types'

export interface UploadProgressCallback {
  (progress: number): void
}

export interface TranscribeWithProgressOptions extends TranscribeOptions {
  onUploadProgress?: UploadProgressCallback
  signal?: AbortSignal
}

/**
 * 单文件转写 (带上传进度)
 */
export async function transcribeAudio(
  file: File,
  options: TranscribeWithProgressOptions = {}
): Promise<TranscribeResponse> {
  const { onUploadProgress, signal, ...transcribeOptions } = options
  const formData = new FormData()
  formData.append('file', file)

  if (transcribeOptions.with_speaker !== undefined) {
    formData.append('with_speaker', String(transcribeOptions.with_speaker))
  }
  if (transcribeOptions.apply_hotword !== undefined) {
    formData.append('apply_hotword', String(transcribeOptions.apply_hotword))
  }
  if (transcribeOptions.apply_llm !== undefined) {
    formData.append('apply_llm', String(transcribeOptions.apply_llm))
  }
  if (transcribeOptions.llm_role) {
    formData.append('llm_role', transcribeOptions.llm_role)
  }
  if (transcribeOptions.hotwords) {
    formData.append('hotwords', transcribeOptions.hotwords)
  }

  const response = await apiClient.post<TranscribeResponse>(
    '/api/v1/transcribe',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      signal,
      onUploadProgress: onUploadProgress
        ? (progressEvent) => {
            const total = progressEvent.total || file.size
            const progress = Math.round((progressEvent.loaded * 100) / total)
            onUploadProgress(progress)
          }
        : undefined,
    }
  )
  return response.data
}

/**
 * 批量文件转写 (带上传进度)
 */
export async function transcribeBatch(
  files: File[],
  options: TranscribeWithProgressOptions & { max_concurrent?: number } = {}
): Promise<BatchTranscribeResponse> {
  const { onUploadProgress, signal, ...transcribeOptions } = options
  const formData = new FormData()

  files.forEach((file) => {
    formData.append('files', file)
  })

  if (transcribeOptions.with_speaker !== undefined) {
    formData.append('with_speaker', String(transcribeOptions.with_speaker))
  }
  if (transcribeOptions.apply_hotword !== undefined) {
    formData.append('apply_hotword', String(transcribeOptions.apply_hotword))
  }
  if (transcribeOptions.apply_llm !== undefined) {
    formData.append('apply_llm', String(transcribeOptions.apply_llm))
  }
  if (transcribeOptions.llm_role) {
    formData.append('llm_role', transcribeOptions.llm_role)
  }
  if (transcribeOptions.hotwords) {
    formData.append('hotwords', transcribeOptions.hotwords)
  }
  if (transcribeOptions.max_concurrent !== undefined) {
    formData.append('max_concurrent', String(transcribeOptions.max_concurrent))
  }

  const totalSize = files.reduce((acc, f) => acc + f.size, 0)

  const response = await apiClient.post<BatchTranscribeResponse>(
    '/api/v1/transcribe/batch',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      signal,
      onUploadProgress: onUploadProgress
        ? (progressEvent) => {
            const total = progressEvent.total || totalSize
            const progress = Math.round((progressEvent.loaded * 100) / total)
            onUploadProgress(progress)
          }
        : undefined,
    }
  )
  return response.data
}

/**
 * URL 转写 (异步)
 */
export async function transcribeUrl(
  url: string,
  options: TranscribeOptions = {}
): Promise<UrlTranscribeResponse> {
  const response = await apiClient.post<UrlTranscribeResponse>(
    '/api/v1/trans/url',
    {
      url,
      with_speaker: options.with_speaker,
      apply_hotword: options.apply_hotword,
      apply_llm: options.apply_llm,
      llm_role: options.llm_role,
      hotwords: options.hotwords,
    }
  )
  return response.data
}

/**
 * 查询异步任务结果
 */
export async function getTaskResult(
  taskId: string,
  wait: boolean = false,
  timeout: number = 30
): Promise<TaskResultResponse> {
  const response = await apiClient.post<TaskResultResponse>(
    '/api/v1/result',
    {
      task_id: taskId,
      wait,
      timeout,
    }
  )
  return response.data
}

/**
 * 视频转写 (带上传进度)
 */
export async function transcribeVideo(
  file: File,
  options: TranscribeWithProgressOptions = {}
): Promise<VideoTranscribeResponse> {
  const { onUploadProgress, signal, ...transcribeOptions } = options
  const formData = new FormData()
  formData.append('file', file)

  if (transcribeOptions.with_speaker !== undefined) {
    formData.append('with_speaker', String(transcribeOptions.with_speaker))
  }
  if (transcribeOptions.apply_hotword !== undefined) {
    formData.append('apply_hotword', String(transcribeOptions.apply_hotword))
  }
  if (transcribeOptions.apply_llm !== undefined) {
    formData.append('apply_llm', String(transcribeOptions.apply_llm))
  }
  if (transcribeOptions.llm_role) {
    formData.append('llm_role', transcribeOptions.llm_role)
  }
  if (transcribeOptions.hotwords) {
    formData.append('hotwords', transcribeOptions.hotwords)
  }

  const response = await apiClient.post<VideoTranscribeResponse>(
    '/api/v1/trans/video',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      signal,
      onUploadProgress: onUploadProgress
        ? (progressEvent) => {
            const total = progressEvent.total || file.size
            const progress = Math.round((progressEvent.loaded * 100) / total)
            onUploadProgress(progress)
          }
        : undefined,
    }
  )
  return response.data
}

/**
 * 判断文件是否为视频
 */
export function isVideoFile(file: File): boolean {
  const videoTypes = ['video/mp4', 'video/avi', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm']
  const videoExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

  if (videoTypes.includes(file.type)) return true

  const ext = file.name.toLowerCase().slice(file.name.lastIndexOf('.'))
  return videoExtensions.includes(ext)
}

/**
 * 智能转写 - 根据文件类型选择接口
 */
export async function transcribeSmart(
  file: File,
  options: TranscribeWithProgressOptions = {}
): Promise<TranscribeResponse> {
  if (isVideoFile(file)) {
    return transcribeVideo(file, options)
  }
  return transcribeAudio(file, options)
}
