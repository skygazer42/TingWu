// API 类型定义 - 基于后端 src/api/schemas.py

export interface SentenceInfo {
  text: string
  start: number  // 毫秒
  end: number    // 毫秒
  speaker?: string
  speaker_id?: number
}

export interface SpeakerTurn {
  speaker: string
  speaker_id: number
  start: number  // 毫秒
  end: number    // 毫秒
  text: string
  sentence_count: number
}

export interface TranscribeResponse {
  code: number
  text: string
  text_accu?: string | null
  sentences: SentenceInfo[]
  speaker_turns?: SpeakerTurn[] | null
  transcript?: string
  raw_text?: string
}

export interface BatchTranscribeItem {
  index: number
  filename: string
  success: boolean
  result?: TranscribeResponse
  error?: string
}

export interface BatchTranscribeResponse {
  code: number
  total: number
  success_count: number
  failed_count: number
  results: BatchTranscribeItem[]
}

// URL 转写相关
export interface UrlTranscribeRequest {
  url: string
  with_speaker?: boolean
  apply_hotword?: boolean
  apply_llm?: boolean
  llm_role?: string
  hotwords?: string
}

export interface UrlTranscribeResponse {
  code: number
  task_id: string
  message: string
}

// 异步任务相关
export interface TaskResultRequest {
  task_id: string
  wait?: boolean
  timeout?: number
}

export interface TaskResultResponse {
  code: number
  task_id: string
  status: 'pending' | 'processing' | 'success' | 'error'
  result?: TranscribeResponse
  error?: string
  progress?: number
}

// 视频转写相关
export interface VideoTranscribeResponse extends TranscribeResponse {
  video_duration?: number
  audio_extracted?: boolean
}

export interface HealthResponse {
  status: string
  version: string
}

export interface MetricsResponse {
  uptime_seconds: number
  total_requests: number
  successful_requests: number
  failed_requests: number
  total_audio_seconds: number
  avg_rtf: number
  llm_cache_stats: Record<string, unknown>
}

export interface BackendCapabilities {
  supports_speaker: boolean
  supports_streaming: boolean
  supports_hotwords: boolean
}

export interface BackendInfoResponse {
  backend: string
  info: Record<string, unknown>
  capabilities: BackendCapabilities
  speaker_unsupported_behavior: 'error' | 'fallback' | 'ignore'
}

// 热词相关
export interface HotwordsListResponse {
  code: number
  hotwords: string[]
  count: number
}

export interface HotwordsUpdateRequest {
  hotwords: string[]
}

export interface HotwordsUpdateResponse {
  code: number
  count: number
  message: string
}

// 配置相关
export interface ConfigResponse {
  config: Record<string, unknown>
}

export interface ConfigAllResponse {
  config: Record<string, unknown>
  mutable_keys: string[]
}

export interface ConfigUpdateRequest {
  updates: Record<string, unknown>
}

// 转写请求选项
export interface TranscribeOptions {
  with_speaker?: boolean
  apply_hotword?: boolean
  apply_llm?: boolean
  llm_role?: 'default' | 'translator' | 'code' | 'corrector'
  hotwords?: string
  speaker_label_style?: 'numeric' | 'zh'
}

// WebSocket 消息类型
export interface WSConnectedMessage {
  type: 'connected'
  connection_id: string
  config: {
    chunk_size: number
    heartbeat_interval: number
    compression: boolean
  }
}

export interface WSResultMessage {
  mode: '2pass-online' | '2pass-offline' | 'online' | 'offline'
  text: string
  is_final: boolean
}

export interface WSPingMessage {
  type: 'ping'
  timestamp: number
}

export interface WSWarningMessage {
  warning: string
  backend: string
}

export type WSMessage = WSConnectedMessage | WSResultMessage | WSPingMessage | WSWarningMessage
