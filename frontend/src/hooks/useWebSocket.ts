import { useState, useRef, useCallback, useEffect } from 'react'
import type { WSMessage } from '@/lib/api/types'

type ConnectionStatus = 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
type LLMStatus = 'idle' | 'generating' | 'cancelled'

interface UseWebSocketOptions {
  onMessage?: (message: WSMessage) => void
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  onLLMStatusChange?: (status: LLMStatus) => void
  reconnectAttempts?: number
  reconnectInterval?: number
  autoReconnect?: boolean
}

interface UseWebSocketReturn {
  isConnected: boolean
  isConnecting: boolean
  connectionStatus: ConnectionStatus
  connectionId: string | null
  llmStatus: LLMStatus
  latency: number | null
  connect: () => void
  disconnect: () => void
  send: (data: string | ArrayBuffer) => void
  sendJson: (data: object) => void
  cancelLLM: () => void
}

export function useWebSocket(
  url: string,
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    onMessage,
    onOpen,
    onClose,
    onError,
    onLLMStatusChange,
    reconnectAttempts = 3,
    reconnectInterval = 2000,
    autoReconnect = true,
  } = options

  const [connectionStatus, setConnectionStatus] = useState<ConnectionStatus>('disconnected')
  const [connectionId, setConnectionId] = useState<string | null>(null)
  const [llmStatus, setLLMStatus] = useState<LLMStatus>('idle')
  const [latency, setLatency] = useState<number | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pingTimestampRef = useRef<number | null>(null)

  const isConnected = connectionStatus === 'connected'
  const isConnecting = connectionStatus === 'connecting' || connectionStatus === 'reconnecting'

  const updateLLMStatus = useCallback((status: LLMStatus) => {
    setLLMStatus(status)
    onLLMStatusChange?.(status)
  }, [onLLMStatusChange])

  const connect = useCallback(function connectImpl() {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const isReconnect = reconnectCountRef.current > 0
    setConnectionStatus(isReconnect ? 'reconnecting' : 'connecting')

    try {
      const ws = new WebSocket(url)
      wsRef.current = ws

      ws.onopen = () => {
        setConnectionStatus('connected')
        reconnectCountRef.current = 0
        onOpen?.()
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WSMessage

          // 处理连接确认消息
          if ('type' in data && data.type === 'connected') {
            setConnectionId(data.connection_id)
          }

          // 处理心跳/延迟测量
          if ('type' in data && data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }))
          }

          // 处理 pong 响应 (测量延迟)
          if ('type' in data && (data as { type: string }).type === 'pong') {
            if (pingTimestampRef.current) {
              setLatency(Date.now() - pingTimestampRef.current)
              pingTimestampRef.current = null
            }
          }

          // 处理 LLM 状态消息
          if ('llm_status' in data) {
            const status = (data as { llm_status: string }).llm_status
            if (status === 'generating') {
              updateLLMStatus('generating')
            } else if (status === 'cancelled') {
              updateLLMStatus('cancelled')
              // 短暂显示取消状态后重置
              setTimeout(() => updateLLMStatus('idle'), 1000)
            } else if (status === 'completed' || status === 'idle') {
              updateLLMStatus('idle')
            }
          }

          // 如果收到带有 mode 的结果消息，检查是否有 LLM 润色
          if ('mode' in data && 'text' in data) {
            // 结果消息，如果正在生成则标记完成
            if (llmStatus === 'generating' && (data as { is_final?: boolean }).is_final) {
              updateLLMStatus('idle')
            }
          }

          onMessage?.(data)
        } catch {
          // 非 JSON 消息，忽略
        }
      }

      ws.onclose = () => {
        setConnectionStatus('disconnected')
        setConnectionId(null)
        updateLLMStatus('idle')
        onClose?.()

        // 尝试重连
        if (autoReconnect && reconnectCountRef.current < reconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectCountRef.current++
            connectImpl()
          }, reconnectInterval * Math.pow(2, reconnectCountRef.current)) // 指数退避
        }
      }

      ws.onerror = (error) => {
        onError?.(error)
      }
    } catch (error) {
      setConnectionStatus('disconnected')
      console.error('WebSocket connection error:', error)
    }
  }, [url, onMessage, onOpen, onClose, onError, reconnectAttempts, reconnectInterval, autoReconnect, updateLLMStatus, llmStatus])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    reconnectCountRef.current = reconnectAttempts // 防止重连
    wsRef.current?.close()
    wsRef.current = null
    setConnectionStatus('disconnected')
    setConnectionId(null)
    updateLLMStatus('idle')
  }, [reconnectAttempts, updateLLMStatus])

  const send = useCallback((data: string | ArrayBuffer) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data)
    }
  }, [])

  const sendJson = useCallback((data: object) => {
    send(JSON.stringify(data))
  }, [send])

  /**
   * 取消 LLM 生成
   */
  const cancelLLM = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN && llmStatus === 'generating') {
      sendJson({ type: 'cancel_llm' })
      updateLLMStatus('cancelled')
    }
  }, [sendJson, llmStatus, updateLLMStatus])

  // 清理
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      wsRef.current?.close()
    }
  }, [])

  return {
    isConnected,
    isConnecting,
    connectionStatus,
    connectionId,
    llmStatus,
    latency,
    connect,
    disconnect,
    send,
    sendJson,
    cancelLLM,
  }
}
