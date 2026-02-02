import { useState, useCallback, useEffect } from 'react'
import { toast } from 'sonner'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Mic, MicOff, Copy, Check, Trash2 } from 'lucide-react'
import { AudioWaveform, VolumeLevel, RecordingTimer } from '@/components/audio'
import { StreamingText, ModeSelector } from '@/components/realtime'
import { ConnectionStatus as ConnectionStatusComponent } from '@/components/realtime/ConnectionStatus'
import { useWebSocket, useAudioRecorder } from '@/hooks'
import type { WSMessage, WSResultMessage } from '@/lib/api/types'

type RealtimeMode = '2pass' | 'online' | 'offline'

interface TextSegment {
  text: string
  isFinal: boolean
}

export default function RealtimePage() {
  const [mode, setMode] = useState<RealtimeMode>('2pass')
  const [segments, setSegments] = useState<TextSegment[]>([])
  const [copied, setCopied] = useState(false)

  const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/realtime`

  // WebSocket 处理
  const handleMessage = useCallback((message: WSMessage) => {
    if ('mode' in message && 'text' in message) {
      const resultMsg = message as WSResultMessage
      setSegments((prev) => {
        // 如果是最终结果，替换最后一个非最终的段落
        if (resultMsg.is_final) {
          const newSegments = [...prev]
          // 移除连续的非最终段落
          while (newSegments.length > 0 && !newSegments[newSegments.length - 1].isFinal) {
            newSegments.pop()
          }
          newSegments.push({ text: resultMsg.text, isFinal: true })
          return newSegments
        } else {
          // 非最终结果，追加或更新
          const newSegments = [...prev]
          if (newSegments.length > 0 && !newSegments[newSegments.length - 1].isFinal) {
            newSegments[newSegments.length - 1] = { text: resultMsg.text, isFinal: false }
          } else {
            newSegments.push({ text: resultMsg.text, isFinal: false })
          }
          return newSegments
        }
      })
    }

    if ('warning' in message) {
      toast.warning(message.warning)
    }
  }, [])

  const { isConnected, isConnecting, connect, disconnect, send, sendJson } = useWebSocket(wsUrl, {
    onMessage: handleMessage,
    onOpen: () => {
      toast.success('已连接到实时转写服务')
    },
    onClose: () => {
      if (isRecording) {
        stop()
      }
    },
    onError: () => {
      toast.error('WebSocket 连接失败')
    },
  })

  // 音频录制
  const handleAudioData = useCallback(
    (data: ArrayBuffer) => {
      if (isConnected) {
        send(data)
      }
    },
    [isConnected, send]
  )

  const { isRecording, duration, volume, start, stop, getAnalyserNode } = useAudioRecorder({
    onAudioData: handleAudioData,
  })

  // 开始录制
  const handleStart = async () => {
    try {
      // 先连接 WebSocket
      if (!isConnected) {
        connect()
        // 等待连接
        await new Promise((resolve) => setTimeout(resolve, 500))
      }

      // 开始录制
      await start()

      // 发送配置
      sendJson({ is_speaking: true, mode })

      toast.success('开始录制')
    } catch (error) {
      console.error('Start recording error:', error)
      toast.error('无法启动麦克风，请检查权限设置')
    }
  }

  // 停止录制
  const handleStop = () => {
    // 通知服务器说话结束
    sendJson({ is_speaking: false })

    // 停止录制
    stop()

    toast.info('录制已停止')
  }

  // 复制文本
  const handleCopy = async () => {
    const text = segments
      .filter((s) => s.isFinal)
      .map((s) => s.text)
      .join('')

    if (text) {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      toast.success('已复制到剪贴板')
    }
  }

  // 清空文本
  const handleClear = () => {
    setSegments([])
  }

  // 组件卸载时断开连接
  useEffect(() => {
    return () => {
      if (isRecording) {
        stop()
      }
      disconnect()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const finalText = segments
    .filter((s) => s.isFinal)
    .map((s) => s.text)
    .join('')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">实时转写</h1>
          <p className="text-muted-foreground">实时录制音频并进行语音识别</p>
        </div>
        <ConnectionStatusComponent
          status={isConnected ? 'connected' : isConnecting ? 'connecting' : 'disconnected'}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* 录制控制 */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>录制控制</CardTitle>
            <CardDescription>点击按钮开始或停止录制</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* 模式选择 */}
            <ModeSelector
              value={mode}
              onChange={setMode}
              disabled={isRecording}
            />

            {/* 波形显示 */}
            <div className="space-y-2">
              <p className="text-sm text-muted-foreground">音频波形</p>
              <AudioWaveform
                analyser={getAnalyserNode()}
                isActive={isRecording}
              />
            </div>

            {/* 音量和计时 */}
            <div className="flex items-center justify-between">
              <VolumeLevel volume={volume} />
              <RecordingTimer seconds={duration} isRecording={isRecording} />
            </div>

            {/* 录制按钮 */}
            <Button
              size="lg"
              className="w-full"
              variant={isRecording ? 'destructive' : 'default'}
              onClick={isRecording ? handleStop : handleStart}
              disabled={isConnecting}
            >
              {isRecording ? (
                <>
                  <MicOff className="h-5 w-5 mr-2" />
                  停止录制
                </>
              ) : (
                <>
                  <Mic className="h-5 w-5 mr-2" />
                  开始录制
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* 实时结果 */}
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
            <div>
              <CardTitle>实时转写结果</CardTitle>
              <CardDescription>语音识别结果将实时显示在这里</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
                disabled={!finalText}
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 mr-1" />
                    已复制
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4 mr-1" />
                    复制
                  </>
                )}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={handleClear}
                disabled={segments.length === 0}
              >
                <Trash2 className="h-4 w-4 mr-1" />
                清空
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[300px] rounded-lg border bg-muted/30 p-4">
              <StreamingText segments={segments} />
            </ScrollArea>

            {/* 统计信息 */}
            <div className="flex items-center justify-between mt-4 text-sm text-muted-foreground">
              <span>已识别 {segments.filter((s) => s.isFinal).length} 段</span>
              <span>共 {finalText.length} 字</span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
