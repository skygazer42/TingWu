import { useState, useRef, useCallback, useEffect } from 'react'

interface UseAudioRecorderOptions {
  sampleRate?: number
  onAudioData?: (data: ArrayBuffer) => void
  onVolumeChange?: (volume: number) => void
}

interface UseAudioRecorderReturn {
  isRecording: boolean
  isPaused: boolean
  duration: number
  volume: number
  start: () => Promise<void>
  stop: () => void
  pause: () => void
  resume: () => void
  getAnalyserNode: () => AnalyserNode | null
}

export function useAudioRecorder(
  options: UseAudioRecorderOptions = {}
): UseAudioRecorderReturn {
  const { sampleRate = 16000, onAudioData, onVolumeChange } = options

  const [isRecording, setIsRecording] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(0)

  const audioContextRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const durationIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const volumeIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const startTimeRef = useRef<number>(0)

  const start = useCallback(async () => {
    try {
      // 请求麦克风权限
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })

      streamRef.current = stream

      // 创建音频上下文
      const audioContext = new AudioContext({ sampleRate })
      audioContextRef.current = audioContext

      // 创建音频源
      const source = audioContext.createMediaStreamSource(stream)

      // 创建分析器（用于波形和音量）
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      analyserRef.current = analyser
      source.connect(analyser)

      // 创建处理器（用于获取 PCM 数据）
      const processor = audioContext.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      processor.onaudioprocess = (e) => {
        if (!isPaused) {
          const inputData = e.inputBuffer.getChannelData(0)
          // 转换为 16bit PCM
          const pcm16 = floatTo16BitPCM(inputData)
          onAudioData?.(pcm16.buffer as ArrayBuffer)
        }
      }

      source.connect(processor)
      processor.connect(audioContext.destination)

      // 开始计时
      startTimeRef.current = Date.now()
      durationIntervalRef.current = setInterval(() => {
        setDuration(Math.floor((Date.now() - startTimeRef.current) / 1000))
      }, 1000)

      // 音量检测
      const dataArray = new Uint8Array(analyser.frequencyBinCount)
      volumeIntervalRef.current = setInterval(() => {
        analyser.getByteFrequencyData(dataArray)
        const avg = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
        const normalizedVolume = avg / 255
        setVolume(normalizedVolume)
        onVolumeChange?.(normalizedVolume)
      }, 100)

      setIsRecording(true)
      setIsPaused(false)
      setDuration(0)
    } catch (error) {
      console.error('Failed to start recording:', error)
      throw error
    }
  }, [sampleRate, onAudioData, onVolumeChange, isPaused])

  const stop = useCallback(() => {
    // 停止计时
    if (durationIntervalRef.current) {
      clearInterval(durationIntervalRef.current)
    }
    if (volumeIntervalRef.current) {
      clearInterval(volumeIntervalRef.current)
    }

    // 断开处理器
    if (processorRef.current) {
      processorRef.current.disconnect()
      processorRef.current = null
    }

    // 关闭音频上下文
    if (audioContextRef.current) {
      audioContextRef.current.close()
      audioContextRef.current = null
    }

    // 停止媒体流
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    analyserRef.current = null
    setIsRecording(false)
    setIsPaused(false)
    setVolume(0)
  }, [])

  const pause = useCallback(() => {
    setIsPaused(true)
  }, [])

  const resume = useCallback(() => {
    setIsPaused(false)
  }, [])

  const getAnalyserNode = useCallback(() => {
    return analyserRef.current
  }, [])

  // 清理
  useEffect(() => {
    return () => {
      stop()
    }
  }, [stop])

  return {
    isRecording,
    isPaused,
    duration,
    volume,
    start,
    stop,
    pause,
    resume,
    getAnalyserNode,
  }
}

function floatTo16BitPCM(float32Array: Float32Array): Int16Array {
  const int16Array = new Int16Array(float32Array.length)
  for (let i = 0; i < float32Array.length; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]))
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  return int16Array
}
