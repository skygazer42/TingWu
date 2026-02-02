import { useRef, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface AudioWaveformProps {
  analyser: AnalyserNode | null
  isActive?: boolean
  className?: string
}

export function AudioWaveform({ analyser, isActive = false, className }: AudioWaveformProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      const width = canvas.width
      const height = canvas.height

      // 清除画布
      ctx.clearRect(0, 0, width, height)

      if (!analyser || !isActive) {
        // 绘制静态线
        ctx.beginPath()
        ctx.moveTo(0, height / 2)
        ctx.lineTo(width, height / 2)
        ctx.strokeStyle = 'hsl(var(--muted-foreground) / 0.3)'
        ctx.lineWidth = 2
        ctx.stroke()
        return
      }

      // 获取波形数据
      const bufferLength = analyser.frequencyBinCount
      const dataArray = new Uint8Array(bufferLength)
      analyser.getByteTimeDomainData(dataArray)

      // 绘制波形
      ctx.beginPath()
      ctx.strokeStyle = 'hsl(var(--primary))'
      ctx.lineWidth = 2

      const sliceWidth = width / bufferLength
      let x = 0

      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0
        const y = (v * height) / 2

        if (i === 0) {
          ctx.moveTo(x, y)
        } else {
          ctx.lineTo(x, y)
        }

        x += sliceWidth
      }

      ctx.lineTo(width, height / 2)
      ctx.stroke()

      animationRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [analyser, isActive])

  // 处理画布尺寸
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        canvas.width = width * window.devicePixelRatio
        canvas.height = height * window.devicePixelRatio
        const ctx = canvas.getContext('2d')
        if (ctx) {
          ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
        }
      }
    })

    resizeObserver.observe(canvas)
    return () => resizeObserver.disconnect()
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={cn('w-full h-16 rounded-lg bg-muted/50', className)}
      style={{ width: '100%', height: '64px' }}
    />
  )
}
