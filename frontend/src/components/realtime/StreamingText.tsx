import { useEffect, useRef } from 'react'
import { cn } from '@/lib/utils'

interface StreamingTextProps {
  segments: { text: string; isFinal: boolean }[]
  className?: string
}

export function StreamingText({ segments, className }: StreamingTextProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  // 自动滚动到底部
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight
    }
  }, [segments])

  if (segments.length === 0) {
    return (
      <div className={cn('text-muted-foreground', className)}>
        等待语音输入...
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className={cn('space-y-2 overflow-y-auto', className)}
    >
      {segments.map((segment, index) => (
        <span
          key={index}
          className={cn(
            'inline',
            segment.isFinal
              ? 'text-foreground'
              : 'text-muted-foreground italic'
          )}
        >
          {segment.text}
          {!segment.isFinal && (
            <span className="inline-block w-2 h-4 ml-0.5 bg-primary/50 animate-pulse" />
          )}
        </span>
      ))}
    </div>
  )
}
