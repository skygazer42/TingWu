import { useRef, useState, useEffect, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { getSpeakerColor } from '@/components/transcript/speakerColors'
import type { SentenceInfo } from '@/lib/api/types'

interface TimelineProps {
  sentences: SentenceInfo[]
  selectedIndex?: number
  onSelectSentence?: (sentence: SentenceInfo, index: number) => void
  className?: string
}

export function Timeline({
  sentences,
  selectedIndex,
  onSelectSentence,
  className,
}: TimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerWidth, setContainerWidth] = useState(0)
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  // 计算总时长
  const totalDuration = sentences.length > 0
    ? sentences[sentences.length - 1].end
    : 0

  // 监听容器宽度变化
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setContainerWidth(entry.contentRect.width)
      }
    })

    observer.observe(container)
    setContainerWidth(container.offsetWidth)

    return () => observer.disconnect()
  }, [])

  const getSegmentStyle = useCallback(
    (sentence: SentenceInfo) => {
      if (totalDuration === 0 || containerWidth === 0) {
        return { left: 0, width: 0 }
      }

      const left = (sentence.start / totalDuration) * 100
      const width = ((sentence.end - sentence.start) / totalDuration) * 100

      return {
        left: `${left}%`,
        width: `${Math.max(width, 0.5)}%`, // 最小宽度 0.5%
      }
    },
    [totalDuration, containerWidth]
  )

  if (sentences.length === 0) {
    return null
  }

  return (
    <div className={cn('space-y-2', className)}>
      {/* 时间刻度 */}
      <div className="flex justify-between text-xs text-muted-foreground px-1">
        <span>00:00</span>
        <span>{formatTime(totalDuration / 2)}</span>
        <span>{formatTime(totalDuration)}</span>
      </div>

      {/* 时间轴 */}
      <div
        ref={containerRef}
        className="relative h-8 bg-muted rounded-lg overflow-hidden"
      >
        {sentences.map((sentence, index) => {
          const style = getSegmentStyle(sentence)
          const colors = sentence.speaker_id !== undefined
            ? getSpeakerColor(sentence.speaker_id)
            : { bg: 'bg-primary/50' }
          const isSelected = selectedIndex === index
          const isHovered = hoveredIndex === index

          return (
            <div
              key={index}
              className={cn(
                'absolute top-0 h-full cursor-pointer transition-all',
                colors.bg.replace('/10', '/60'),
                isSelected && 'ring-2 ring-primary z-10',
                isHovered && 'brightness-110'
              )}
              style={style}
              onClick={() => onSelectSentence?.(sentence, index)}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            />
          )
        })}

        {/* Tooltip */}
        {hoveredIndex !== null && sentences[hoveredIndex] && (
          <TimelineTooltip
            sentence={sentences[hoveredIndex]}
            containerWidth={containerWidth}
            totalDuration={totalDuration}
          />
        )}
      </div>

      {/* 图例 */}
      {sentences.some(s => s.speaker_id !== undefined) && (
        <TimelineLegend sentences={sentences} />
      )}
    </div>
  )
}

interface TimelineTooltipProps {
  sentence: SentenceInfo
  containerWidth: number
  totalDuration: number
}

function TimelineTooltip({ sentence, containerWidth, totalDuration }: TimelineTooltipProps) {
  const left = (sentence.start / totalDuration) * containerWidth
  const adjustedLeft = Math.min(Math.max(left, 100), containerWidth - 100)

  return (
    <div
      className="absolute -top-12 px-2 py-1 bg-popover text-popover-foreground text-xs rounded shadow-lg z-20 max-w-[200px] truncate"
      style={{ left: adjustedLeft, transform: 'translateX(-50%)' }}
    >
      <div className="font-medium">{formatTime(sentence.start)} - {formatTime(sentence.end)}</div>
      <div className="truncate">{sentence.text.slice(0, 30)}{sentence.text.length > 30 ? '...' : ''}</div>
    </div>
  )
}

interface TimelineLegendProps {
  sentences: SentenceInfo[]
}

function TimelineLegend({ sentences }: TimelineLegendProps) {
  // 获取唯一的说话人
  const speakers = new Map<number, string | undefined>()
  for (const sentence of sentences) {
    if (sentence.speaker_id !== undefined && !speakers.has(sentence.speaker_id)) {
      speakers.set(sentence.speaker_id, sentence.speaker)
    }
  }

  if (speakers.size === 0) return null

  return (
    <div className="flex flex-wrap gap-3 text-xs">
      {Array.from(speakers.entries()).map(([id, speaker]) => {
        const colors = getSpeakerColor(id)
        const label = speaker || `说话人${id + 1}`
        return (
          <div key={id} className="flex items-center gap-1.5">
            <div className={cn('w-3 h-3 rounded', colors.bg.replace('/10', '/60'))} />
            <span className="text-muted-foreground">{label}</span>
          </div>
        )
      })}
    </div>
  )
}

function formatTime(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}
