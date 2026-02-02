import { cn } from '@/lib/utils'
import { SpeakerBadge } from './SpeakerBadge'
import { getSpeakerColor } from './speakerColors'
import { DiffHighlight } from './DiffHighlight'
import type { SentenceInfo } from '@/lib/api/types'

interface SentenceListProps {
  sentences: SentenceInfo[]
  rawText?: string
  showDiff?: boolean
  showTimestamp?: boolean
  showSpeaker?: boolean
  selectedIndex?: number
  onSelectSentence?: (sentence: SentenceInfo, index: number) => void
  className?: string
}

export function SentenceList({
  sentences,
  rawText,
  showDiff = false,
  showTimestamp = true,
  showSpeaker = true,
  selectedIndex,
  onSelectSentence,
  className,
}: SentenceListProps) {
  // 从 rawText 解析原始句子（用于 diff）
  const rawSentences = rawText ? parseRawSentences(rawText, sentences) : []

  return (
    <div className={cn('space-y-2', className)}>
      {sentences.map((sentence, index) => {
        const isSelected = selectedIndex === index
        const colors = sentence.speaker_id !== undefined
          ? getSpeakerColor(sentence.speaker_id)
          : null

        return (
          <div
            key={index}
            className={cn(
              'p-3 rounded-lg transition-colors cursor-pointer',
              'hover:bg-muted/50',
              isSelected && 'bg-primary/5 ring-1 ring-primary/20',
              colors && `border-l-4 ${colors.border}`
            )}
            onClick={() => onSelectSentence?.(sentence, index)}
          >
            <div className="flex items-start gap-3">
              {/* 时间戳 */}
              {showTimestamp && (
                <span className="text-xs text-muted-foreground font-mono shrink-0 pt-0.5">
                  {formatTime(sentence.start)}
                </span>
              )}

              <div className="flex-1 min-w-0">
                {/* 说话人标签 */}
                {showSpeaker && (sentence.speaker || sentence.speaker_id !== undefined) && (
                  <div className="mb-1">
                    <SpeakerBadge
                      speaker={sentence.speaker}
                      speakerId={sentence.speaker_id}
                    />
                  </div>
                )}

                {/* 文本内容 */}
                {showDiff && rawSentences[index] ? (
                  <DiffHighlight
                    original={rawSentences[index]}
                    corrected={sentence.text}
                    className="text-sm leading-relaxed"
                  />
                ) : (
                  <p className="text-sm leading-relaxed">{sentence.text}</p>
                )}
              </div>

              {/* 时长 */}
              {showTimestamp && (
                <span className="text-xs text-muted-foreground shrink-0">
                  {formatDuration(sentence.end - sentence.start)}
                </span>
              )}
            </div>
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

function formatDuration(ms: number): string {
  const seconds = (ms / 1000).toFixed(1)
  return `${seconds}s`
}

function parseRawSentences(rawText: string, sentences: SentenceInfo[]): string[] {
  // 简单的句子分割，尝试与 sentences 对应
  // 这是一个简化的实现，实际可能需要更复杂的对齐算法
  const rawParts: string[] = []
  let remaining = rawText

  for (let i = 0; i < sentences.length; i++) {
    const sentenceLength = sentences[i].text.length
    // 取相似长度的原始文本
    const rawPart = remaining.slice(0, Math.min(sentenceLength + 5, remaining.length))
    rawParts.push(rawPart.trim())
    remaining = remaining.slice(rawPart.length).trim()
  }

  return rawParts
}
