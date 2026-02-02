import { useMemo } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { SpeakerBadge } from './SpeakerBadge'
import { getSpeakerColor } from './speakerColors'
import type { SentenceInfo } from '@/lib/api/types'

interface SpeakerStatsProps {
  sentences: SentenceInfo[]
}

interface SpeakerStat {
  speakerId: number
  speaker?: string
  count: number
  duration: number
  percentage: number
}

export function SpeakerStats({ sentences }: SpeakerStatsProps) {
  const stats = useMemo(() => {
    const speakerMap = new Map<number, SpeakerStat>()
    let totalDuration = 0

    for (const sentence of sentences) {
      const id = sentence.speaker_id ?? -1
      const duration = sentence.end - sentence.start
      totalDuration += duration

      if (speakerMap.has(id)) {
        const stat = speakerMap.get(id)!
        stat.count++
        stat.duration += duration
      } else {
        speakerMap.set(id, {
          speakerId: id,
          speaker: sentence.speaker,
          count: 1,
          duration,
          percentage: 0,
        })
      }
    }

    // 计算百分比
    const result: SpeakerStat[] = []
    for (const stat of speakerMap.values()) {
      stat.percentage = totalDuration > 0 ? (stat.duration / totalDuration) * 100 : 0
      result.push(stat)
    }

    return result.sort((a, b) => b.duration - a.duration)
  }, [sentences])

  if (stats.length === 0) {
    return null
  }

  // 过滤掉未识别的说话人
  const validStats = stats.filter(s => s.speakerId >= 0)

  if (validStats.length === 0) {
    return null
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">说话人统计</CardTitle>
        <CardDescription>各说话人发言时长占比</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {validStats.map((stat) => {
          const colors = getSpeakerColor(stat.speakerId)
          return (
            <div key={stat.speakerId} className="space-y-1">
              <div className="flex items-center justify-between">
                <SpeakerBadge
                  speaker={stat.speaker}
                  speakerId={stat.speakerId}
                />
                <span className="text-sm text-muted-foreground">
                  {stat.count} 句 · {formatDuration(stat.duration)}
                </span>
              </div>
              <div className="h-2 rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all ${colors.bg.replace('/10', '/50')}`}
                  style={{ width: `${stat.percentage}%` }}
                />
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes > 0) {
    return `${minutes}分${seconds}秒`
  }
  return `${seconds}秒`
}
