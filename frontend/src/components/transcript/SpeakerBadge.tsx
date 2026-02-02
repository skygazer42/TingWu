import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import { getSpeakerColor } from './speakerColors'

interface SpeakerBadgeProps {
  speaker?: string
  speakerId?: number
  className?: string
}

export function SpeakerBadge({ speaker, speakerId, className }: SpeakerBadgeProps) {
  if (!speaker && speakerId === undefined) {
    return null
  }

  const colors = getSpeakerColor(speakerId ?? 0)
  const label = speaker || `说话人 ${String.fromCharCode(65 + (speakerId ?? 0))}`

  return (
    <Badge
      variant="outline"
      className={cn(
        colors.bg,
        colors.text,
        colors.border,
        'font-medium',
        className
      )}
    >
      {label}
    </Badge>
  )
}
