import { cn } from '@/lib/utils'

interface VolumeLevelProps {
  volume: number // 0-1
  className?: string
}

export function VolumeLevel({ volume, className }: VolumeLevelProps) {
  const bars = 10
  const activeCount = Math.round(volume * bars)

  return (
    <div className={cn('flex items-end gap-1 h-8', className)}>
      {Array.from({ length: bars }).map((_, i) => {
        const isActive = i < activeCount
        const height = ((i + 1) / bars) * 100

        return (
          <div
            key={i}
            className={cn(
              'w-2 rounded-full transition-all duration-100',
              isActive
                ? i < bars * 0.6
                  ? 'bg-green-500'
                  : i < bars * 0.8
                    ? 'bg-yellow-500'
                    : 'bg-red-500'
                : 'bg-muted'
            )}
            style={{ height: `${height}%` }}
          />
        )
      })}
    </div>
  )
}
