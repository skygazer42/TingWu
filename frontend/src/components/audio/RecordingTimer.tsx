import { cn } from '@/lib/utils'

interface RecordingTimerProps {
  seconds: number
  isRecording?: boolean
  className?: string
}

export function RecordingTimer({ seconds, isRecording = false, className }: RecordingTimerProps) {
  const minutes = Math.floor(seconds / 60)
  const secs = seconds % 60

  return (
    <div className={cn('flex items-center gap-2', className)}>
      {isRecording && (
        <span className="relative flex h-3 w-3">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
        </span>
      )}
      <span className="font-mono text-lg tabular-nums">
        {String(minutes).padStart(2, '0')}:{String(secs).padStart(2, '0')}
      </span>
    </div>
  )
}
