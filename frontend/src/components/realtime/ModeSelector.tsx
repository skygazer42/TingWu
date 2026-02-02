import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Radio } from 'lucide-react'

type RealtimeMode = '2pass' | 'online' | 'offline'

interface ModeSelectorProps {
  value: RealtimeMode
  onChange: (mode: RealtimeMode) => void
  disabled?: boolean
}

export function ModeSelector({ value, onChange, disabled }: ModeSelectorProps) {
  return (
    <div className="space-y-2">
      <Label htmlFor="realtime-mode" className="flex items-center gap-2">
        <Radio className="h-4 w-4" />
        识别模式
      </Label>
      <Select
        value={value}
        onValueChange={(v) => onChange(v as RealtimeMode)}
        disabled={disabled}
      >
        <SelectTrigger id="realtime-mode">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="2pass">
            <div>
              <div className="font-medium">双通道 (推荐)</div>
              <div className="text-xs text-muted-foreground">
                实时预览 + 最终纠正
              </div>
            </div>
          </SelectItem>
          <SelectItem value="online">
            <div>
              <div className="font-medium">实时模式</div>
              <div className="text-xs text-muted-foreground">
                低延迟，适合实时字幕
              </div>
            </div>
          </SelectItem>
          <SelectItem value="offline">
            <div>
              <div className="font-medium">离线模式</div>
              <div className="text-xs text-muted-foreground">
                说完后识别，精度更高
              </div>
            </div>
          </SelectItem>
        </SelectContent>
      </Select>
    </div>
  )
}
