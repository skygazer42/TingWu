import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useTranscriptionStore } from '@/stores'
import { Sparkles, Users, BookText, Bot } from 'lucide-react'

export function TranscribeOptions() {
  const { options, setOptions, tempHotwords, setTempHotwords } = useTranscriptionStore()

  return (
    <Card>
      <CardHeader>
        <CardTitle>转写选项</CardTitle>
        <CardDescription>配置转写参数</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* 说话人识别 */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Users className="h-5 w-5 text-muted-foreground" />
            <div>
              <Label htmlFor="speaker" className="text-base">说话人识别</Label>
              <p className="text-sm text-muted-foreground">区分不同说话人</p>
            </div>
          </div>
          <Switch
            id="speaker"
            checked={options.with_speaker}
            onCheckedChange={(checked) => setOptions({ with_speaker: checked })}
          />
        </div>

        {/* 热词纠错 */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <BookText className="h-5 w-5 text-muted-foreground" />
            <div>
              <Label htmlFor="hotword" className="text-base">热词纠错</Label>
              <p className="text-sm text-muted-foreground">使用热词库进行纠错</p>
            </div>
          </div>
          <Switch
            id="hotword"
            checked={options.apply_hotword}
            onCheckedChange={(checked) => setOptions({ apply_hotword: checked })}
          />
        </div>

        {/* LLM 润色 */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Sparkles className="h-5 w-5 text-muted-foreground" />
              <div>
                <Label htmlFor="llm" className="text-base">LLM 润色</Label>
                <p className="text-sm text-muted-foreground">使用大语言模型优化文本</p>
              </div>
            </div>
            <Switch
              id="llm"
              checked={options.apply_llm}
              onCheckedChange={(checked) => setOptions({ apply_llm: checked })}
            />
          </div>

          {options.apply_llm && (
            <div className="ml-8 space-y-2">
              <Label htmlFor="llm-role" className="flex items-center gap-2">
                <Bot className="h-4 w-4" />
                LLM 角色
              </Label>
              <Select
                value={options.llm_role}
                onValueChange={(value) => setOptions({ llm_role: value as typeof options.llm_role })}
              >
                <SelectTrigger id="llm-role">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">默认 (通用纠错)</SelectItem>
                  <SelectItem value="translator">翻译助手</SelectItem>
                  <SelectItem value="code">代码助手</SelectItem>
                  <SelectItem value="corrector">专业校对</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}
        </div>

        {/* 临时热词 */}
        <div className="space-y-2">
          <Label htmlFor="temp-hotwords">临时热词</Label>
          <Textarea
            id="temp-hotwords"
            placeholder="输入临时热词，用空格分隔..."
            value={tempHotwords}
            onChange={(e) => setTempHotwords(e.target.value)}
            className="min-h-[80px] resize-none"
          />
          <p className="text-xs text-muted-foreground">
            临时热词仅对本次转写有效，不会保存到热词库
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
