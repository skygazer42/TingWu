import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Copy, Check, AlignLeft, List, Users } from 'lucide-react'
import { SentenceList } from './SentenceList'
import { SpeakerStats } from './SpeakerStats'
import { SpeakerBadge } from './SpeakerBadge'
import { ExportMenu } from './ExportMenu'
import { useTranscriptionStore } from '@/stores'
import type { TranscribeResponse, SentenceInfo } from '@/lib/api/types'

interface TranscriptViewProps {
  result: TranscribeResponse
  filename?: string
}

export function TranscriptView({ result, filename }: TranscriptViewProps) {
  const [copied, setCopied] = useState(false)
  const [showDiff, setShowDiff] = useState(false)
  const { setSelectedSentence } = useTranscriptionStore()

  const hasSpeakers = result.sentences.some(s => s.speaker_id !== undefined)
  const hasTurns = (result.speaker_turns?.length ?? 0) > 0
  const hasRawText = !!result.raw_text && result.raw_text !== result.text
  const bestPlainText = result.transcript || result.text_accu || result.text

  const handleCopy = async () => {
    await navigator.clipboard.writeText(bestPlainText || '')
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleSelectSentence = (sentence: SentenceInfo) => {
    setSelectedSentence(sentence)
  }

  return (
    <div className="space-y-4">
      {/* 工具栏 */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-4">
          {hasRawText && (
            <div className="flex items-center gap-2">
              <Switch
                id="show-diff"
                checked={showDiff}
                onCheckedChange={setShowDiff}
              />
              <Label htmlFor="show-diff" className="text-sm">
                显示纠错对比
              </Label>
            </div>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleCopy}>
            {copied ? (
              <>
                <Check className="h-4 w-4 mr-2" />
                已复制
              </>
            ) : (
              <>
                <Copy className="h-4 w-4 mr-2" />
                复制
              </>
            )}
          </Button>
          <ExportMenu result={result} filename={filename} />
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {/* 主内容区 */}
        <Card className="lg:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">转写结果</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue={hasTurns ? 'turns' : 'sentences'}>
              <TabsList className="mb-4">
                <TabsTrigger value="sentences" className="gap-2">
                  <List className="h-4 w-4" />
                  分句视图
                </TabsTrigger>
                {hasTurns && (
                  <TabsTrigger value="turns" className="gap-2">
                    <Users className="h-4 w-4" />
                    说话人段落
                  </TabsTrigger>
                )}
                <TabsTrigger value="full" className="gap-2">
                  <AlignLeft className="h-4 w-4" />
                  全文视图
                </TabsTrigger>
              </TabsList>

              <TabsContent value="sentences">
                <ScrollArea className="h-[400px] pr-4">
                  <SentenceList
                    sentences={result.sentences}
                    rawText={showDiff ? result.raw_text : undefined}
                    showDiff={showDiff}
                    showSpeaker={hasSpeakers}
                    onSelectSentence={handleSelectSentence}
                  />
                </ScrollArea>
              </TabsContent>

              {hasTurns && (
                <TabsContent value="turns">
                  <ScrollArea className="h-[400px] pr-4">
                    <div className="space-y-3">
                      {(result.speaker_turns || []).map((t, idx) => (
                        <div key={idx} className="rounded-md border p-3">
                          <div className="flex items-center justify-between gap-2">
                            <SpeakerBadge speaker={t.speaker} speakerId={t.speaker_id} />
                            <span className="text-xs text-muted-foreground tabular-nums">
                              {formatDuration(t.start)} - {formatDuration(t.end)}
                            </span>
                          </div>
                          <div className="mt-2 whitespace-pre-wrap text-sm leading-relaxed">
                            {t.text}
                          </div>
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                </TabsContent>
              )}

              <TabsContent value="full">
                <ScrollArea className="h-[400px]">
                  <div className="whitespace-pre-wrap text-sm leading-relaxed p-2">
                    {bestPlainText}
                  </div>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>

        {/* 侧边栏 */}
        <div className="space-y-4">
          {/* 说话人统计 */}
          {hasSpeakers && <SpeakerStats sentences={result.sentences} />}

          {/* 基本信息 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">基本信息</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">句子数</span>
                <span className="font-medium">{result.sentences.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">字符数</span>
                <span className="font-medium">{result.text.length}</span>
              </div>
              {result.sentences.length > 0 && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">总时长</span>
                  <span className="font-medium">
                    {formatDuration(result.sentences[result.sentences.length - 1].end)}
                  </span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function formatDuration(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${String(seconds).padStart(2, '0')}`
}
