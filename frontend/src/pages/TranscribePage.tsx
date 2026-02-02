import { useState } from 'react'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Loader2, Play, FileAudio, Link } from 'lucide-react'
import { FileDropzone } from '@/components/upload'
import { TranscribeOptions } from '@/components/transcribe'
import { TranscriptView } from '@/components/transcript'
import { Timeline } from '@/components/timeline'
import { HistoryList } from '@/components/history/HistoryList'
import { UrlTranscribe } from '@/components/url/UrlTranscribe'
import { useTranscriptionStore } from '@/stores'
import { useHistoryStore } from '@/stores/historyStore'
import { transcribeAudio, transcribeBatch } from '@/lib/api'
import type { SentenceInfo } from '@/lib/api/types'

export default function TranscribePage() {
  const {
    files,
    options,
    tempHotwords,
    result,
    setResult,
    isTranscribing,
    setTranscribing,
    clearFiles,
    setSelectedSentence,
  } = useTranscriptionStore()

  const { addItem } = useHistoryStore()

  const [selectedIndex, setSelectedIndex] = useState<number>()
  const [inputMode, setInputMode] = useState<string>('file')

  const handleTranscribe = async () => {
    if (files.length === 0) {
      toast.error('请先上传音频文件')
      return
    }

    setTranscribing(true)
    setResult(null)

    try {
      const transcribeOptions = {
        ...options,
        hotwords: tempHotwords || undefined,
      }

      if (files.length === 1) {
        // 单文件转写
        const response = await transcribeAudio(files[0], transcribeOptions)
        if (response.code === 0) {
          setResult(response)
          // 保存到历史记录
          addItem({
            filename: files[0].name,
            text: response.text,
            sentences: response.sentences,
            rawText: response.raw_text,
            options: {
              withSpeaker: options.with_speaker,
              applyHotword: options.apply_hotword,
              applyLlm: options.apply_llm,
              llmRole: options.llm_role,
            },
          })
          toast.success('转写完成')
        } else {
          toast.error('转写失败')
        }
      } else {
        // 批量转写
        const response = await transcribeBatch(files, transcribeOptions)
        if (response.success_count > 0) {
          // 显示第一个成功的结果
          const firstSuccess = response.results.find(r => r.success && r.result)
          if (firstSuccess?.result) {
            setResult(firstSuccess.result)
          }
          // 保存批量结果到历史
          response.results.forEach((r, idx) => {
            if (r.success && r.result) {
              addItem({
                filename: files[idx]?.name ?? `文件${idx + 1}`,
                text: r.result.text,
                sentences: r.result.sentences,
                rawText: r.result.raw_text,
                options: {
                  withSpeaker: options.with_speaker,
                  applyHotword: options.apply_hotword,
                  applyLlm: options.apply_llm,
                },
              })
            }
          })
          toast.success(`转写完成: ${response.success_count}/${response.total} 成功`)
        } else {
          toast.error('所有文件转写失败')
        }
      }
    } catch (error) {
      console.error('Transcription error:', error)
      toast.error('转写请求失败，请检查服务连接')
    } finally {
      setTranscribing(false)
    }
  }

  const handleUrlSubmit = async (url: string) => {
    toast.info(`URL 转写已提交: ${url}`)
  }

  const handleSelectSentence = (sentence: SentenceInfo, index: number) => {
    setSelectedSentence(sentence)
    setSelectedIndex(index)
  }

  const handleClear = () => {
    clearFiles()
    setResult(null)
    setSelectedSentence(null)
    setSelectedIndex(undefined)
  }

  const handleViewHistoryItem = (item: { text: string; sentences: SentenceInfo[] }) => {
    setResult({ ...item, code: 0, raw_text: item.text })
    toast.success('已加载历史记录')
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">音频转写</h1>
        <p className="text-muted-foreground">上传音频文件进行语音识别和转写</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* 输入区域 - 文件上传 / URL 输入 */}
        <Card>
          <CardHeader className="pb-3">
            <Tabs value={inputMode} onValueChange={setInputMode}>
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="file">
                  <FileAudio className="h-4 w-4 mr-1" />
                  文件上传
                </TabsTrigger>
                <TabsTrigger value="url">
                  <Link className="h-4 w-4 mr-1" />
                  URL转写
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </CardHeader>
          <CardContent className="space-y-4">
            {inputMode === 'file' ? (
              <>
                <FileDropzone />
                <div className="flex gap-2">
                  <Button
                    className="flex-1"
                    onClick={handleTranscribe}
                    disabled={files.length === 0 || isTranscribing}
                  >
                    {isTranscribing ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        转写中...
                      </>
                    ) : (
                      <>
                        <Play className="h-4 w-4 mr-2" />
                        开始转写
                      </>
                    )}
                  </Button>
                  {files.length > 0 && !isTranscribing && (
                    <Button variant="outline" onClick={handleClear}>
                      清空
                    </Button>
                  )}
                </div>
              </>
            ) : (
              <UrlTranscribe onSubmit={handleUrlSubmit} />
            )}
          </CardContent>
        </Card>

        {/* 转写选项 */}
        <TranscribeOptions />
      </div>

      {/* 时间轴 */}
      {result && result.sentences.length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">时间轴</CardTitle>
          </CardHeader>
          <CardContent>
            <Timeline
              sentences={result.sentences}
              selectedIndex={selectedIndex}
              onSelectSentence={handleSelectSentence}
            />
          </CardContent>
        </Card>
      )}

      {/* 转写结果 */}
      {result && (
        <TranscriptView
          result={result}
          filename={files[0]?.name?.replace(/\.[^/.]+$/, '')}
        />
      )}

      {/* 空状态 */}
      {!result && !isTranscribing && (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <p className="text-muted-foreground text-center">
              上传音频文件并点击"开始转写"按钮，转写结果将显示在这里
            </p>
          </CardContent>
        </Card>
      )}

      {/* 转写历史 */}
      <HistoryList onViewResult={handleViewHistoryItem} />
    </div>
  )
}
