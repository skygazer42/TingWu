import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Download, FileText, Subtitles, FileJson } from 'lucide-react'
import type { TranscribeResponse } from '@/lib/api/types'

interface ExportMenuProps {
  result: TranscribeResponse
  filename?: string
}

export function ExportMenu({ result, filename = 'transcript' }: ExportMenuProps) {
  const handleExportTxt = () => {
    const content = result.transcript || result.text
    downloadFile(`${filename}.txt`, content, 'text/plain')
  }

  const handleExportSrt = () => {
    const srt = generateSrt(result)
    downloadFile(`${filename}.srt`, srt, 'text/plain')
  }

  const handleExportJson = () => {
    const json = JSON.stringify(result, null, 2)
    downloadFile(`${filename}.json`, json, 'application/json')
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          导出
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem onClick={handleExportTxt}>
          <FileText className="h-4 w-4 mr-2" />
          纯文本 (.txt)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleExportSrt}>
          <Subtitles className="h-4 w-4 mr-2" />
          字幕文件 (.srt)
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleExportJson}>
          <FileJson className="h-4 w-4 mr-2" />
          JSON 数据 (.json)
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

function downloadFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function generateSrt(result: TranscribeResponse): string {
  const lines: string[] = []

  result.sentences.forEach((sentence, index) => {
    lines.push(String(index + 1))
    lines.push(`${formatSrtTime(sentence.start)} --> ${formatSrtTime(sentence.end)}`)

    // 添加说话人标签（如果有）
    const speakerPrefix = sentence.speaker
      ? `[${sentence.speaker}] `
      : sentence.speaker_id !== undefined
        ? `[说话人${String.fromCharCode(65 + sentence.speaker_id)}] `
        : ''

    lines.push(`${speakerPrefix}${sentence.text}`)
    lines.push('')
  })

  return lines.join('\n')
}

function formatSrtTime(ms: number): string {
  const hours = Math.floor(ms / 3600000)
  const minutes = Math.floor((ms % 3600000) / 60000)
  const seconds = Math.floor((ms % 60000) / 1000)
  const milliseconds = ms % 1000

  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`
}
