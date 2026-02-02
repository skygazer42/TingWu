"use client"

import * as React from "react"
import { FileAudio, FileVideo, Clock, HardDrive, X, Play } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { AudioPlayer } from "@/components/audio/AudioPlayer"
import { isVideoFile } from "@/lib/api/transcribe"

export interface FileInfo {
  file: File
  duration?: number // 秒
  format?: string
  bitrate?: number
}

export interface FilePreviewProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 文件对象 */
  file: File
  /** 移除回调 */
  onRemove?: () => void
  /** 是否禁用移除 */
  removeDisabled?: boolean
  /** 是否显示播放器 */
  showPlayer?: boolean
  /** 上传进度 (0-100) */
  uploadProgress?: number
  /** 文件状态 */
  status?: "pending" | "uploading" | "processing" | "success" | "error"
  /** 错误信息 */
  error?: string
}

function FilePreview({
  className,
  file,
  onRemove,
  removeDisabled = false,
  showPlayer = true,
  uploadProgress,
  status = "pending",
  error,
  ...props
}: FilePreviewProps) {
  const [duration, setDuration] = React.useState<number | null>(null)
  const [showAudioPlayer, setShowAudioPlayer] = React.useState(false)

  const isVideo = isVideoFile(file)
  const Icon = isVideo ? FileVideo : FileAudio

  // 获取音频时长
  React.useEffect(() => {
    const url = URL.createObjectURL(file)
    const media = document.createElement(isVideo ? "video" : "audio")

    media.preload = "metadata"
    media.onloadedmetadata = () => {
      setDuration(media.duration)
      URL.revokeObjectURL(url)
    }
    media.onerror = () => {
      URL.revokeObjectURL(url)
    }
    media.src = url

    return () => {
      URL.revokeObjectURL(url)
    }
  }, [file, isVideo])

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 B"
    const k = 1024
    const sizes = ["B", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
  }

  const formatDuration = (seconds: number): string => {
    if (!isFinite(seconds)) return "--:--"
    const hrs = Math.floor(seconds / 3600)
    const mins = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)

    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
    }
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const getFileExtension = (): string => {
    const ext = file.name.split(".").pop()?.toUpperCase() || ""
    return ext
  }

  const statusColors = {
    pending: "border-muted",
    uploading: "border-primary",
    processing: "border-yellow-500",
    success: "border-green-500",
    error: "border-red-500",
  }

  return (
    <div
      className={cn(
        "rounded-lg border bg-card overflow-hidden",
        statusColors[status],
        className
      )}
      {...props}
    >
      <div className="p-4">
        <div className="flex items-start gap-3">
          {/* 图标 */}
          <div className="p-2 rounded-lg bg-muted shrink-0">
            <Icon className="h-6 w-6 text-muted-foreground" />
          </div>

          {/* 文件信息 */}
          <div className="flex-1 min-w-0 space-y-1">
            <p className="font-medium truncate">{file.name}</p>
            <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
              <span className="flex items-center gap-1">
                <HardDrive className="h-3 w-3" />
                {formatFileSize(file.size)}
              </span>
              {duration !== null && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {formatDuration(duration)}
                </span>
              )}
              <span>{getFileExtension()}</span>
            </div>
          </div>

          {/* 操作按钮 */}
          <div className="flex items-center gap-1 shrink-0">
            {showPlayer && !isVideo && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={() => setShowAudioPlayer(!showAudioPlayer)}
              >
                <Play className="h-4 w-4" />
              </Button>
            )}
            {onRemove && (
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={onRemove}
                disabled={removeDisabled}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* 上传进度 */}
        {status === "uploading" && uploadProgress !== undefined && (
          <div className="mt-3 space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-muted-foreground">上传中...</span>
              <span className="font-medium">{uploadProgress}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${uploadProgress}%` }}
              />
            </div>
          </div>
        )}

        {/* 处理中状态 */}
        {status === "processing" && (
          <div className="mt-3 flex items-center gap-2 text-xs text-yellow-600 dark:text-yellow-400">
            <div className="h-2 w-2 rounded-full bg-yellow-500 animate-pulse" />
            <span>{isVideo ? "提取音轨中..." : "转写中..."}</span>
          </div>
        )}

        {/* 错误信息 */}
        {status === "error" && error && (
          <div className="mt-3 text-xs text-red-500">{error}</div>
        )}
      </div>

      {/* 音频播放器 */}
      {showAudioPlayer && !isVideo && (
        <div className="border-t bg-muted/50 p-3">
          <AudioPlayer src={file} compact />
        </div>
      )}
    </div>
  )
}

export { FilePreview }
