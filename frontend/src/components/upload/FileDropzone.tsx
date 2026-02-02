import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { cn } from '@/lib/utils'
import { Upload, File, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { useTranscriptionStore } from '@/stores'

const ACCEPTED_TYPES = {
  'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'],
  'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm'],
}

interface FileDropzoneProps {
  onFilesSelected?: (files: File[]) => void
  multiple?: boolean
  disabled?: boolean
  uploadProgress?: number
}

export function FileDropzone({
  onFilesSelected,
  multiple = true,
  disabled = false,
  uploadProgress,
}: FileDropzoneProps) {
  const { files, addFiles, removeFile, isTranscribing } = useTranscriptionStore()

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (multiple) {
        addFiles(acceptedFiles)
      } else {
        addFiles([acceptedFiles[0]])
      }
      onFilesSelected?.(acceptedFiles)
    },
    [addFiles, multiple, onFilesSelected]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    multiple,
    disabled: disabled || isTranscribing,
  })

  const isUploading = uploadProgress !== undefined && uploadProgress < 100

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={cn(
          'flex flex-col items-center justify-center gap-4 p-8 border-2 border-dashed rounded-lg cursor-pointer transition-colors',
          isDragActive && 'border-primary bg-primary/5',
          !isDragActive && 'border-muted-foreground/25 hover:border-primary/50',
          (disabled || isTranscribing) && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input {...getInputProps()} />
        <div className={cn(
          'p-4 rounded-full',
          isDragActive ? 'bg-primary/10' : 'bg-muted'
        )}>
          <Upload className={cn(
            'h-8 w-8',
            isDragActive ? 'text-primary' : 'text-muted-foreground'
          )} />
        </div>
        <div className="text-center">
          {isDragActive ? (
            <p className="text-primary font-medium">释放文件开始上传</p>
          ) : (
            <>
              <p className="font-medium">拖拽文件到此处或点击上传</p>
              <p className="text-sm text-muted-foreground mt-1">
                支持 WAV, MP3, M4A, FLAC, OGG, MP4 等格式
              </p>
            </>
          )}
        </div>
      </div>

      {/* 文件列表 */}
      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${index}`}
              className="flex items-center gap-3 p-3 rounded-lg bg-muted/50"
            >
              <File className="h-5 w-5 text-muted-foreground shrink-0" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">{file.name}</p>
                <p className="text-xs text-muted-foreground">
                  {formatFileSize(file.size)}
                </p>
              </div>
              {!isTranscribing && (
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8 shrink-0"
                  onClick={() => removeFile(index)}
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* 上传进度 */}
      {isUploading && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>上传中...</span>
            <span>{uploadProgress}%</span>
          </div>
          <Progress value={uploadProgress} />
        </div>
      )}
    </div>
  )
}

function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
}
