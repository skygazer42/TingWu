"use client"

import * as React from "react"
import {
  CheckCircle2,
  XCircle,
  Loader2,
  Download,
  Trash2,
  RotateCcw,
  ChevronDown,
  FileDown,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { EmptyStateNoData } from "@/components/ui/empty-state"
import type { BatchTranscribeItem } from "@/lib/api/types"

export interface BatchResultItem extends BatchTranscribeItem {
  selected?: boolean
  isRetrying?: boolean
}

export interface BatchResultManagerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "results"> {
  /** 结果列表 */
  results: BatchResultItem[]
  /** 总数 */
  total: number
  /** 已完成数 */
  completedCount: number
  /** 进行中数 */
  processingCount?: number
  /** 查看结果 */
  onViewResult?: (item: BatchResultItem) => void
  /** 导出结果 */
  onExport?: (item: BatchResultItem) => void
  /** 批量导出 */
  onExportSelected?: (items: BatchResultItem[]) => void
  /** 导出全部 */
  onExportAll?: () => void
  /** 删除 */
  onRemove?: (index: number) => void
  /** 重试 */
  onRetry?: (item: BatchResultItem) => void
  /** 选中变化 */
  onSelectionChange?: (indices: number[]) => void
  /** 是否正在处理 */
  isProcessing?: boolean
  /** 上传进度 */
  uploadProgress?: number
}

function BatchResultManager({
  className,
  results,
  total,
  completedCount,
  processingCount = 0,
  onViewResult,
  onExport,
  onExportSelected,
  onExportAll,
  onRemove,
  onRetry,
  onSelectionChange,
  isProcessing = false,
  uploadProgress,
  ...props
}: BatchResultManagerProps) {
  const [selectedIndices, setSelectedIndices] = React.useState<Set<number>>(new Set())
  const [expandedIndex, setExpandedIndex] = React.useState<number | null>(null)

  const successCount = results.filter((r) => r.success).length
  const failedCount = results.filter((r) => !r.success && r.error).length
  const pendingCount = results.filter((r) => !r.success && !r.error).length

  const handleSelectItem = (index: number, checked: boolean) => {
    const newSelected = new Set(selectedIndices)
    if (checked) {
      newSelected.add(index)
    } else {
      newSelected.delete(index)
    }
    setSelectedIndices(newSelected)
    onSelectionChange?.(Array.from(newSelected))
  }

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allIndices = results.map((_, i) => i)
      setSelectedIndices(new Set(allIndices))
      onSelectionChange?.(allIndices)
    } else {
      setSelectedIndices(new Set())
      onSelectionChange?.([])
    }
  }

  const handleExportSelected = () => {
    const selectedItems = results.filter((_, i) => selectedIndices.has(i))
    onExportSelected?.(selectedItems)
  }

  const progress = total > 0 ? Math.round((completedCount / total) * 100) : 0
  const allSelected = results.length > 0 && selectedIndices.size === results.length

  if (results.length === 0 && !isProcessing) {
    return null
  }

  return (
    <Card className={className} {...props}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">
            批量转写结果 ({completedCount}/{total})
          </CardTitle>
          <div className="flex items-center gap-2">
            {successCount > 0 && (
              <Badge variant="outline" className="text-green-600">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                {successCount}
              </Badge>
            )}
            {failedCount > 0 && (
              <Badge variant="outline" className="text-red-600">
                <XCircle className="h-3 w-3 mr-1" />
                {failedCount}
              </Badge>
            )}
            {(pendingCount > 0 || processingCount > 0) && (
              <Badge variant="outline">
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                {pendingCount + processingCount}
              </Badge>
            )}
          </div>
        </div>

        {/* 进度条 */}
        {(isProcessing || uploadProgress !== undefined) && (
          <div className="space-y-1 mt-2">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {uploadProgress !== undefined && uploadProgress < 100
                  ? "上传中..."
                  : "处理中..."}
              </span>
              <span>{uploadProgress ?? progress}%</span>
            </div>
            <Progress value={uploadProgress ?? progress} />
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-2">
        {/* 工具栏 */}
        {results.length > 0 && (
          <div className="flex items-center justify-between pb-2 border-b">
            <div className="flex items-center gap-2">
              <Checkbox
                checked={allSelected}
                onCheckedChange={handleSelectAll}
              />
              <span className="text-sm text-muted-foreground">
                {selectedIndices.size > 0
                  ? `已选 ${selectedIndices.size} 项`
                  : "全选"}
              </span>
            </div>
            <div className="flex items-center gap-2">
              {selectedIndices.size > 0 && onExportSelected && (
                <Button variant="outline" size="sm" onClick={handleExportSelected}>
                  <FileDown className="h-4 w-4 mr-1" />
                  导出选中
                </Button>
              )}
              {onExportAll && successCount > 0 && (
                <Button variant="outline" size="sm" onClick={onExportAll}>
                  <Download className="h-4 w-4 mr-1" />
                  导出全部
                </Button>
              )}
            </div>
          </div>
        )}

        {/* 结果列表 */}
        {results.length === 0 ? (
          <EmptyStateNoData
            title="正在处理"
            description="批量转写结果将显示在这里"
            size="sm"
          />
        ) : (
          <div className="space-y-2">
            {results.map((item, index) => (
              <Collapsible
                key={index}
                open={expandedIndex === index}
                onOpenChange={(open) => setExpandedIndex(open ? index : null)}
              >
                <div
                  className={cn(
                    "rounded-lg border bg-card overflow-hidden",
                    item.success && "border-green-200 dark:border-green-800",
                    !item.success && item.error && "border-red-200 dark:border-red-800"
                  )}
                >
                  <div className="flex items-center gap-3 p-3">
                    {/* 选择框 */}
                    <Checkbox
                      checked={selectedIndices.has(index)}
                      onCheckedChange={(checked) =>
                        handleSelectItem(index, checked === true)
                      }
                    />

                    {/* 状态图标 */}
                    {item.success ? (
                      <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0" />
                    ) : item.error ? (
                      <XCircle className="h-5 w-5 text-red-500 shrink-0" />
                    ) : (
                      <Loader2 className="h-5 w-5 text-muted-foreground animate-spin shrink-0" />
                    )}

                    {/* 文件名 */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{item.filename}</p>
                      {item.error && (
                        <p className="text-xs text-red-500 truncate">{item.error}</p>
                      )}
                      {item.success && item.result && (
                        <p className="text-xs text-muted-foreground truncate">
                          {item.result.text.slice(0, 50)}
                          {item.result.text.length > 50 ? "..." : ""}
                        </p>
                      )}
                    </div>

                    {/* 操作按钮 */}
                    <div className="flex items-center gap-1 shrink-0">
                      {item.success && onViewResult && (
                        <CollapsibleTrigger asChild>
                          <Button variant="ghost" size="icon" className="h-8 w-8">
                            <ChevronDown
                              className={cn(
                                "h-4 w-4 transition-transform",
                                expandedIndex === index && "rotate-180"
                              )}
                            />
                          </Button>
                        </CollapsibleTrigger>
                      )}

                      {item.success && onExport && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => onExport(item)}
                        >
                          <Download className="h-4 w-4" />
                        </Button>
                      )}

                      {!item.success && item.error && onRetry && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8"
                          onClick={() => onRetry(item)}
                          disabled={item.isRetrying}
                        >
                          <RotateCcw
                            className={cn(
                              "h-4 w-4",
                              item.isRetrying && "animate-spin"
                            )}
                          />
                        </Button>
                      )}

                      {onRemove && (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-muted-foreground hover:text-destructive"
                          onClick={() => onRemove(index)}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* 展开内容 */}
                  {item.success && item.result && (
                    <CollapsibleContent>
                      <div className="border-t px-3 py-3 bg-muted/30">
                        <div className="text-sm whitespace-pre-wrap">
                          {item.result.text}
                        </div>
                        {item.result.sentences && item.result.sentences.length > 0 && (
                          <div className="mt-2 pt-2 border-t text-xs text-muted-foreground">
                            {item.result.sentences.length} 句话
                          </div>
                        )}
                      </div>
                    </CollapsibleContent>
                  )}
                </div>
              </Collapsible>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export { BatchResultManager }
