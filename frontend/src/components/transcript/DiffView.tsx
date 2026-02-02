"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import { diffTexts, diffStats, type DiffSegment } from "@/lib/diff"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"

export interface DiffViewProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 原始文本 */
  oldText: string
  /** 修改后文本 */
  newText: string
  /** 视图模式 */
  mode?: "inline" | "sideBySide"
  /** 是否显示统计 */
  showStats?: boolean
}

function DiffView({
  className,
  oldText,
  newText,
  mode = "inline",
  showStats = true,
  ...props
}: DiffViewProps) {
  const segments = React.useMemo(() => diffTexts(oldText, newText), [oldText, newText])
  const stats = React.useMemo(() => diffStats(segments), [segments])

  // 差异位置索引 (用于逐处跳转)
  const diffIndices = React.useMemo(() => {
    const indices: number[] = []
    segments.forEach((seg, i) => {
      if (seg.type !== "equal") {
        indices.push(i)
      }
    })
    return indices
  }, [segments])

  const [currentDiffIndex, setCurrentDiffIndex] = React.useState(0)
  const diffRefs = React.useRef<(HTMLSpanElement | null)[]>([])

  const goToDiff = (index: number) => {
    if (index >= 0 && index < diffIndices.length) {
      setCurrentDiffIndex(index)
      const segIndex = diffIndices[index]
      diffRefs.current[segIndex]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      })
    }
  }

  const renderSegment = (seg: DiffSegment, index: number) => {
    const isCurrent =
      diffIndices[currentDiffIndex] === index && seg.type !== "equal"

    switch (seg.type) {
      case "equal":
        return (
          <span key={index} ref={(el) => { diffRefs.current[index] = el }}>
            {seg.text}
          </span>
        )
      case "delete":
        return (
          <span
            key={index}
            ref={(el) => { diffRefs.current[index] = el }}
            className={cn(
              "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300 line-through",
              isCurrent && "ring-2 ring-red-500"
            )}
          >
            {seg.text}
          </span>
        )
      case "insert":
        return (
          <span
            key={index}
            ref={(el) => { diffRefs.current[index] = el }}
            className={cn(
              "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
              isCurrent && "ring-2 ring-green-500"
            )}
          >
            {seg.text}
          </span>
        )
    }
  }

  if (mode === "sideBySide") {
    // 并排模式
    const oldSegments = segments.filter((s) => s.type !== "insert")
    const newSegments = segments.filter((s) => s.type !== "delete")

    return (
      <div className={cn("space-y-3", className)} {...props}>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground">原始文本</p>
            <div className="p-3 rounded-lg border bg-muted/30 text-sm leading-relaxed">
              {oldSegments.map((seg, i) => (
                <span
                  key={i}
                  className={cn(
                    seg.type === "delete" &&
                      "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300 line-through"
                  )}
                >
                  {seg.text}
                </span>
              ))}
            </div>
          </div>
          <div className="space-y-1">
            <p className="text-xs font-medium text-muted-foreground">修改后文本</p>
            <div className="p-3 rounded-lg border bg-muted/30 text-sm leading-relaxed">
              {newSegments.map((seg, i) => (
                <span
                  key={i}
                  className={cn(
                    seg.type === "insert" &&
                      "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300"
                  )}
                >
                  {seg.text}
                </span>
              ))}
            </div>
          </div>
        </div>

        {showStats && <DiffStats stats={stats} />}
      </div>
    )
  }

  // 行内模式
  return (
    <div className={cn("space-y-3", className)} {...props}>
      <div className="p-4 rounded-lg border bg-muted/30 text-sm leading-relaxed">
        {segments.map((seg, i) => renderSegment(seg, i))}
      </div>

      {/* 导航和统计 */}
      {(showStats || diffIndices.length > 0) && (
        <div className="flex items-center justify-between">
          {showStats && <DiffStats stats={stats} />}

          {diffIndices.length > 0 && (
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => goToDiff(currentDiffIndex - 1)}
                disabled={currentDiffIndex === 0}
              >
                <ChevronLeft className="h-4 w-4" />
                上一处
              </Button>
              <span className="text-sm text-muted-foreground">
                {currentDiffIndex + 1}/{diffIndices.length}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() => goToDiff(currentDiffIndex + 1)}
                disabled={currentDiffIndex === diffIndices.length - 1}
              >
                下一处
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function DiffStats({
  stats,
}: {
  stats: ReturnType<typeof diffStats>
}) {
  return (
    <div className="flex items-center gap-2">
      {stats.insertions > 0 && (
        <Badge variant="outline" className="text-green-600 dark:text-green-400">
          +{stats.insertions}
        </Badge>
      )}
      {stats.deletions > 0 && (
        <Badge variant="outline" className="text-red-600 dark:text-red-400">
          -{stats.deletions}
        </Badge>
      )}
      <span className="text-xs text-muted-foreground">
        修改率: {stats.changeRate.toFixed(1)}%
      </span>
    </div>
  )
}

export { DiffView }
