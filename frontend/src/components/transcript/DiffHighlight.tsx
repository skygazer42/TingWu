import { useMemo } from 'react'
import { cn } from '@/lib/utils'

interface DiffHighlightProps {
  original: string
  corrected: string
  className?: string
}

interface DiffSegment {
  text: string
  type: 'unchanged' | 'removed' | 'added'
}

export function DiffHighlight({ original, corrected, className }: DiffHighlightProps) {
  const segments = useMemo(() => {
    return computeDiff(original, corrected)
  }, [original, corrected])

  if (original === corrected) {
    return <span className={className}>{corrected}</span>
  }

  return (
    <span className={className}>
      {segments.map((segment, index) => (
        <span
          key={index}
          className={cn(
            segment.type === 'removed' && 'line-through text-red-500 bg-red-500/10',
            segment.type === 'added' && 'text-green-600 bg-green-500/10 font-medium'
          )}
        >
          {segment.text}
        </span>
      ))}
    </span>
  )
}

function computeDiff(original: string, corrected: string): DiffSegment[] {
  const segments: DiffSegment[] = []

  // 简单的字符级别 diff 算法
  // 使用最长公共子序列 (LCS) 思想
  const lcs = longestCommonSubsequence(original, corrected)

  let i = 0, j = 0, k = 0

  while (i < original.length || j < corrected.length) {
    if (k < lcs.length && i < original.length && original[i] === lcs[k]) {
      // 查找连续的公共部分
      const start = i
      while (k < lcs.length && i < original.length && j < corrected.length &&
             original[i] === lcs[k] && corrected[j] === lcs[k]) {
        i++
        j++
        k++
      }
      if (i > start) {
        segments.push({ text: original.slice(start, i), type: 'unchanged' })
      }
    } else {
      // 删除的部分
      const removedStart = i
      while (i < original.length && (k >= lcs.length || original[i] !== lcs[k])) {
        i++
      }
      if (i > removedStart) {
        segments.push({ text: original.slice(removedStart, i), type: 'removed' })
      }

      // 添加的部分
      const addedStart = j
      while (j < corrected.length && (k >= lcs.length || corrected[j] !== lcs[k])) {
        j++
      }
      if (j > addedStart) {
        segments.push({ text: corrected.slice(addedStart, j), type: 'added' })
      }
    }
  }

  return segments
}

function longestCommonSubsequence(a: string, b: string): string {
  const m = a.length
  const n = b.length

  // 优化：对于较长的字符串使用简单比较
  if (m > 1000 || n > 1000) {
    return simpleCommonChars(a, b)
  }

  const dp: number[][] = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0))

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i - 1] === b[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // 回溯构建 LCS
  let i = m, j = n
  const result: string[] = []
  while (i > 0 && j > 0) {
    if (a[i - 1] === b[j - 1]) {
      result.unshift(a[i - 1])
      i--
      j--
    } else if (dp[i - 1][j] > dp[i][j - 1]) {
      i--
    } else {
      j--
    }
  }

  return result.join('')
}

function simpleCommonChars(a: string, b: string): string {
  // 简单的贪心算法，适用于长字符串
  const result: string[] = []
  let j = 0
  for (let i = 0; i < a.length && j < b.length; i++) {
    if (a[i] === b[j]) {
      result.push(a[i])
      j++
    }
  }
  return result.join('')
}
