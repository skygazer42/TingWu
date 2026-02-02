/**
 * 简易文本差异比较
 * Plan 11: 纠错对比可视化
 */

export interface DiffSegment {
  type: 'equal' | 'insert' | 'delete'
  text: string
}

/**
 * 简单字级别差异比较
 * 将两段文本按字符/词进行对比
 */
export function diffTexts(oldText: string, newText: string): DiffSegment[] {
  const oldChars = [...oldText]
  const newChars = [...newText]

  // 使用最长公共子序列 (LCS) 算法
  const m = oldChars.length
  const n = newChars.length

  // 动态规划表
  const dp: number[][] = Array.from({ length: m + 1 }, () =>
    Array(n + 1).fill(0)
  )

  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (oldChars[i - 1] === newChars[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1] + 1
      } else {
        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1])
      }
    }
  }

  // 回溯得到差异
  const result: DiffSegment[] = []
  let i = m
  let j = n

  const segments: Array<{ type: DiffSegment['type']; char: string }> = []

  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && oldChars[i - 1] === newChars[j - 1]) {
      segments.unshift({ type: 'equal', char: oldChars[i - 1] })
      i--
      j--
    } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
      segments.unshift({ type: 'insert', char: newChars[j - 1] })
      j--
    } else if (i > 0) {
      segments.unshift({ type: 'delete', char: oldChars[i - 1] })
      i--
    }
  }

  // 合并相邻的同类型段
  let currentSegment: DiffSegment | null = null

  for (const seg of segments) {
    if (currentSegment && currentSegment.type === seg.type) {
      currentSegment.text += seg.char
    } else {
      if (currentSegment) {
        result.push(currentSegment)
      }
      currentSegment = { type: seg.type, text: seg.char }
    }
  }

  if (currentSegment) {
    result.push(currentSegment)
  }

  return result
}

/**
 * 统计差异
 */
export function diffStats(segments: DiffSegment[]): {
  insertions: number
  deletions: number
  unchanged: number
  changeRate: number
} {
  let insertions = 0
  let deletions = 0
  let unchanged = 0

  for (const seg of segments) {
    switch (seg.type) {
      case 'insert':
        insertions += seg.text.length
        break
      case 'delete':
        deletions += seg.text.length
        break
      case 'equal':
        unchanged += seg.text.length
        break
    }
  }

  const total = insertions + deletions + unchanged
  const changeRate = total > 0 ? ((insertions + deletions) / total) * 100 : 0

  return { insertions, deletions, unchanged, changeRate }
}
