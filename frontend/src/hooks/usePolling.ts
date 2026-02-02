import { useState, useEffect, useRef, useCallback } from 'react'

interface UsePollingOptions<T> {
  /** 获取数据的函数 */
  fetcher: () => Promise<T>
  /** 轮询间隔 (毫秒) */
  interval: number
  /** 是否启用轮询 */
  enabled?: boolean
  /** 成功回调 */
  onSuccess?: (data: T) => void
  /** 错误回调 */
  onError?: (error: Error) => void
  /** 是否立即执行 */
  immediate?: boolean
}

interface UsePollingReturn<T> {
  data: T | null
  error: Error | null
  isLoading: boolean
  isFetching: boolean
  lastUpdated: Date | null
  refetch: () => Promise<void>
  start: () => void
  stop: () => void
}

export function usePolling<T>({
  fetcher,
  interval,
  enabled = true,
  onSuccess,
  onError,
  immediate = true,
}: UsePollingOptions<T>): UsePollingReturn<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isFetching, setIsFetching] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [isPolling, setIsPolling] = useState(enabled)

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fetcherRef = useRef(fetcher)
  const onSuccessRef = useRef(onSuccess)
  const onErrorRef = useRef(onError)

  // 更新 refs
  useEffect(() => {
    fetcherRef.current = fetcher
    onSuccessRef.current = onSuccess
    onErrorRef.current = onError
  })

  const fetchData = useCallback(async (isInitial = false) => {
    if (isInitial) {
      setIsLoading(true)
    }
    setIsFetching(true)
    setError(null)

    try {
      const result = await fetcherRef.current()
      setData(result)
      setLastUpdated(new Date())
      onSuccessRef.current?.(result)
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      setError(error)
      onErrorRef.current?.(error)
    } finally {
      setIsLoading(false)
      setIsFetching(false)
    }
  }, [])

  const start = useCallback(() => {
    setIsPolling(true)
  }, [])

  const stop = useCallback(() => {
    setIsPolling(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
  }, [])

  const refetch = useCallback(async () => {
    await fetchData(false)
  }, [fetchData])

  // 初始加载
  useEffect(() => {
    if (immediate && isPolling) {
      fetchData(true)
    }
  }, [fetchData, immediate, isPolling])

  // 轮询
  useEffect(() => {
    if (!isPolling || interval <= 0) {
      return
    }

    intervalRef.current = setInterval(() => {
      fetchData(false)
    }, interval)

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [fetchData, interval, isPolling])

  // 当 enabled 变化时更新 isPolling
  useEffect(() => {
    setIsPolling(enabled)
  }, [enabled])

  return {
    data,
    error,
    isLoading,
    isFetching,
    lastUpdated,
    refetch,
    start,
    stop,
  }
}
