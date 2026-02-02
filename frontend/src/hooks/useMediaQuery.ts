import { useSyncExternalStore } from 'react'

type MediaQueryString = string

export function useMediaQuery(query: MediaQueryString): boolean {
  const subscribe = (onStoreChange: () => void) => {
    if (typeof window === 'undefined') return () => {}
    const mediaQuery = window.matchMedia(query)
    const handler = () => onStoreChange()
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }

  const getSnapshot = () => {
    if (typeof window === 'undefined') return false
    return window.matchMedia(query).matches
  }

  return useSyncExternalStore(subscribe, getSnapshot, () => false)
}

// 常用断点
export function useIsMobile() {
  return useMediaQuery('(max-width: 767px)')
}

export function useIsTablet() {
  return useMediaQuery('(min-width: 768px) and (max-width: 1023px)')
}

export function useIsDesktop() {
  return useMediaQuery('(min-width: 1024px)')
}
