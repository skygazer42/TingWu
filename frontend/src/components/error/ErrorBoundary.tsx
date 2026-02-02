"use client"

import * as React from "react"
import { AlertTriangle, RefreshCw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
}

interface ErrorBoundaryProps {
  children: React.ReactNode
  fallback?: React.ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
  onReset?: () => void
}

export class ErrorBoundary extends React.Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo)
    this.props.onError?.(error, errorInfo)
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null })
    this.props.onReset?.()
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }

      return (
        <ErrorFallback
          error={this.state.error}
          onReset={this.handleReset}
        />
      )
    }

    return this.props.children
  }
}

interface ErrorFallbackProps {
  error: Error | null
  onReset?: () => void
  title?: string
  description?: string
}

export function ErrorFallback({
  error,
  onReset,
  title = "出错了",
  description = "应用遇到了一个问题，请尝试刷新页面。",
}: ErrorFallbackProps) {
  return (
    <div className="flex items-center justify-center min-h-[400px] p-4">
      <Card className="max-w-md w-full">
        <CardHeader className="text-center">
          <div className="mx-auto mb-4 p-3 rounded-full bg-red-100 dark:bg-red-900/30 w-fit">
            <AlertTriangle className="h-8 w-8 text-red-600 dark:text-red-400" />
          </div>
          <CardTitle>{title}</CardTitle>
        </CardHeader>
        <CardContent className="text-center space-y-4">
          <p className="text-muted-foreground">{description}</p>

          {error && import.meta.env.DEV && (
            <details className="text-left text-sm bg-muted p-3 rounded-lg">
              <summary className="cursor-pointer font-medium mb-2">
                错误详情
              </summary>
              <pre className="whitespace-pre-wrap text-xs text-red-600 dark:text-red-400">
                {error.message}
              </pre>
              {error.stack && (
                <pre className="whitespace-pre-wrap text-xs text-muted-foreground mt-2 max-h-40 overflow-auto">
                  {error.stack}
                </pre>
              )}
            </details>
          )}

          <div className="flex justify-center gap-3">
            {onReset && (
              <Button onClick={onReset} variant="outline">
                <RefreshCw className="h-4 w-4 mr-2" />
                重试
              </Button>
            )}
            <Button onClick={() => window.location.reload()}>
              刷新页面
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

// 页面级错误边界
export function PageErrorBoundary({ children }: { children: React.ReactNode }) {
  return (
    <ErrorBoundary
      onError={(error) => {
        // 这里可以添加错误上报逻辑
        console.error("Page error:", error)
      }}
    >
      {children}
    </ErrorBoundary>
  )
}
