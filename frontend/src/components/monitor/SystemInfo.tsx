"use client"

import * as React from "react"
import {
  Activity,
  Clock,
  CheckCircle2,
  XCircle,
  Zap,
  HardDrive,
  Cpu,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import type { HealthResponse, MetricsResponse } from "@/lib/api/types"

export interface SystemInfoProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 健康状态 */
  health?: HealthResponse | null
  /** 指标数据 */
  metrics?: MetricsResponse | null
  /** 是否加载中 */
  isLoading?: boolean
}

function formatUptime(seconds: number): string {
  const days = Math.floor(seconds / 86400)
  const hours = Math.floor((seconds % 86400) / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)

  const parts: string[] = []
  if (days > 0) parts.push(`${days}天`)
  if (hours > 0) parts.push(`${hours}小时`)
  if (minutes > 0) parts.push(`${minutes}分钟`)

  return parts.length > 0 ? parts.join(' ') : '刚刚启动'
}

function formatAudioTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)

  if (hours > 0) {
    return `${hours}小时${minutes}分钟`
  }
  return `${minutes}分钟`
}

function SystemInfo({
  className,
  health,
  metrics,
  isLoading = false,
  ...props
}: SystemInfoProps) {
  const isHealthy = health?.status === 'ok'
  const successRate = metrics
    ? ((metrics.successful_requests / metrics.total_requests) * 100).toFixed(1)
    : '0'

  const stats = [
    {
      label: '服务状态',
      value: isHealthy ? '正常' : '异常',
      icon: isHealthy ? CheckCircle2 : XCircle,
      color: isHealthy ? 'text-green-500' : 'text-red-500',
    },
    {
      label: '版本',
      value: health?.version || '-',
      icon: Activity,
      color: 'text-primary',
    },
    {
      label: '运行时长',
      value: metrics ? formatUptime(metrics.uptime_seconds) : '-',
      icon: Clock,
      color: 'text-blue-500',
    },
    {
      label: '总请求',
      value: metrics?.total_requests.toLocaleString() || '0',
      icon: Zap,
      color: 'text-yellow-500',
    },
    {
      label: '成功率',
      value: `${successRate}%`,
      icon: CheckCircle2,
      color: 'text-green-500',
    },
    {
      label: '处理音频',
      value: metrics ? formatAudioTime(metrics.total_audio_seconds) : '-',
      icon: HardDrive,
      color: 'text-purple-500',
    },
    {
      label: '平均 RTF',
      value: metrics?.avg_rtf?.toFixed(3) || '-',
      icon: Cpu,
      color: 'text-orange-500',
    },
  ]

  return (
    <Card className={className} {...props}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">系统信息</CardTitle>
          <Badge variant={isHealthy ? 'default' : 'destructive'}>
            {isHealthy ? '在线' : '离线'}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {stats.map((stat, index) => {
            const Icon = stat.icon
            return (
              <div
                key={index}
                className={cn(
                  'flex items-center gap-3 p-3 rounded-lg bg-muted/50',
                  isLoading && 'animate-pulse'
                )}
              >
                <div className={cn('p-2 rounded-lg bg-background', stat.color)}>
                  <Icon className="h-4 w-4" />
                </div>
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground">{stat.label}</p>
                  <p className="font-medium truncate">{stat.value}</p>
                </div>
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

// 请求统计卡片
export interface RequestStatsProps extends React.HTMLAttributes<HTMLDivElement> {
  metrics?: MetricsResponse | null
}

function RequestStats({ className, metrics, ...props }: RequestStatsProps) {
  const total = metrics?.total_requests || 0
  const success = metrics?.successful_requests || 0
  const failed = metrics?.failed_requests || 0
  const successRate = total > 0 ? (success / total) * 100 : 0
  const failRate = total > 0 ? (failed / total) * 100 : 0

  return (
    <Card className={className} {...props}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">请求统计</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold">{total.toLocaleString()}</p>
            <p className="text-xs text-muted-foreground">总请求</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-green-500">
              {success.toLocaleString()}
            </p>
            <p className="text-xs text-muted-foreground">成功</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-red-500">
              {failed.toLocaleString()}
            </p>
            <p className="text-xs text-muted-foreground">失败</p>
          </div>
        </div>

        {/* 进度条 */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">成功率</span>
            <span className="font-medium">{successRate.toFixed(1)}%</span>
          </div>
          <div className="h-2 rounded-full bg-muted overflow-hidden flex">
            <div
              className="bg-green-500 transition-all"
              style={{ width: `${successRate}%` }}
            />
            <div
              className="bg-red-500 transition-all"
              style={{ width: `${failRate}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

// LLM 缓存统计
export interface LLMCacheStatsProps extends React.HTMLAttributes<HTMLDivElement> {
  cacheStats?: Record<string, unknown> | null
}

function LLMCacheStats({ className, cacheStats, ...props }: LLMCacheStatsProps) {
  if (!cacheStats) return null

  const hits = (cacheStats.hits as number) || 0
  const misses = (cacheStats.misses as number) || 0
  const total = hits + misses
  const hitRate = total > 0 ? (hits / total) * 100 : 0

  return (
    <Card className={className} {...props}>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">LLM 缓存</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-green-500">{hits}</p>
            <p className="text-xs text-muted-foreground">命中</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-yellow-500">{misses}</p>
            <p className="text-xs text-muted-foreground">未命中</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{hitRate.toFixed(1)}%</p>
            <p className="text-xs text-muted-foreground">命中率</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export { SystemInfo, RequestStats, LLMCacheStats }
