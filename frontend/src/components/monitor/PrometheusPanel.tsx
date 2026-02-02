"use client"

import * as React from "react"
import { ExternalLink, Copy, Download, Code } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { toast } from "@/lib/toast"

export interface PrometheusPanelProps extends React.HTMLAttributes<HTMLDivElement> {
  /** Prometheus 端点 URL */
  endpoint?: string
  /** 原始指标数据 */
  metricsData?: string | null
  /** 是否加载中 */
  isLoading?: boolean
}

const PROMETHEUS_CONFIG = `scrape_configs:
  - job_name: 'tingwu'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics/prometheus'`

const GRAFANA_DASHBOARD = {
  dashboard: {
    title: "TingWu ASR Monitor",
    panels: [
      {
        title: "Total Requests",
        type: "stat",
        targets: [{ expr: 'tingwu_total_requests' }],
      },
      {
        title: "Success Rate",
        type: "gauge",
        targets: [{ expr: 'tingwu_successful_requests / tingwu_total_requests * 100' }],
      },
      {
        title: "Average RTF",
        type: "timeseries",
        targets: [{ expr: 'tingwu_avg_rtf' }],
      },
    ],
  },
}

function PrometheusPanel({
  className,
  endpoint = '/metrics/prometheus',
  metricsData,
  isLoading = false,
  ...props
}: PrometheusPanelProps) {
  const fullEndpoint = `${window.location.origin}${endpoint}`

  const handleCopyEndpoint = () => {
    navigator.clipboard.writeText(fullEndpoint)
    toast.success('端点链接已复制')
  }

  const handleCopyConfig = () => {
    navigator.clipboard.writeText(PROMETHEUS_CONFIG)
    toast.success('Prometheus 配置已复制')
  }

  const handleDownloadDashboard = () => {
    const json = JSON.stringify(GRAFANA_DASHBOARD, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'tingwu-grafana-dashboard.json'
    a.click()
    URL.revokeObjectURL(url)
    toast.success('Grafana Dashboard 已下载')
  }

  const handleOpenEndpoint = () => {
    window.open(fullEndpoint, '_blank')
  }

  return (
    <Card className={className} {...props}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Prometheus 集成</CardTitle>
            <CardDescription>监控指标导出和集成配置</CardDescription>
          </div>
          <Badge variant="outline">OpenMetrics</Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 端点信息 */}
        <div className="space-y-2">
          <label className="text-sm font-medium">端点地址</label>
          <div className="flex items-center gap-2 p-3 rounded-lg bg-muted font-mono text-sm">
            <code className="flex-1 truncate">{fullEndpoint}</code>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={handleCopyEndpoint}
            >
              <Copy className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 shrink-0"
              onClick={handleOpenEndpoint}
            >
              <ExternalLink className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Prometheus 配置 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Prometheus 配置</label>
            <Button variant="ghost" size="sm" onClick={handleCopyConfig}>
              <Copy className="h-4 w-4 mr-1" />
              复制
            </Button>
          </div>
          <pre className="p-3 rounded-lg bg-muted text-xs font-mono overflow-x-auto">
            {PROMETHEUS_CONFIG}
          </pre>
        </div>

        {/* Grafana Dashboard */}
        <div className="flex items-center justify-between p-3 rounded-lg border">
          <div className="flex items-center gap-3">
            <Code className="h-5 w-5 text-muted-foreground" />
            <div>
              <p className="text-sm font-medium">Grafana Dashboard</p>
              <p className="text-xs text-muted-foreground">
                预配置的仪表盘模板
              </p>
            </div>
          </div>
          <Button variant="outline" size="sm" onClick={handleDownloadDashboard}>
            <Download className="h-4 w-4 mr-1" />
            下载
          </Button>
        </div>

        {/* 指标预览 */}
        {metricsData && (
          <div className="space-y-2">
            <label className="text-sm font-medium">指标预览</label>
            <pre className={cn(
              "p-3 rounded-lg bg-muted text-xs font-mono overflow-auto max-h-[200px]",
              isLoading && "animate-pulse"
            )}>
              {metricsData}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export { PrometheusPanel }
