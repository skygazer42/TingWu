import * as React from "react"
import { cn } from "@/lib/utils"

export interface CircularProgressProps extends React.SVGAttributes<SVGSVGElement> {
  /** 进度值 (0-100) */
  value?: number
  /** 尺寸 */
  size?: "xs" | "sm" | "md" | "lg" | "xl" | number
  /** 圆环宽度 */
  strokeWidth?: number
  /** 是否显示百分比文本 */
  showValue?: boolean
  /** 自定义中心内容 */
  label?: React.ReactNode
  /** 颜色 */
  color?: "primary" | "secondary" | "success" | "warning" | "error"
  /** 轨道颜色 */
  trackColor?: string
  /** 是否为不确定模式 (旋转动画) */
  indeterminate?: boolean
}

const sizeMap = {
  xs: 24,
  sm: 32,
  md: 48,
  lg: 64,
  xl: 80,
}

const strokeWidthMap = {
  xs: 3,
  sm: 3,
  md: 4,
  lg: 5,
  xl: 6,
}

const colorMap = {
  primary: "stroke-primary",
  secondary: "stroke-secondary",
  success: "stroke-green-500",
  warning: "stroke-yellow-500",
  error: "stroke-red-500",
}

function CircularProgress({
  className,
  value = 0,
  size = "md",
  strokeWidth,
  showValue = false,
  label,
  color = "primary",
  trackColor = "stroke-muted",
  indeterminate = false,
  ...props
}: CircularProgressProps) {
  const computedSize = typeof size === "number" ? size : sizeMap[size]
  const computedStrokeWidth =
    strokeWidth ?? (typeof size === "number" ? 4 : strokeWidthMap[size])

  const radius = (computedSize - computedStrokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const clampedValue = Math.min(100, Math.max(0, value))
  const offset = circumference - (clampedValue / 100) * circumference

  const center = computedSize / 2

  return (
    <div className={cn("relative inline-flex", className)}>
      <svg
        width={computedSize}
        height={computedSize}
        viewBox={`0 0 ${computedSize} ${computedSize}`}
        className={cn(indeterminate && "animate-spin")}
        {...props}
      >
        {/* 背景轨道 */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          strokeWidth={computedStrokeWidth}
          className={trackColor}
        />
        {/* 进度圆弧 */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          strokeWidth={computedStrokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={indeterminate ? circumference * 0.75 : offset}
          className={cn(colorMap[color], "transition-all duration-300")}
          style={{
            transform: "rotate(-90deg)",
            transformOrigin: "center",
          }}
        />
      </svg>

      {/* 中心内容 */}
      {(showValue || label) && (
        <div className="absolute inset-0 flex items-center justify-center">
          {label || (
            <span
              className={cn(
                "font-medium",
                computedSize < 40 ? "text-xs" : "text-sm"
              )}
            >
              {Math.round(clampedValue)}%
            </span>
          )}
        </div>
      )}
    </div>
  )
}

// 预设组件
function CircularProgressIndeterminate(
  props: Omit<CircularProgressProps, "indeterminate" | "value">
) {
  return <CircularProgress indeterminate value={25} {...props} />
}

export { CircularProgress, CircularProgressIndeterminate }
