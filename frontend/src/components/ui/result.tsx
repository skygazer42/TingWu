import * as React from "react"
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  type LucideIcon,
} from "lucide-react"

const resultVariants = cva(
  "flex flex-col items-center justify-center text-center py-8 px-4",
  {
    variants: {
      status: {
        success: "",
        error: "",
        warning: "",
        info: "",
      },
    },
    defaultVariants: {
      status: "success",
    },
  }
)

const statusConfig: Record<
  string,
  { icon: LucideIcon; iconClass: string; titleClass: string }
> = {
  success: {
    icon: CheckCircle2,
    iconClass: "text-green-500",
    titleClass: "text-green-700 dark:text-green-400",
  },
  error: {
    icon: XCircle,
    iconClass: "text-red-500",
    titleClass: "text-red-700 dark:text-red-400",
  },
  warning: {
    icon: AlertTriangle,
    iconClass: "text-yellow-500",
    titleClass: "text-yellow-700 dark:text-yellow-400",
  },
  info: {
    icon: Info,
    iconClass: "text-blue-500",
    titleClass: "text-blue-700 dark:text-blue-400",
  },
}

export interface ResultProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof resultVariants> {
  /** 状态类型 */
  status?: "success" | "error" | "warning" | "info"
  /** 自定义图标 */
  icon?: React.ReactNode
  /** 标题 */
  title: string
  /** 副标题/描述 */
  description?: string
  /** 额外内容 */
  extra?: React.ReactNode
  /** 操作按钮 */
  actions?: React.ReactNode
}

function Result({
  className,
  status = "success",
  icon,
  title,
  description,
  extra,
  actions,
  ...props
}: ResultProps) {
  const config = statusConfig[status]
  const IconComponent = config.icon

  return (
    <div className={cn(resultVariants({ status }), className)} {...props}>
      {/* Icon */}
      <div className="mb-4">
        {icon || (
          <IconComponent className={cn("h-16 w-16", config.iconClass)} />
        )}
      </div>

      {/* Title */}
      <h3 className={cn("text-xl font-semibold mb-2", config.titleClass)}>
        {title}
      </h3>

      {/* Description */}
      {description && (
        <p className="text-sm text-muted-foreground max-w-md mb-4">
          {description}
        </p>
      )}

      {/* Extra content */}
      {extra && <div className="mb-4">{extra}</div>}

      {/* Actions */}
      {actions && <div className="flex items-center gap-3">{actions}</div>}
    </div>
  )
}

// 预设结果组件
function ResultSuccess({
  title = "操作成功",
  ...props
}: Partial<ResultProps>) {
  return <Result status="success" title={title} {...props} />
}

function ResultError({
  title = "操作失败",
  ...props
}: Partial<ResultProps>) {
  return <Result status="error" title={title} {...props} />
}

function ResultWarning({
  title = "警告",
  ...props
}: Partial<ResultProps>) {
  return <Result status="warning" title={title} {...props} />
}

function ResultInfo({
  title = "提示",
  ...props
}: Partial<ResultProps>) {
  return <Result status="info" title={title} {...props} />
}

export {
  Result,
  ResultSuccess,
  ResultError,
  ResultWarning,
  ResultInfo,
  resultVariants,
}
