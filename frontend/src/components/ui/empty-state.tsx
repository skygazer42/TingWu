import * as React from "react"
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"
import {
  FileAudio,
  FileQuestion,
  Inbox,
  Search,
  type LucideIcon,
} from "lucide-react"

const emptyStateVariants = cva(
  "flex flex-col items-center justify-center text-center",
  {
    variants: {
      size: {
        sm: "py-8 gap-3",
        default: "py-12 gap-4",
        lg: "py-16 gap-5",
      },
    },
    defaultVariants: {
      size: "default",
    },
  }
)

const iconSizes = {
  sm: "h-10 w-10",
  default: "h-12 w-12",
  lg: "h-16 w-16",
}

const titleSizes = {
  sm: "text-base",
  default: "text-lg",
  lg: "text-xl",
}

export interface EmptyStateProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof emptyStateVariants> {
  /** 图标 */
  icon?: LucideIcon | React.ReactNode
  /** 标题 */
  title: string
  /** 描述文本 */
  description?: string
  /** 操作按钮 */
  action?: React.ReactNode
  /** 次要操作 */
  secondaryAction?: React.ReactNode
}

// 预设图标
const presetIcons = {
  audio: FileAudio,
  search: Search,
  inbox: Inbox,
  question: FileQuestion,
}

function EmptyState({
  className,
  size = "default",
  icon,
  title,
  description,
  action,
  secondaryAction,
  ...props
}: EmptyStateProps) {
  const IconComponent =
    typeof icon === "string" && icon in presetIcons
      ? presetIcons[icon as keyof typeof presetIcons]
      : null

  const renderIcon = () => {
    if (!icon) return null

    if (IconComponent) {
      return (
        <IconComponent
          className={cn(
            iconSizes[size || "default"],
            "text-muted-foreground/50"
          )}
        />
      )
    }

    if (React.isValidElement(icon)) {
      return icon
    }

    // LucideIcon
    const Icon = icon as LucideIcon
    return (
      <Icon
        className={cn(iconSizes[size || "default"], "text-muted-foreground/50")}
      />
    )
  }

  return (
    <div className={cn(emptyStateVariants({ size }), className)} {...props}>
      {renderIcon()}

      <div className="space-y-1.5">
        <h3
          className={cn(
            "font-medium text-foreground",
            titleSizes[size || "default"]
          )}
        >
          {title}
        </h3>
        {description && (
          <p className="text-sm text-muted-foreground max-w-sm">{description}</p>
        )}
      </div>

      {(action || secondaryAction) && (
        <div className="flex items-center gap-3 mt-2">
          {action}
          {secondaryAction}
        </div>
      )}
    </div>
  )
}

// 预设空状态
function EmptyStateNoData({
  title = "暂无数据",
  description = "还没有任何内容",
  action,
  ...props
}: Partial<EmptyStateProps>) {
  return (
    <EmptyState
      icon={Inbox}
      title={title}
      description={description}
      action={action}
      {...props}
    />
  )
}

function EmptyStateNoResults({
  title = "未找到结果",
  description = "尝试调整搜索条件或筛选器",
  action,
  ...props
}: Partial<EmptyStateProps>) {
  return (
    <EmptyState
      icon={Search}
      title={title}
      description={description}
      action={action}
      {...props}
    />
  )
}

function EmptyStateNoAudio({
  title = "暂无转写结果",
  description = "上传音频文件并点击开始转写",
  action,
  ...props
}: Partial<EmptyStateProps>) {
  return (
    <EmptyState
      icon={FileAudio}
      title={title}
      description={description}
      action={action}
      {...props}
    />
  )
}

export {
  EmptyState,
  EmptyStateNoData,
  EmptyStateNoResults,
  EmptyStateNoAudio,
  emptyStateVariants,
}
