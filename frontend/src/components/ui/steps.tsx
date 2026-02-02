import * as React from "react"
import { Check, AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"

// 步骤状态
export type StepStatus = "wait" | "process" | "finish" | "error"

// 单个步骤数据
export interface StepItem {
  title: string
  description?: string
  icon?: React.ReactNode
  status?: StepStatus
  disabled?: boolean
}

const stepsVariants = cva("flex", {
  variants: {
    direction: {
      horizontal: "flex-row items-start",
      vertical: "flex-col",
    },
    size: {
      sm: "",
      default: "",
      lg: "",
    },
  },
  defaultVariants: {
    direction: "horizontal",
    size: "default",
  },
})

const stepSizes = {
  sm: {
    icon: "h-6 w-6 text-xs",
    title: "text-xs",
    description: "text-xs",
    line: "h-0.5",
    lineVertical: "w-0.5",
  },
  default: {
    icon: "h-8 w-8 text-sm",
    title: "text-sm",
    description: "text-xs",
    line: "h-0.5",
    lineVertical: "w-0.5",
  },
  lg: {
    icon: "h-10 w-10 text-base",
    title: "text-base",
    description: "text-sm",
    line: "h-1",
    lineVertical: "w-1",
  },
}

export interface StepsProps
  extends Omit<React.HTMLAttributes<HTMLDivElement>, "onChange">,
    VariantProps<typeof stepsVariants> {
  /** 步骤数据 */
  items: StepItem[]
  /** 当前步骤 (从 0 开始) */
  current?: number
  /** 步骤变化回调 */
  onChange?: (current: number) => void
  /** 是否可点击 */
  clickable?: boolean
  /** 连接线是否渐变 */
  progressLine?: boolean
}

function Steps({
  className,
  items,
  current = 0,
  onChange,
  direction = "horizontal",
  size = "default",
  clickable = false,
  progressLine = false,
  ...props
}: StepsProps) {
  const sizeStyles = stepSizes[size || "default"]

  const getStepStatus = (index: number, item: StepItem): StepStatus => {
    if (item.status) return item.status
    if (index < current) return "finish"
    if (index === current) return "process"
    return "wait"
  }

  const handleStepClick = (index: number, item: StepItem) => {
    if (!clickable || item.disabled) return
    onChange?.(index)
  }

  return (
    <div className={cn(stepsVariants({ direction, size }), className)} {...props}>
      {items.map((item, index) => {
        const status = getStepStatus(index, item)
        const isLast = index === items.length - 1

        return (
          <div
            key={index}
            className={cn(
              "flex",
              direction === "horizontal" ? "flex-1 items-start" : "items-start",
              clickable && !item.disabled && "cursor-pointer",
              item.disabled && "opacity-50 cursor-not-allowed"
            )}
            onClick={() => handleStepClick(index, item)}
          >
            <div
              className={cn(
                "flex",
                direction === "horizontal"
                  ? "flex-col items-center flex-1"
                  : "items-start gap-3"
              )}
            >
              {/* 图标和连接线 */}
              <div
                className={cn(
                  "flex items-center",
                  direction === "horizontal" ? "w-full" : "flex-col"
                )}
              >
                {/* 左侧连接线 (水平模式) */}
                {direction === "horizontal" && index > 0 && (
                  <div
                    className={cn(
                      "flex-1",
                      sizeStyles.line,
                      progressLine && index <= current
                        ? "bg-primary"
                        : "bg-muted"
                    )}
                  />
                )}

                {/* 步骤图标 */}
                <StepIcon
                  status={status}
                  icon={item.icon}
                  index={index}
                  className={sizeStyles.icon}
                />

                {/* 右侧连接线 (水平模式) */}
                {direction === "horizontal" && !isLast && (
                  <div
                    className={cn(
                      "flex-1",
                      sizeStyles.line,
                      progressLine && index < current
                        ? "bg-primary"
                        : "bg-muted"
                    )}
                  />
                )}

                {/* 下方连接线 (垂直模式) */}
                {direction === "vertical" && !isLast && (
                  <div
                    className={cn(
                      "flex-1 min-h-8 my-1",
                      sizeStyles.lineVertical,
                      progressLine && index < current
                        ? "bg-primary"
                        : "bg-muted"
                    )}
                  />
                )}
              </div>

              {/* 标题和描述 */}
              <div
                className={cn(
                  "flex flex-col",
                  direction === "horizontal"
                    ? "items-center mt-2 text-center px-2"
                    : "flex-1"
                )}
              >
                <span
                  className={cn(
                    "font-medium",
                    sizeStyles.title,
                    status === "process" && "text-primary",
                    status === "error" && "text-destructive",
                    status === "wait" && "text-muted-foreground"
                  )}
                >
                  {item.title}
                </span>
                {item.description && (
                  <span
                    className={cn(
                      "text-muted-foreground mt-0.5",
                      sizeStyles.description
                    )}
                  >
                    {item.description}
                  </span>
                )}
              </div>
            </div>
          </div>
        )
      })}
    </div>
  )
}

// 步骤图标
interface StepIconProps {
  status: StepStatus
  icon?: React.ReactNode
  index: number
  className?: string
}

function StepIcon({ status, icon, index, className }: StepIconProps) {
  const baseClass = cn(
    "rounded-full flex items-center justify-center font-medium transition-colors shrink-0",
    className
  )

  if (icon) {
    return (
      <div
        className={cn(
          baseClass,
          status === "finish" && "bg-primary text-primary-foreground",
          status === "process" && "border-2 border-primary text-primary",
          status === "wait" && "border-2 border-muted text-muted-foreground",
          status === "error" && "bg-destructive text-destructive-foreground"
        )}
      >
        {icon}
      </div>
    )
  }

  switch (status) {
    case "finish":
      return (
        <div className={cn(baseClass, "bg-primary text-primary-foreground")}>
          <Check className="h-4 w-4" />
        </div>
      )
    case "error":
      return (
        <div className={cn(baseClass, "bg-destructive text-destructive-foreground")}>
          <AlertCircle className="h-4 w-4" />
        </div>
      )
    case "process":
      return (
        <div className={cn(baseClass, "border-2 border-primary text-primary bg-background")}>
          {index + 1}
        </div>
      )
    default:
      return (
        <div className={cn(baseClass, "border-2 border-muted text-muted-foreground bg-background")}>
          {index + 1}
        </div>
      )
  }
}

export { Steps }
