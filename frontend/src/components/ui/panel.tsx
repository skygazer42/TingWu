"use client"

import * as React from "react"
import * as CollapsiblePrimitive from "@radix-ui/react-collapsible"
import { ChevronDown } from "lucide-react"
import { cn } from "@/lib/utils"

export interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 面板标题 */
  title: string
  /** 是否默认展开 */
  defaultOpen?: boolean
  /** 受控展开状态 */
  open?: boolean
  /** 展开状态变化回调 */
  onOpenChange?: (open: boolean) => void
  /** 标题右侧额外内容 */
  extra?: React.ReactNode
  /** 是否可折叠 */
  collapsible?: boolean
  /** 是否无边框 */
  bordered?: boolean
  /** 自定义头部类名 */
  headerClassName?: string
  /** 自定义内容类名 */
  contentClassName?: string
}

function Panel({
  className,
  title,
  defaultOpen = true,
  open,
  onOpenChange,
  extra,
  collapsible = true,
  bordered = true,
  headerClassName,
  contentClassName,
  children,
  ...props
}: PanelProps) {
  if (!collapsible) {
    return (
      <div
        className={cn(
          "rounded-lg",
          bordered && "border bg-card",
          className
        )}
        {...props}
      >
        <div
          className={cn(
            "flex items-center justify-between px-4 py-3",
            bordered && "border-b",
            headerClassName
          )}
        >
          <h3 className="text-sm font-medium">{title}</h3>
          {extra && <div>{extra}</div>}
        </div>
        <div className={cn("px-4 py-3", contentClassName)}>{children}</div>
      </div>
    )
  }

  return (
    <CollapsiblePrimitive.Root
      defaultOpen={defaultOpen}
      open={open}
      onOpenChange={onOpenChange}
      className={cn(
        "rounded-lg",
        bordered && "border bg-card",
        className
      )}
      {...props}
    >
      <CollapsiblePrimitive.Trigger
        className={cn(
          "flex w-full items-center justify-between px-4 py-3 text-left transition-colors hover:bg-muted/50",
          bordered && "border-b data-[state=closed]:border-b-0",
          headerClassName
        )}
      >
        <div className="flex items-center gap-2">
          <ChevronDown className="h-4 w-4 transition-transform duration-200 [[data-state=closed]_&]:-rotate-90" />
          <h3 className="text-sm font-medium">{title}</h3>
        </div>
        {extra && (
          <div onClick={(e) => e.stopPropagation()}>{extra}</div>
        )}
      </CollapsiblePrimitive.Trigger>
      <CollapsiblePrimitive.Content className="overflow-hidden data-[state=closed]:animate-collapse data-[state=open]:animate-expand">
        <div className={cn("px-4 py-3", contentClassName)}>{children}</div>
      </CollapsiblePrimitive.Content>
    </CollapsiblePrimitive.Root>
  )
}

// 面板组 - 手风琴模式
export interface PanelGroupProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 是否只能展开一个 */
  accordion?: boolean
  /** 默认展开的面板 key */
  defaultActiveKey?: string | string[]
  /** 无边框模式 */
  bordered?: boolean
}

function PanelGroup({
  className,
  accordion = false,
  bordered = true,
  children,
  ...props
}: PanelGroupProps) {
  const [activeKeys, setActiveKeys] = React.useState<string[]>([])

  const handleOpenChange = (key: string, open: boolean) => {
    if (accordion) {
      setActiveKeys(open ? [key] : [])
    } else {
      setActiveKeys((prev) =>
        open ? [...prev, key] : prev.filter((k) => k !== key)
      )
    }
  }

  // Clone children and inject props
  const panels = React.Children.map(children, (child, index) => {
    if (React.isValidElement(child)) {
      const key = (child.key as string) || `panel-${index}`
      return React.cloneElement(child as React.ReactElement<PanelProps>, {
        open: activeKeys.includes(key),
        onOpenChange: (open: boolean) => handleOpenChange(key, open),
        bordered: bordered,
      })
    }
    return child
  })

  return (
    <div
      className={cn("space-y-2", className)}
      {...props}
    >
      {panels}
    </div>
  )
}

export { Panel, PanelGroup }
