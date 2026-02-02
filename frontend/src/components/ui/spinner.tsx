import { cn } from "@/lib/utils"
import { cva, type VariantProps } from "class-variance-authority"
import { Loader2 } from "lucide-react"

const spinnerVariants = cva("animate-spin text-muted-foreground", {
  variants: {
    size: {
      xs: "h-3 w-3",
      sm: "h-4 w-4",
      md: "h-6 w-6",
      lg: "h-8 w-8",
      xl: "h-12 w-12",
    },
  },
  defaultVariants: {
    size: "md",
  },
})

export interface SpinnerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof spinnerVariants> {
  /** 显示在加载器旁边的文本 */
  label?: string
  /** 文本位置 */
  labelPosition?: "right" | "bottom"
}

function Spinner({
  className,
  size,
  label,
  labelPosition = "right",
  ...props
}: SpinnerProps) {
  const spinnerElement = (
    <Loader2 className={cn(spinnerVariants({ size }), className)} />
  )

  if (!label) {
    return spinnerElement
  }

  const textSizes = {
    xs: "text-xs",
    sm: "text-sm",
    md: "text-sm",
    lg: "text-base",
    xl: "text-lg",
  }

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2",
        labelPosition === "bottom" && "flex-col"
      )}
      {...props}
    >
      {spinnerElement}
      <span className={cn("text-muted-foreground", textSizes[size || "md"])}>
        {label}
      </span>
    </div>
  )
}

// 全屏加载遮罩
function SpinnerOverlay({
  label = "加载中...",
  className,
}: {
  label?: string
  className?: string
}) {
  return (
    <div
      className={cn(
        "fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm",
        className
      )}
    >
      <Spinner size="xl" label={label} labelPosition="bottom" />
    </div>
  )
}

// 内联加载占位
function SpinnerInline({
  label,
  className,
}: {
  label?: string
  className?: string
}) {
  return (
    <div
      className={cn(
        "flex items-center justify-center py-8",
        className
      )}
    >
      <Spinner size="lg" label={label} />
    </div>
  )
}

export { Spinner, SpinnerOverlay, SpinnerInline, spinnerVariants }
