import * as React from "react"
import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"

export interface FormFieldProps extends React.HTMLAttributes<HTMLDivElement> {
  /** 字段标签 */
  label?: string
  /** 是否必填 */
  required?: boolean
  /** 错误信息 */
  error?: string
  /** 帮助文本 */
  helperText?: string
  /** 字段描述 */
  description?: string
  /** 标签位置 */
  labelPosition?: "top" | "left"
  /** 标签宽度 (labelPosition="left" 时生效) */
  labelWidth?: string
  /** 禁用状态 */
  disabled?: boolean
}

const FormField = React.forwardRef<HTMLDivElement, FormFieldProps>(
  (
    {
      className,
      label,
      required,
      error,
      helperText,
      description,
      labelPosition = "top",
      labelWidth = "120px",
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const id = React.useId()
    const errorId = `${id}-error`
    const descriptionId = `${id}-description`

    const hasError = !!error

    // Clone children to inject aria attributes
    const childrenWithProps = React.Children.map(children, (child) => {
      if (React.isValidElement(child)) {
        const childProps = child.props as Record<string, unknown>
        return React.cloneElement(child as React.ReactElement<Record<string, unknown>>, {
          id: id,
          "aria-invalid": hasError || undefined,
          "aria-describedby": cn(
            hasError && errorId,
            description && descriptionId
          ) || undefined,
          disabled: disabled || childProps.disabled,
        })
      }
      return child
    })

    if (labelPosition === "left") {
      return (
        <div
          ref={ref}
          className={cn("flex items-start gap-4", className)}
          {...props}
        >
          {label && (
            <Label
              htmlFor={id}
              className={cn(
                "pt-2 text-sm font-medium shrink-0",
                disabled && "opacity-50",
                hasError && "text-destructive"
              )}
              style={{ width: labelWidth }}
            >
              {label}
              {required && <span className="text-destructive ml-1">*</span>}
            </Label>
          )}
          <div className="flex-1 space-y-1.5">
            {childrenWithProps}
            {description && !error && (
              <p
                id={descriptionId}
                className="text-xs text-muted-foreground"
              >
                {description}
              </p>
            )}
            {error && (
              <p id={errorId} className="text-xs text-destructive">
                {error}
              </p>
            )}
            {helperText && !error && (
              <p className="text-xs text-muted-foreground">{helperText}</p>
            )}
          </div>
        </div>
      )
    }

    return (
      <div ref={ref} className={cn("space-y-2", className)} {...props}>
        {label && (
          <Label
            htmlFor={id}
            className={cn(
              disabled && "opacity-50",
              hasError && "text-destructive"
            )}
          >
            {label}
            {required && <span className="text-destructive ml-1">*</span>}
          </Label>
        )}
        {description && (
          <p id={descriptionId} className="text-xs text-muted-foreground">
            {description}
          </p>
        )}
        {childrenWithProps}
        {error && (
          <p id={errorId} className="text-xs text-destructive">
            {error}
          </p>
        )}
        {helperText && !error && (
          <p className="text-xs text-muted-foreground">{helperText}</p>
        )}
      </div>
    )
  }
)
FormField.displayName = "FormField"

// 表单分组
export interface FormSectionProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string
  description?: string
}

function FormSection({
  className,
  title,
  description,
  children,
  ...props
}: FormSectionProps) {
  return (
    <div className={cn("space-y-4", className)} {...props}>
      {(title || description) && (
        <div className="space-y-1">
          {title && (
            <h3 className="text-lg font-medium leading-6">{title}</h3>
          )}
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
      )}
      <div className="space-y-4">{children}</div>
    </div>
  )
}

// 表单操作栏
export interface FormActionsProps extends React.HTMLAttributes<HTMLDivElement> {
  align?: "left" | "center" | "right" | "between"
}

function FormActions({
  className,
  align = "right",
  children,
  ...props
}: FormActionsProps) {
  const alignClass = {
    left: "justify-start",
    center: "justify-center",
    right: "justify-end",
    between: "justify-between",
  }

  return (
    <div
      className={cn(
        "flex items-center gap-3 pt-4",
        alignClass[align],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
}

export { FormField, FormSection, FormActions }
