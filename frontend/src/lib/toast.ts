import { toast as sonnerToast, type ExternalToast } from "sonner"
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Info,
  Loader2,
} from "lucide-react"
import * as React from "react"

type ToastType = "success" | "error" | "warning" | "info" | "loading"

interface ToastOptions extends Omit<ExternalToast, "description"> {
  description?: string
}

interface ToastPromiseOptions<T> {
  loading: string
  success: string | ((data: T) => string)
  error: string | ((error: unknown) => string)
}

const icons: Record<ToastType, React.ReactNode> = {
  success: React.createElement(CheckCircle2, { className: "h-5 w-5 text-green-500" }),
  error: React.createElement(XCircle, { className: "h-5 w-5 text-red-500" }),
  warning: React.createElement(AlertTriangle, { className: "h-5 w-5 text-yellow-500" }),
  info: React.createElement(Info, { className: "h-5 w-5 text-blue-500" }),
  loading: React.createElement(Loader2, { className: "h-5 w-5 animate-spin text-primary" }),
}

function createToast(type: ToastType) {
  return (message: string, options?: ToastOptions) => {
    const { description, ...rest } = options || {}
    return sonnerToast(message, {
      icon: icons[type],
      description,
      ...rest,
    })
  }
}

export const toast = {
  // 基础消息
  message: (message: string, options?: ToastOptions) => {
    const { description, ...rest } = options || {}
    return sonnerToast(message, { description, ...rest })
  },

  // 成功
  success: createToast("success"),

  // 错误
  error: createToast("error"),

  // 警告
  warning: createToast("warning"),

  // 信息
  info: createToast("info"),

  // 加载中
  loading: createToast("loading"),

  // Promise 包装
  promise: <T,>(
    promise: Promise<T>,
    options: ToastPromiseOptions<T>
  ) => {
    return sonnerToast.promise(promise, {
      loading: options.loading,
      success: options.success,
      error: options.error,
    })
  },

  // 自定义
  custom: (content: React.ReactNode, options?: ExternalToast) => {
    return sonnerToast.custom(() => content as React.ReactElement, options)
  },

  // 关闭
  dismiss: (id?: string | number) => {
    sonnerToast.dismiss(id)
  },

  // 关闭所有
  dismissAll: () => {
    sonnerToast.dismiss()
  },
}

export type { ToastOptions, ToastPromiseOptions }
