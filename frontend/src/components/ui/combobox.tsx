"use client"

import * as React from "react"
import { Check, ChevronsUpDown, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"

export interface ComboboxOption {
  value: string
  label: string
  disabled?: boolean
  description?: string
  icon?: React.ReactNode
}

export interface ComboboxProps {
  /** 选项列表 */
  options: ComboboxOption[]
  /** 当前值 (单选时为 string, 多选时为 string[]) */
  value?: string | string[]
  /** 值变化回调 */
  onValueChange?: (value: string | string[]) => void
  /** 占位符 */
  placeholder?: string
  /** 搜索占位符 */
  searchPlaceholder?: string
  /** 空结果文本 */
  emptyText?: string
  /** 是否多选 */
  multiple?: boolean
  /** 是否禁用 */
  disabled?: boolean
  /** 是否可清空 */
  clearable?: boolean
  /** 自定义类名 */
  className?: string
  /** 弹出框宽度 */
  popoverWidth?: string
  /** 最大显示标签数 (多选时) */
  maxTagCount?: number
}

function Combobox({
  options,
  value,
  onValueChange,
  placeholder = "请选择...",
  searchPlaceholder = "搜索...",
  emptyText = "未找到结果",
  multiple = false,
  disabled = false,
  clearable = false,
  className,
  popoverWidth = "300px",
  maxTagCount = 3,
}: ComboboxProps) {
  const [open, setOpen] = React.useState(false)
  const [searchValue, setSearchValue] = React.useState("")

  // 标准化 value
  const selectedValues = React.useMemo(() => {
    if (!value) return []
    return Array.isArray(value) ? value : [value]
  }, [value])

  // 获取选中项的标签
  const getLabel = (val: string) => {
    const option = options.find((o) => o.value === val)
    return option?.label || val
  }

  // 处理选择
  const handleSelect = (optionValue: string) => {
    if (multiple) {
      const newValues = selectedValues.includes(optionValue)
        ? selectedValues.filter((v) => v !== optionValue)
        : [...selectedValues, optionValue]
      onValueChange?.(newValues)
    } else {
      onValueChange?.(optionValue)
      setOpen(false)
    }
    setSearchValue("")
  }

  // 处理清空
  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation()
    onValueChange?.(multiple ? [] : "")
  }

  // 移除单个标签
  const handleRemoveTag = (val: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (multiple) {
      onValueChange?.(selectedValues.filter((v) => v !== val))
    }
  }

  // 过滤选项
  const filteredOptions = options.filter((option) =>
    option.label.toLowerCase().includes(searchValue.toLowerCase())
  )

  // 渲染触发器内容
  const renderTriggerContent = () => {
    if (selectedValues.length === 0) {
      return <span className="text-muted-foreground">{placeholder}</span>
    }

    if (multiple) {
      const displayValues = selectedValues.slice(0, maxTagCount)
      const remainingCount = selectedValues.length - maxTagCount

      return (
        <div className="flex flex-wrap gap-1">
          {displayValues.map((val) => (
            <Badge
              key={val}
              variant="secondary"
              className="h-5 text-xs font-normal"
            >
              {getLabel(val)}
              <button
                type="button"
                className="ml-1 rounded-full hover:bg-muted"
                onClick={(e) => handleRemoveTag(val, e)}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {remainingCount > 0 && (
            <Badge variant="secondary" className="h-5 text-xs font-normal">
              +{remainingCount}
            </Badge>
          )}
        </div>
      )
    }

    return <span>{getLabel(selectedValues[0])}</span>
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          disabled={disabled}
          className={cn(
            "w-full justify-between font-normal",
            !selectedValues.length && "text-muted-foreground",
            className
          )}
        >
          <div className="flex-1 text-left truncate">
            {renderTriggerContent()}
          </div>
          <div className="flex items-center gap-1 ml-2 shrink-0">
            {clearable && selectedValues.length > 0 && (
              <X
                className="h-4 w-4 opacity-50 hover:opacity-100"
                onClick={handleClear}
              />
            )}
            <ChevronsUpDown className="h-4 w-4 opacity-50" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="p-0" style={{ width: popoverWidth }}>
        <Command shouldFilter={false}>
          <CommandInput
            placeholder={searchPlaceholder}
            value={searchValue}
            onValueChange={setSearchValue}
          />
          <CommandList>
            <CommandEmpty>{emptyText}</CommandEmpty>
            <CommandGroup>
              {filteredOptions.map((option) => (
                <CommandItem
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                  onSelect={() => handleSelect(option.value)}
                >
                  {option.icon && (
                    <span className="mr-2 shrink-0">{option.icon}</span>
                  )}
                  <div className="flex-1">
                    <div>{option.label}</div>
                    {option.description && (
                      <div className="text-xs text-muted-foreground">
                        {option.description}
                      </div>
                    )}
                  </div>
                  <Check
                    className={cn(
                      "ml-2 h-4 w-4 shrink-0",
                      selectedValues.includes(option.value)
                        ? "opacity-100"
                        : "opacity-0"
                    )}
                  />
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}

export { Combobox }
