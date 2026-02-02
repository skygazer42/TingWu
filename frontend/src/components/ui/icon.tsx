import * as React from "react"
import {
  Upload,
  Download,
  Copy,
  Trash2,
  Pencil,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Info,
  Loader2,
  FileAudio,
  FileVideo,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  Home,
  Settings,
  HelpCircle,
  Play,
  Pause,
  Square,
  RotateCcw,
  Save,
  FolderOpen,
  File,
  FileText,
  Clock,
  Calendar,
  Search,
  Filter,
  SortAsc,
  SortDesc,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
  X,
  Plus,
  Minus,
  Check,
  MoreHorizontal,
  MoreVertical,
  ExternalLink,
  Link,
  Unlink,
  Eye,
  EyeOff,
  Moon,
  Sun,
  Globe,
  Wifi,
  WifiOff,
  Zap,
  Sparkles,
  type LucideIcon,
} from "lucide-react"
import { cn } from "@/lib/utils"

// 图标映射表
export const ICONS = {
  // 操作
  upload: Upload,
  download: Download,
  copy: Copy,
  delete: Trash2,
  edit: Pencil,
  save: Save,
  open: FolderOpen,
  add: Plus,
  remove: Minus,
  close: X,
  check: Check,
  more: MoreHorizontal,
  moreVertical: MoreVertical,
  external: ExternalLink,
  link: Link,
  unlink: Unlink,

  // 状态
  success: CheckCircle,
  error: XCircle,
  warning: AlertTriangle,
  info: Info,
  loading: Loader2,

  // 媒体
  audio: FileAudio,
  video: FileVideo,
  microphone: Mic,
  microphoneOff: MicOff,
  speaker: Volume2,
  speakerOff: VolumeX,
  play: Play,
  pause: Pause,
  stop: Square,
  replay: RotateCcw,

  // 文件
  file: File,
  fileText: FileText,
  folder: FolderOpen,

  // 导航
  home: Home,
  settings: Settings,
  help: HelpCircle,
  search: Search,
  filter: Filter,

  // 排序
  sortAsc: SortAsc,
  sortDesc: SortDesc,

  // 方向
  chevronDown: ChevronDown,
  chevronUp: ChevronUp,
  chevronLeft: ChevronLeft,
  chevronRight: ChevronRight,

  // 时间
  clock: Clock,
  calendar: Calendar,

  // 可见性
  show: Eye,
  hide: EyeOff,

  // 主题
  dark: Moon,
  light: Sun,

  // 网络
  globe: Globe,
  online: Wifi,
  offline: WifiOff,

  // 特效
  zap: Zap,
  sparkles: Sparkles,
} as const

export type IconName = keyof typeof ICONS

// 图标尺寸映射
const sizeMap = {
  xs: "h-3 w-3",    // 12px
  sm: "h-4 w-4",    // 16px
  md: "h-5 w-5",    // 20px
  lg: "h-6 w-6",    // 24px
  xl: "h-8 w-8",    // 32px
  "2xl": "h-12 w-12", // 48px
  "3xl": "h-16 w-16", // 64px
}

// 图标颜色映射
const colorMap = {
  default: "text-current",
  muted: "text-muted-foreground",
  primary: "text-primary",
  secondary: "text-secondary",
  success: "text-green-500",
  warning: "text-yellow-500",
  error: "text-red-500",
  info: "text-blue-500",
}

export interface IconProps extends React.SVGAttributes<SVGSVGElement> {
  /** 图标名称 */
  name: IconName
  /** 尺寸 */
  size?: keyof typeof sizeMap
  /** 颜色 */
  color?: keyof typeof colorMap
  /** 是否旋转 (用于加载状态) */
  spin?: boolean
}

function Icon({
  name,
  size = "md",
  color = "default",
  spin = false,
  className,
  ...props
}: IconProps) {
  const IconComponent = ICONS[name]

  if (!IconComponent) {
    console.warn(`Icon "${name}" not found`)
    return null
  }

  return (
    <IconComponent
      className={cn(
        sizeMap[size],
        colorMap[color],
        spin && "animate-spin",
        className
      )}
      {...props}
    />
  )
}

// 直接获取图标组件
function getIcon(name: IconName): LucideIcon {
  return ICONS[name]
}

// 图标尺寸常量 (用于内联样式)
export const ICON_SIZES = {
  button: 16,      // 按钮内图标
  nav: 20,         // 导航图标
  card: 24,        // 卡片图标
  empty: 48,       // 空状态图标
  hero: 64,        // 大型展示
} as const

export { Icon, getIcon }
