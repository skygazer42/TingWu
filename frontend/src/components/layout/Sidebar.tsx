import { Link, useLocation } from 'react-router-dom'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet'
import { Logo, LogoIcon } from '@/components/brand'
import { useAppStore } from '@/stores'
import { useIsMobile } from '@/hooks'
import {
  Radio,
  BookText,
  Settings,
  BarChart3,
  Menu,
  ChevronLeft,
  ChevronRight,
  FileAudio,
} from 'lucide-react'

const navItems = [
  { path: '/', label: '转写', icon: FileAudio },
  { path: '/realtime', label: '实时转写', icon: Radio },
  { path: '/hotwords', label: '热词管理', icon: BookText },
  { path: '/config', label: '配置管理', icon: Settings },
  { path: '/monitor', label: '系统监控', icon: BarChart3 },
]

interface NavItemProps {
  path: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  collapsed?: boolean
}

function NavItem({ path, label, icon: Icon, collapsed }: NavItemProps) {
  const location = useLocation()
  const isActive = location.pathname === path

  return (
    <Link to={path}>
      <Button
        variant={isActive ? 'secondary' : 'ghost'}
        className={cn(
          'w-full justify-start gap-3',
          collapsed && 'justify-center px-2'
        )}
      >
        <Icon className="h-5 w-5 shrink-0" />
        {!collapsed && <span>{label}</span>}
      </Button>
    </Link>
  )
}

interface SidebarContentProps {
  collapsed?: boolean
  onCollapse?: () => void
}

function SidebarContent({ collapsed, onCollapse }: SidebarContentProps) {
  return (
    <div className="flex h-full flex-col">
      {/* Logo */}
      <div className={cn(
        'flex h-14 items-center border-b px-4',
        collapsed ? 'justify-center' : 'gap-2'
      )}>
        {collapsed ? (
          <LogoIcon size={28} />
        ) : (
          <Logo size="sm" />
        )}
      </div>

      {/* Navigation */}
      <ScrollArea className="flex-1 px-2 py-4">
        <nav className="flex flex-col gap-1">
          {navItems.map((item) => (
            <NavItem
              key={item.path}
              {...item}
              collapsed={collapsed}
            />
          ))}
        </nav>
      </ScrollArea>

      {/* Collapse Button (Desktop) */}
      {onCollapse && (
        <div className="border-t p-2">
          <Button
            variant="ghost"
            size="sm"
            className="w-full"
            onClick={onCollapse}
          >
            {collapsed ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <>
                <ChevronLeft className="h-4 w-4 mr-2" />
                <span>收起</span>
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  )
}

export function Sidebar() {
  const isMobile = useIsMobile()
  const { sidebarOpen, setSidebarOpen, sidebarCollapsed, setSidebarCollapsed } = useAppStore()

  // 移动端使用 Sheet
  if (isMobile) {
    return (
      <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
        <SheetTrigger asChild>
          <Button variant="ghost" size="icon" className="md:hidden">
            <Menu className="h-5 w-5" />
          </Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-64 p-0">
          <SidebarContent />
        </SheetContent>
      </Sheet>
    )
  }

  // 桌面端显示固定侧边栏
  return (
    <aside
      className={cn(
        'hidden md:flex flex-col border-r bg-card transition-all duration-300',
        sidebarCollapsed ? 'w-16' : 'w-56'
      )}
    >
      <SidebarContent
        collapsed={sidebarCollapsed}
        onCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
    </aside>
  )
}
