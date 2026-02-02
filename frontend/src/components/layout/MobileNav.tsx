"use client"

import { useLocation, useNavigate } from "react-router-dom"
import {
  FileAudio,
  Mic,
  BookOpen,
  Settings,
  Activity,
} from "lucide-react"
import { cn } from "@/lib/utils"

const navItems = [
  { path: "/", icon: FileAudio, label: "转写" },
  { path: "/realtime", icon: Mic, label: "实时" },
  { path: "/hotwords", icon: BookOpen, label: "热词" },
  { path: "/config", icon: Settings, label: "设置" },
  { path: "/monitor", icon: Activity, label: "监控" },
]

function MobileNav({ className }: { className?: string }) {
  const location = useLocation()
  const navigate = useNavigate()

  return (
    <nav
      className={cn(
        "fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80 md:hidden",
        className
      )}
    >
      <div className="flex items-center justify-around px-2 py-1 safe-area-bottom">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = location.pathname === item.path

          return (
            <button
              key={item.path}
              onClick={() => navigate(item.path)}
              className={cn(
                "flex flex-col items-center gap-0.5 px-3 py-2 rounded-lg transition-colors min-w-[56px]",
                isActive
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <Icon className={cn("h-5 w-5", isActive && "text-primary")} />
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}

export { MobileNav }
