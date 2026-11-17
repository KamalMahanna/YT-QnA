import React from 'react'
import {
  PanelLeftClose,
  PanelLeft,
  Home,
  Layers,
  LayoutGrid,
  Clock,
  MessageSquare,
  FileText,
  Sparkles,
  History,
  Sun,
  Moon,
} from 'lucide-react'

interface SidebarProps {
  isCollapsed: boolean
  setIsCollapsed: (val: boolean) => void
  activeTab: 'timeline' | 'chat' | 'summary'
  setActiveTab: (tab: 'timeline' | 'chat' | 'summary') => void
  videoId: string | null
  recentVideos: string[]
  onSelectVideo: (videoId: string) => void
  theme?: 'dark' | 'light'
  onToggleTheme?: () => void
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed,
  setIsCollapsed,
  activeTab,
  setActiveTab,
  videoId,
  recentVideos,
  onSelectVideo,
  theme = 'dark',
  onToggleTheme,
}) => {
  const isLight = theme === 'light'

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleSelectTab = (tab: 'timeline' | 'chat' | 'summary') => {
    setActiveTab(tab)
    scrollTo('studio-workspace')
  }

  return (
    <>
      {/* Mobile Backdrop Overlay when expanded on small screens */}
      {!isCollapsed && (
        <div
          onClick={() => setIsCollapsed(true)}
          className="fixed inset-0 bg-black/60 backdrop-blur-xs z-40 md:hidden"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex flex-col justify-between apple-glass transition-[width,transform] duration-200 ease-out ${
          isLight ? 'border-r border-black/[0.08]' : 'border-r border-white/[0.08]'
        } ${
          isCollapsed
            ? '-translate-x-full md:translate-x-0 md:w-16'
            : 'w-64 translate-x-0 shadow-2xl md:shadow-none'
        }`}
      >
        {/* Top Header & Brand */}
        <div className="flex flex-col">
          <div
            className={`flex items-center justify-between p-3.5 border-b ${
              isLight ? 'border-black/[0.06]' : 'border-white/[0.06]'
            }`}
          >
            {!isCollapsed ? (
              <div
                onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
                className="flex items-center gap-2.5 cursor-pointer group select-none"
              >
                <div
                  className={`w-7 h-7 rounded-lg flex items-center justify-center transition-colors ${
                    isLight
                      ? 'bg-zinc-100 border border-black/10'
                      : 'bg-zinc-900 border border-white/10 group-hover:border-white/30'
                  }`}
                >
                  <Sparkles className={`w-3.5 h-3.5 ${isLight ? 'text-black' : 'text-white'}`} />
                </div>
                <div>
                  <span
                    className={`font-semibold text-xs tracking-tight leading-none block ${
                      isLight ? 'text-black' : 'text-white'
                    }`}
                  >
                    YouTube QnA
                  </span>
                  <span className="text-[10px] text-zinc-500 font-mono tracking-wider uppercase block mt-0.5">
                    Studio
                  </span>
                </div>
              </div>
            ) : (
              <div
                onClick={() => setIsCollapsed(false)}
                className="w-full flex items-center justify-center cursor-pointer group"
                title="Expand Sidebar"
              >
                <div
                  className={`w-8 h-8 rounded-lg flex items-center justify-center transition-colors ${
                    isLight
                      ? 'bg-zinc-100 border border-black/10'
                      : 'bg-zinc-900 border border-white/10 group-hover:border-white/30'
                  }`}
                >
                  <Sparkles className={`w-4 h-4 ${isLight ? 'text-black' : 'text-white'}`} />
                </div>
              </div>
            )}

            {!isCollapsed && (
              <button
                onClick={() => setIsCollapsed(true)}
                className={`p-1 rounded-md active:scale-95 transition-all ${
                  isLight
                    ? 'text-zinc-500 hover:text-black hover:bg-black/[0.05]'
                    : 'text-zinc-400 hover:text-white hover:bg-white/[0.06]'
                }`}
                title="Collapse Sidebar"
              >
                <PanelLeftClose className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Navigation Links */}
          <div className="p-2 space-y-1">
            {!isCollapsed && (
              <span className="px-2.5 py-1 text-[10px] font-mono tracking-wider uppercase text-zinc-500 block">
                Navigation
              </span>
            )}

            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium active:scale-[0.98] transition-all ${
                isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Overview"
            >
              <Home className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Overview</span>}
            </button>

            <button
              onClick={() => scrollTo('storyline')}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium active:scale-[0.98] transition-all ${
                isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Architecture"
            >
              <Layers className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Architecture</span>}
            </button>

            <button
              onClick={() => scrollTo('studio-workspace')}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium active:scale-[0.98] transition-all ${
                isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Studio Workspace"
            >
              <LayoutGrid className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Studio Workspace</span>}
            </button>
          </div>

          {/* Studio Modes */}
          <div
            className={`p-2 space-y-1 border-t mt-2 ${
              isLight ? 'border-black/[0.06]' : 'border-white/[0.06]'
            }`}
          >
            {!isCollapsed && (
              <span className="px-2.5 py-1 text-[10px] font-mono tracking-wider uppercase text-zinc-500 block">
                Studio Modes
              </span>
            )}

            <button
              onClick={() => handleSelectTab('timeline')}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium transition-all ${
                activeTab === 'timeline'
                  ? isLight
                    ? 'bg-black text-white font-semibold shadow-xs'
                    : 'bg-white text-black font-semibold shadow-xs'
                  : isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Storyline Beats"
            >
              <Clock className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Storyline Beats</span>}
            </button>

            <button
              onClick={() => handleSelectTab('chat')}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium transition-all ${
                activeTab === 'chat'
                  ? isLight
                    ? 'bg-black text-white font-semibold shadow-xs'
                    : 'bg-white text-black font-semibold shadow-xs'
                  : isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Interactive Q&A"
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Interactive Q&A</span>}
            </button>

            <button
              onClick={() => handleSelectTab('summary')}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium transition-all ${
                activeTab === 'summary'
                  ? isLight
                    ? 'bg-black text-white font-semibold shadow-xs'
                    : 'bg-white text-black font-semibold shadow-xs'
                  : isLight
                  ? 'text-zinc-600 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title="Summary & Audio"
            >
              <FileText className="w-4 h-4 shrink-0" />
              {!isCollapsed && <span>Summary & Audio</span>}
            </button>
          </div>

          {/* Recent Video Sessions */}
          {!isCollapsed && recentVideos && recentVideos.length > 0 && (
            <div
              className={`p-2 space-y-1 border-t mt-2 ${
                isLight ? 'border-black/[0.06]' : 'border-white/[0.06]'
              }`}
            >
              <span className="px-2.5 py-1 text-[10px] font-mono tracking-wider uppercase text-zinc-500 block">
                Recent Sessions
              </span>
              <div className="space-y-0.5 max-h-32 overflow-y-auto">
                {recentVideos.slice(0, 4).map((vid) => (
                  <button
                    key={vid}
                    onClick={() => {
                      onSelectVideo(vid)
                      scrollTo('studio-workspace')
                    }}
                    className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-xs font-mono text-left truncate transition-colors ${
                      videoId === vid
                        ? isLight
                          ? 'bg-black/10 text-black font-semibold'
                          : 'bg-white/10 text-white font-semibold'
                        : isLight
                        ? 'text-zinc-600 hover:text-black hover:bg-black/[0.03]'
                        : 'text-zinc-400 hover:text-zinc-200 hover:bg-white/[0.03]'
                    }`}
                  >
                    <History className="w-3 h-3 shrink-0 text-zinc-400" />
                    <span className="truncate">{vid}</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer: Expand Button & Theme Toggle */}
        <div
          className={`p-2 border-t flex flex-col gap-1 ${
            isLight ? 'border-black/[0.06]' : 'border-white/[0.06]'
          }`}
        >
          {/* Light / Dark Mode Toggle */}
          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              className={`w-full flex items-center gap-3 px-2.5 py-2 rounded-xl text-xs font-medium active:scale-[0.98] transition-all cursor-pointer ${
                isLight
                  ? 'text-zinc-700 hover:text-black hover:bg-black/[0.05]'
                  : 'text-zinc-300 hover:text-white hover:bg-white/[0.06]'
              } ${isCollapsed ? 'justify-center' : ''}`}
              title={`Switch to ${isLight ? 'Dark' : 'Light'} Mode`}
            >
              {isLight ? (
                <Moon className="w-4 h-4 shrink-0 text-zinc-700" />
              ) : (
                <Sun className="w-4 h-4 shrink-0 text-zinc-300" />
              )}
              {!isCollapsed && (
                <span>{isLight ? 'Dark Mode' : 'Light Mode'}</span>
              )}
            </button>
          )}

          {/* Collapse/Expand Toggle on Desktop */}
          {isCollapsed && (
            <button
              onClick={() => setIsCollapsed(false)}
              className={`w-full flex items-center justify-center p-2 rounded-xl active:scale-95 transition-all cursor-pointer ${
                isLight
                  ? 'text-zinc-500 hover:text-black hover:bg-black/[0.04]'
                  : 'text-zinc-400 hover:text-white hover:bg-white/[0.04]'
              }`}
              title="Expand Sidebar"
            >
              <PanelLeft className="w-4 h-4" />
            </button>
          )}
        </div>
      </aside>
    </>
  )
}
