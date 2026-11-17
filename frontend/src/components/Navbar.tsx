import React, { useState } from 'react'
import { Sparkles, RefreshCw, ArrowRight, X } from 'lucide-react'

interface NavbarProps {
  url: string
  setUrl: (url: string) => void
  onProcess: (force: boolean) => void
  isProcessing: boolean
  backendStatus: 'connected' | 'waking' | 'error'
  forceReindex: boolean
  setForceReindex: (val: boolean) => void
}

export const Navbar: React.FC<NavbarProps> = ({
  url,
  setUrl,
  onProcess,
  isProcessing,
  backendStatus,
  forceReindex,
  setForceReindex,
}) => {
  const [isFocused, setIsFocused] = useState(false)

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (url.trim() && !isProcessing) {
      onProcess(forceReindex)
      document.getElementById('studio-workspace')?.scrollIntoView({ behavior: 'smooth' })
    }
  }

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text) {
        setUrl(text.trim())
      }
    } catch {
      // Clipboard permissions denied, ignore silently
    }
  }

  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <header className="sticky top-0 z-50 w-full px-3 sm:px-6 py-2.5 sm:py-3 apple-glass border-b border-white/[0.08]">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-3 sm:gap-6">
        {/* Brand & Nav Anchors */}
        <div className="flex items-center gap-6">
          <div
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="flex items-center gap-2.5 cursor-pointer group"
          >
            <div className="w-8 h-8 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center shadow-md group-hover:border-white/30 transition-colors">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-sm text-white tracking-tight leading-tight">YouTube QnA</h1>
              <p className="text-[10px] text-zinc-400 font-mono tracking-wider uppercase">Studio Edition</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center gap-5 text-xs font-medium text-zinc-400">
            <button
              onClick={() => scrollTo('storyline')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Storyline
            </button>
            <button
              onClick={() => scrollTo('architecture')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Architecture
            </button>
            <button
              onClick={() => scrollTo('studio-workspace')}
              className="hover:text-white transition-colors cursor-pointer"
            >
              Studio
            </button>
          </nav>
        </div>

        {/* Dynamic Island Search Bar */}
        <form
          onSubmit={handleSubmit}
          className={`flex items-center flex-1 max-w-md sm:max-w-lg px-2 py-1 rounded-full transition-transform duration-100 ${
            isFocused
              ? 'bg-zinc-900 ring-1 ring-white/25 shadow-xl shadow-black/80'
              : 'bg-zinc-950/80 border border-white/[0.08]'
          }`}
        >
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Paste YouTube link..."
            className="w-full bg-transparent text-base sm:text-xs text-zinc-100 placeholder:text-zinc-500 px-2.5 outline-none"
          />

          {url && (
            <button
              type="button"
              onClick={() => setUrl('')}
              className="p-1 rounded-full text-zinc-400 hover:text-white mr-1 active:scale-95 transition-transform"
              title="Clear input"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}

          {!url && (
            <button
              type="button"
              onClick={handlePaste}
              className="text-[10px] text-zinc-400 hover:text-white px-2 py-0.5 rounded-md bg-white/[0.05] hover:bg-white/[0.1] mr-1 active:scale-95 transition-transform font-mono"
            >
              Paste
            </button>
          )}

          <button
            type="submit"
            disabled={!url.trim() || isProcessing}
            className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium transition-transform duration-100 shrink-0 ${
              url.trim() && !isProcessing
                ? 'bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md shadow-white/10 cursor-pointer'
                : 'bg-white/[0.06] text-zinc-600 cursor-not-allowed'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span className="hidden sm:inline">Indexing...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <ArrowRight className="w-3 h-3" />
              </>
            )}
          </button>
        </form>

        {/* Server Status Pill & Re-index */}
        <div className="flex items-center gap-3">
          <label className="hidden xl:flex items-center gap-1.5 text-zinc-400 hover:text-zinc-200 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={forceReindex}
              onChange={(e) => setForceReindex(e.target.checked)}
              className="rounded bg-zinc-900 border-zinc-700 text-white focus:ring-0 w-3 h-3 accent-white cursor-pointer"
            />
            <span className="text-[11px] font-mono">Re-index</span>
          </label>

          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-[11px]">
            {backendStatus === 'connected' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-white shadow-sm shadow-white" />
                <span className="text-zinc-300 hidden sm:inline">Server Live</span>
              </>
            )}
            {backendStatus === 'waking' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-ping" />
                <span className="text-zinc-400 hidden sm:inline">Waking...</span>
              </>
            )}
            {backendStatus === 'error' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                <span className="text-zinc-500 hidden sm:inline">Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
