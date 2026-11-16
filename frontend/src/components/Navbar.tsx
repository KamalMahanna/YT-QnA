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

  return (
    <header className="sticky top-0 z-50 w-full px-4 py-3 apple-glass border-b border-white/[0.08]">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Logo and Backend Status Badge */}
        <div className="flex items-center gap-3 w-full md:w-auto justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-tr from-red-600 to-rose-400 flex items-center justify-center shadow-lg shadow-red-500/20">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-sm text-white tracking-tight">YouTube QnA</h1>
              <p className="text-[11px] text-zinc-400">Apple Fluid Intelligence</p>
            </div>
          </div>

          {/* Ambient Backend Wake Badge */}
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.06] text-[11px]">
            {backendStatus === 'connected' && (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500 shadow-sm shadow-emerald-500/50" />
                <span className="text-zinc-300">Server Ready</span>
              </>
            )}
            {backendStatus === 'waking' && (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
                <span className="text-amber-200">Waking Render backend...</span>
              </>
            )}
            {backendStatus === 'error' && (
              <>
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                <span className="text-rose-300">Offline</span>
              </>
            )}
          </div>
        </div>

        {/* Dynamic Island Search Pill */}
        <form
          onSubmit={handleSubmit}
          className={`flex items-center w-full md:max-w-xl px-2 py-1.5 rounded-full transition-all duration-200 ${
            isFocused
              ? 'bg-zinc-800/80 ring-1 ring-white/20 shadow-xl shadow-black/40'
              : 'bg-zinc-900/60 border border-white/[0.08]'
          }`}
        >
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Paste YouTube video link..."
            className="w-full bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 px-3 outline-none"
          />

          {url && (
            <button
              type="button"
              onClick={() => setUrl('')}
              className="p-1 rounded-full text-zinc-400 hover:text-white mr-1 transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}

          {!url && (
            <button
              type="button"
              onClick={handlePaste}
              className="text-[11px] text-zinc-400 hover:text-white px-2 py-0.5 rounded-md bg-white/[0.05] hover:bg-white/[0.1] mr-1.5 transition-colors"
            >
              Paste
            </button>
          )}

          <button
            type="submit"
            disabled={!url.trim() || isProcessing}
            className={`flex items-center gap-1 px-4 py-1.5 rounded-full text-xs font-medium transition-all duration-150 ${
              url.trim() && !isProcessing
                ? 'bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md shadow-white/10'
                : 'bg-white/10 text-zinc-500 cursor-not-allowed'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-3 h-3 animate-spin" />
                <span>Indexing...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <ArrowRight className="w-3 h-3" />
              </>
            )}
          </button>
        </form>

        {/* Options */}
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={forceReindex}
              onChange={(e) => setForceReindex(e.target.checked)}
              className="rounded bg-zinc-800 border-zinc-700 text-white focus:ring-0 w-3.5 h-3.5 accent-white"
            />
            <span>Re-index</span>
          </label>
        </div>
      </div>
    </header>
  )
}
