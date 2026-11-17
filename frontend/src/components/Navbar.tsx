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
    <header className="sticky top-0 z-50 w-full px-3 sm:px-6 py-2.5 sm:py-3.5 apple-glass">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Top Mobile Bar: Brand & Status */}
        <div className="flex items-center justify-between w-full md:w-auto gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-zinc-900 border border-white/10 flex items-center justify-center shadow-md">
              <Sparkles className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-sm text-white tracking-tight leading-tight">YouTube QnA</h1>
              <p className="text-[10px] text-zinc-400 font-mono tracking-wider uppercase">Monochrome Engine</p>
            </div>
          </div>

          {/* Minimalist Monochrome Status Indicator */}
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-[11px]">
            {backendStatus === 'connected' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-white shadow-sm shadow-white" />
                <span className="text-zinc-300">Ready</span>
              </>
            )}
            {backendStatus === 'waking' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-400 animate-ping" />
                <span className="text-zinc-400">Waking Server...</span>
              </>
            )}
            {backendStatus === 'error' && (
              <>
                <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
                <span className="text-zinc-500">Offline</span>
              </>
            )}
          </div>
        </div>

        {/* Dynamic Island Search Input */}
        <form
          onSubmit={handleSubmit}
          className={`flex items-center w-full md:max-w-xl px-2 py-1.5 rounded-full transition-transform duration-100 ${
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
            placeholder="Paste YouTube video link..."
            className="w-full bg-transparent text-base sm:text-sm text-zinc-100 placeholder:text-zinc-500 px-3 outline-none"
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
              className="text-[11px] text-zinc-400 hover:text-white px-2 py-0.5 rounded-md bg-white/[0.05] hover:bg-white/[0.1] mr-1.5 active:scale-95 transition-transform"
            >
              Paste
            </button>
          )}

          <button
            type="submit"
            disabled={!url.trim() || isProcessing}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-medium transition-transform duration-100 shrink-0 ${
              url.trim() && !isProcessing
                ? 'bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md shadow-white/10'
                : 'bg-white/[0.06] text-zinc-600 cursor-not-allowed'
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
        <div className="hidden md:flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-zinc-400 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={forceReindex}
              onChange={(e) => setForceReindex(e.target.checked)}
              className="rounded bg-zinc-900 border-zinc-700 text-white focus:ring-0 w-3.5 h-3.5 accent-white cursor-pointer"
            />
            <span>Re-index</span>
          </label>
        </div>
      </div>
    </header>
  )
}
