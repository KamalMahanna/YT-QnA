import React, { useState } from 'react'
import { ArrowRight, Sparkles, RefreshCw, X, ChevronDown, Play, Zap, Cpu, Compass } from 'lucide-react'

interface HeroSectionProps {
  url: string
  setUrl: (url: string) => void
  onProcess: (force: boolean) => void
  isProcessing: boolean
  forceReindex: boolean
  setForceReindex: (val: boolean) => void
}

const SAMPLE_VIDEOS = [
  {
    title: 'Intro to Large Language Models',
    creator: 'Andrej Karpathy',
    url: 'https://www.youtube.com/watch?v=zjkBMFhNj_g',
    duration: '1h 00m',
  },
  {
    title: 'The Math Paradox That Broke Logic',
    creator: 'Veritasium',
    url: 'https://www.youtube.com/watch?v=HeQX2HjkcNo',
    duration: '22m',
  },
  {
    title: 'Original iPhone Keynote 2007',
    creator: 'Steve Jobs',
    url: 'https://www.youtube.com/watch?v=MnrJzXM7U6o',
    duration: '1h 20m',
  },
]

export const HeroSection: React.FC<HeroSectionProps> = ({
  url,
  setUrl,
  onProcess,
  isProcessing,
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

  const handleSelectSample = (sampleUrl: string) => {
    setUrl(sampleUrl)
    onProcess(false)
    document.getElementById('studio-workspace')?.scrollIntoView({ behavior: 'smooth' })
  }

  const scrollToStoryline = () => {
    document.getElementById('storyline')?.scrollIntoView({ behavior: 'smooth' })
  }

  const handlePaste = async () => {
    try {
      const text = await navigator.clipboard.readText()
      if (text) {
        setUrl(text.trim())
      }
    } catch {
      // ignore
    }
  }

  return (
    <section className="relative w-full min-h-[85vh] flex flex-col items-center justify-center pt-8 pb-16 px-4 sm:px-6 overflow-hidden text-center">
      {/* Background ambient monochrome gradient */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[350px] bg-white/[0.02] rounded-full blur-[140px] pointer-events-none" />

      {/* Top Capsule Badge */}
      <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] mb-6 sm:mb-8 animate-fade-in">
        <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]" />
        <span className="text-xs font-mono tracking-wide text-zinc-300 uppercase">
          Studio Engine 2.0 • Pinecone Serverless
        </span>
      </div>

      {/* Main Editorial Headline */}
      <h1 className="text-3xl sm:text-5xl md:text-6xl font-bold tracking-tight text-white max-w-4xl mx-auto leading-[1.12]">
        Turn hours of video into <br className="hidden sm:inline" />
        <span className="bg-clip-text text-transparent bg-gradient-to-b from-white via-zinc-200 to-zinc-500">
          instant intelligence.
        </span>
      </h1>

      {/* Subheadline */}
      <p className="mt-5 text-sm sm:text-base md:text-lg text-zinc-400 max-w-2xl mx-auto leading-relaxed">
        Scroll-synchronized playhead tracking, grounded conversational RAG with zero hallucination, and executive voice briefings. All inside a pure monochrome studio.
      </p>

      {/* Hero Interactive Input Box */}
      <div className="w-full max-w-2xl mt-8 sm:mt-10">
        <form
          onSubmit={handleSubmit}
          className={`flex items-center p-2 rounded-2xl transition-all duration-200 ${
            isFocused
              ? 'bg-zinc-900 border border-white/30 shadow-2xl shadow-black/80 ring-2 ring-white/10'
              : 'bg-zinc-950/90 border border-white/[0.1] shadow-xl'
          }`}
        >
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onFocus={() => setIsFocused(true)}
            onBlur={() => setIsFocused(false)}
            placeholder="Paste any YouTube video link here..."
            className="flex-1 bg-transparent text-sm sm:text-base text-white placeholder:text-zinc-500 px-3.5 outline-none"
          />

          {url ? (
            <button
              type="button"
              onClick={() => setUrl('')}
              className="p-1.5 rounded-full text-zinc-400 hover:text-white mr-1"
              title="Clear"
            >
              <X className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={handlePaste}
              className="text-xs text-zinc-400 hover:text-white px-2.5 py-1 rounded-md bg-white/[0.05] hover:bg-white/[0.1] mr-1.5 transition-colors font-mono"
            >
              Paste
            </button>
          )}

          <button
            type="submit"
            disabled={!url.trim() || isProcessing}
            className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-xs sm:text-sm font-medium transition-all shrink-0 ${
              url.trim() && !isProcessing
                ? 'bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md shadow-white/10 cursor-pointer'
                : 'bg-white/[0.06] text-zinc-600 cursor-not-allowed'
            }`}
          >
            {isProcessing ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Indexing...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </form>

        {/* Re-index switch & options */}
        <div className="flex items-center justify-between px-2 pt-2 text-[11px] text-zinc-500">
          <span>Free Tier Ready • Single-Flight Mutex Coordinated</span>
          <label className="flex items-center gap-1.5 cursor-pointer hover:text-zinc-400 transition-colors">
            <input
              type="checkbox"
              checked={forceReindex}
              onChange={(e) => setForceReindex(e.target.checked)}
              className="rounded bg-zinc-900 border-zinc-700 text-white focus:ring-0 w-3 h-3 accent-white cursor-pointer"
            />
            <span>Force Re-index</span>
          </label>
        </div>
      </div>

      {/* Quick Try Sample Chips */}
      <div className="mt-8 flex flex-col items-center gap-2.5">
        <span className="text-[11px] font-mono tracking-wider uppercase text-zinc-500">
          Or test with instant curated lectures:
        </span>
        <div className="flex flex-wrap items-center justify-center gap-2">
          {SAMPLE_VIDEOS.map((sample, i) => (
            <button
              key={i}
              onClick={() => handleSelectSample(sample.url)}
              disabled={isProcessing}
              className="group flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] hover:border-white/20 text-xs text-zinc-300 hover:text-white transition-all active:scale-95"
            >
              <Play className="w-3 h-3 fill-zinc-400 group-hover:fill-white text-zinc-400 group-hover:text-white transition-colors" />
              <span>{sample.title}</span>
              <span className="font-mono text-[10px] text-zinc-500">({sample.duration})</span>
            </button>
          ))}
        </div>
      </div>

      {/* Studio Specs Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 sm:gap-6 max-w-4xl w-full mt-14 pt-8 border-t border-white/[0.06] text-left">
        <div className="p-3 rounded-xl bg-white/[0.015] border border-white/[0.04]">
          <div className="flex items-center gap-1.5 text-zinc-400 text-xs mb-1">
            <Zap className="w-3.5 h-3.5 text-white" />
            <span>Playhead Sync</span>
          </div>
          <p className="text-base sm:text-lg font-bold font-mono text-white">0.25s</p>
          <p className="text-[11px] text-zinc-500">Bidirectional event tracking</p>
        </div>

        <div className="p-3 rounded-xl bg-white/[0.015] border border-white/[0.04]">
          <div className="flex items-center gap-1.5 text-zinc-400 text-xs mb-1">
            <Cpu className="w-3.5 h-3.5 text-white" />
            <span>Matryoshka Embeddings</span>
          </div>
          <p className="text-base sm:text-lg font-bold font-mono text-white">768-dim</p>
          <p className="text-[11px] text-zinc-500">75% memory reduction</p>
        </div>

        <div className="p-3 rounded-xl bg-white/[0.015] border border-white/[0.04]">
          <div className="flex items-center gap-1.5 text-zinc-400 text-xs mb-1">
            <Compass className="w-3.5 h-3.5 text-white" />
            <span>Serverless Vectors</span>
          </div>
          <p className="text-base sm:text-lg font-bold font-mono text-white">Pinecone</p>
          <p className="text-[11px] text-zinc-500">Isolated video namespaces</p>
        </div>

        <div className="p-3 rounded-xl bg-white/[0.015] border border-white/[0.04]">
          <div className="flex items-center gap-1.5 text-zinc-400 text-xs mb-1">
            <Sparkles className="w-3.5 h-3.5 text-white" />
            <span>Grounded Citations</span>
          </div>
          <p className="text-base sm:text-lg font-bold font-mono text-white">100%</p>
          <p className="text-[11px] text-zinc-500">Clickable timestamp pills</p>
        </div>
      </div>

      {/* Down arrow link to Storyline */}
      <button
        onClick={scrollToStoryline}
        className="mt-12 flex flex-col items-center gap-1 text-xs text-zinc-500 hover:text-white transition-colors cursor-pointer group"
      >
        <span>Explore the Storyline</span>
        <ChevronDown className="w-4 h-4 group-hover:translate-y-1 transition-transform" />
      </button>
    </section>
  )
}
