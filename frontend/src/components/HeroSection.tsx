import React, { useState } from 'react'
import {
  ArrowRight,
  RefreshCw,
  X,
  Link2,
  Play,
  ChevronDown,
  MessageSquare,
  Clock,
} from 'lucide-react'

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
    name: 'Karpathy: Intro to LLMs',
    url: 'https://www.youtube.com/watch?v=zjkBMFhNj_g',
  },
  {
    name: 'Veritasium: Math Paradox',
    url: 'https://www.youtube.com/watch?v=HeQX2HjkcNo',
  },
  {
    name: 'Jobs: iPhone Keynote 2007',
    url: 'https://www.youtube.com/watch?v=MnrJzXM7U6o',
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

  const scrollToStudio = () => {
    document.getElementById('studio-workspace')?.scrollIntoView({ behavior: 'smooth' })
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
    <section className="relative w-full min-h-[92vh] flex flex-col items-center justify-between pt-16 sm:pt-24 pb-12 px-4 sm:px-6 overflow-hidden text-center">
      {/* Background ambient monochrome radiance */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] sm:w-[900px] h-[350px] bg-white/[0.03] rounded-full blur-[160px] pointer-events-none" />

      {/* Main Hero Header Stack */}
      <div className="flex flex-col items-center max-w-4xl mx-auto w-full z-10 space-y-6 sm:space-y-8">
        {/* Subtle Top Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] backdrop-blur-md">
          <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.9)] animate-pulse" />
          <span className="text-[11px] font-mono tracking-widest text-zinc-300 uppercase">
            Studio Engine 2.0 • Pinecone Serverless
          </span>
        </div>

        {/* Crisp Editorial Headline */}
        <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold tracking-tight text-white leading-[1.08]">
          Turn hours of video into <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-b from-white via-zinc-200 to-zinc-500">
            instant intelligence.
          </span>
        </h1>

        {/* Elegant Subtitle */}
        <p className="text-sm sm:text-base md:text-lg text-zinc-400 max-w-2xl mx-auto font-normal leading-relaxed">
          Scroll-synchronized playhead tracking, conversational RAG with grounded citations, and executive voice briefings. In pure monochrome.
        </p>

        {/* Clean Command Input Island */}
        <div className="w-full max-w-xl mx-auto pt-2">
          <form
            onSubmit={handleSubmit}
            className={`relative flex items-center p-1.5 rounded-full backdrop-blur-xl transition-all duration-150 ${
              isFocused
                ? 'bg-zinc-900/90 border border-white/30 shadow-2xl shadow-black/90 ring-1 ring-white/10'
                : 'bg-zinc-950/80 border border-white/[0.09] shadow-xl'
            }`}
          >
            <div className="pl-3.5 pr-2 text-zinc-500 pointer-events-none">
              <Link2 className="w-4 h-4" />
            </div>

            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onFocus={() => setIsFocused(true)}
              onBlur={() => setIsFocused(false)}
              placeholder="Paste any YouTube video link..."
              className="flex-1 bg-transparent text-sm sm:text-base text-white placeholder:text-zinc-500 px-1 outline-none"
            />

            {url ? (
              <button
                type="button"
                onClick={() => setUrl('')}
                className="p-1 rounded-full text-zinc-400 hover:text-white mr-1 active:scale-95 transition-transform"
                title="Clear"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            ) : (
              <button
                type="button"
                onClick={handlePaste}
                className="text-[11px] text-zinc-400 hover:text-white px-2 py-1 rounded-md bg-white/[0.04] hover:bg-white/[0.08] mr-1.5 active:scale-95 transition-transform font-mono"
              >
                Paste
              </button>
            )}

            <button
              type="submit"
              disabled={!url.trim() || isProcessing}
              className={`flex items-center gap-1.5 px-4 sm:px-5 py-2 rounded-full text-xs sm:text-sm font-medium transition-transform duration-100 shrink-0 ${
                url.trim() && !isProcessing
                  ? 'bg-white text-black hover:bg-zinc-200 active:scale-95 shadow-md shadow-white/10 cursor-pointer'
                  : 'bg-white/[0.06] text-zinc-600 cursor-not-allowed'
              }`}
            >
              {isProcessing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Indexing</span>
                </>
              ) : (
                <>
                  <span>Analyze</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </form>

          {/* Quick Curated Samples Ribbon & Options */}
          <div className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1.5 pt-3.5 text-xs text-zinc-500 font-sans">
            <span className="text-zinc-500 text-[11px]">Try a lecture:</span>
            {SAMPLE_VIDEOS.map((sample, i) => (
              <React.Fragment key={i}>
                <button
                  type="button"
                  onClick={() => handleSelectSample(sample.url)}
                  disabled={isProcessing}
                  className="text-zinc-400 hover:text-white underline-offset-4 hover:underline transition-colors cursor-pointer text-[12px] active:scale-95"
                >
                  {sample.name}
                </button>
                <span className="text-zinc-700 select-none">•</span>
              </React.Fragment>
            ))}

            <label className="flex items-center gap-1.5 text-zinc-500 hover:text-zinc-300 cursor-pointer select-none text-[11px] ml-1">
              <input
                type="checkbox"
                checked={forceReindex}
                onChange={(e) => setForceReindex(e.target.checked)}
                className="rounded bg-zinc-900 border-zinc-700 text-white focus:ring-0 w-3 h-3 accent-white cursor-pointer"
              />
              <span>Re-index</span>
            </label>
          </div>
        </div>

        {/* Interactive Floating Product Frame (The Industry Showcase Teaser) */}
        <div
          onClick={scrollToStudio}
          className="w-full max-w-3xl mt-6 rounded-2xl apple-glass-card border border-white/[0.08] p-3 sm:p-4 text-left shadow-2xl shadow-black/80 cursor-pointer group hover:border-white/20 transition-all duration-200 active:scale-[0.99]"
        >
          {/* Mock Window Controls */}
          <div className="flex items-center justify-between pb-3 border-b border-white/[0.06]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 group-hover:bg-zinc-600 transition-colors" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 group-hover:bg-zinc-600 transition-colors" />
              <span className="w-2.5 h-2.5 rounded-full bg-zinc-700 group-hover:bg-zinc-600 transition-colors" />
              <span className="ml-3 text-[11px] font-mono text-zinc-500 tracking-wider">
                STUDIO WORKSPACE PREVIEW
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-[11px] font-mono text-zinc-400">
              <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)] animate-pulse" />
              <span>04:12 • Synchronized</span>
            </div>
          </div>

          {/* Split Mock Preview */}
          <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 pt-3 items-center">
            {/* Left Mock Player */}
            <div className="sm:col-span-5 aspect-video rounded-xl bg-zinc-950 border border-white/[0.06] flex flex-col items-center justify-center p-3 relative overflow-hidden group-hover:border-white/15 transition-colors">
              <div className="w-9 h-9 rounded-full bg-white/10 flex items-center justify-center text-white mb-2">
                <Play className="w-4 h-4 fill-white" />
              </div>
              <p className="text-[11px] text-zinc-400 font-mono">1080p Video Playhead</p>
              <div className="absolute bottom-2 left-2 right-2 h-1 bg-white/10 rounded-full overflow-hidden">
                <div className="w-1/3 h-full bg-white rounded-full" />
              </div>
            </div>

            {/* Right Mock Sync Stream & RAG Answer */}
            <div className="sm:col-span-7 space-y-2 text-xs">
              {/* Beat Segment */}
              <div className="flex items-center gap-2 p-2 rounded-lg bg-white/[0.04] border border-white/[0.08]">
                <Clock className="w-3 h-3 text-zinc-400 shrink-0" />
                <span className="font-mono text-[10px] bg-white text-black px-1.5 py-0.5 rounded font-bold">
                  04:12
                </span>
                <span className="text-zinc-200 truncate">"Transformers replace recurrent steps with attention..."</span>
              </div>

              {/* Chat Citation */}
              <div className="flex items-start gap-2 p-2 rounded-lg bg-black/60 border border-white/[0.04] text-zinc-300">
                <MessageSquare className="w-3 h-3 text-zinc-400 shrink-0 mt-0.5" />
                <p className="leading-snug text-[11px] text-zinc-400">
                  Self-attention scales linearly across tokens{' '}
                  <span className="inline-flex items-center gap-1 font-mono text-[10px] text-white bg-white/15 px-1 py-0.5 rounded border border-white/20">
                    ▶ 04:12
                  </span>
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Clean Minimalist Spec Divider & Storyline Prompt */}
      <div className="w-full max-w-4xl mx-auto pt-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono text-zinc-500 border-t border-white/[0.06] z-10">
        <div className="flex items-center gap-3 sm:gap-6 text-[11px]">
          <span>0.25s Playhead Sync</span>
          <span className="text-zinc-800">•</span>
          <span>768-dim Matryoshka</span>
          <span className="text-zinc-800">•</span>
          <span>Zero Hallucination</span>
        </div>

        <button
          onClick={scrollToStoryline}
          className="flex items-center gap-1.5 text-zinc-400 hover:text-white transition-colors cursor-pointer group text-[11px]"
        >
          <span>Explore Storyline</span>
          <ChevronDown className="w-3.5 h-3.5 group-hover:translate-y-0.5 transition-transform" />
        </button>
      </div>
    </section>
  )
}
