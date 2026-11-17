import React from 'react'
import {
  Layers,
  ArrowRight,
  Zap,
  Volume2,
  Terminal,
  Database,
  Sliders,
} from 'lucide-react'

export const StorylineSection: React.FC = () => {
  const scrollToStudio = () => {
    document.getElementById('studio-workspace')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <section id="storyline" className="w-full max-w-7xl mx-auto px-4 sm:px-6 py-16 sm:py-24 space-y-20">
      {/* Section Header */}
      <div className="text-center space-y-3 max-w-3xl mx-auto">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.08]">
          <Layers className="w-3.5 h-3.5 text-zinc-300" />
          <span className="text-xs font-mono tracking-widest text-zinc-300 uppercase">The Architecture Storyline</span>
        </div>
        <h2 className="text-2xl sm:text-4xl md:text-5xl font-bold tracking-tight text-white">
          A four-act choreography of video intelligence.
        </h2>
        <p className="text-xs sm:text-sm md:text-base text-zinc-400 leading-relaxed">
          How raw streaming video is ingested, vector-partitioned, scroll-synchronized, and synthesized into grounded answers.
        </p>
      </div>

      {/* The 4 Acts Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 sm:gap-8">
        {/* Act 01 */}
        <div className="p-6 sm:p-8 rounded-3xl apple-glass-card border border-white/[0.08] flex flex-col justify-between gap-6 group hover:border-white/20 transition-all">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-zinc-500 uppercase tracking-wider">Act 01 / Ingestion</span>
              <span className="px-2.5 py-0.5 rounded-full bg-white/[0.05] border border-white/[0.08] text-[10px] font-mono text-zinc-300">
                Single-Flight Mutex
              </span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Zero-Stampede Neural Pipeline
            </h3>
            <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
              When 1,000 users paste a viral video link at once, standard backends crash. Our coordinator uses in-memory mutex locks to ensure only one worker processes the transcript while all other concurrent requests await the identical result.
            </p>
          </div>

          {/* Micro visual diagram */}
          <div className="p-4 rounded-2xl bg-black/60 border border-white/[0.06] font-mono text-xs space-y-2.5">
            <div className="flex items-center justify-between text-zinc-400 text-[11px]">
              <span>YouTube Subtitles API</span>
              <span className="text-white">Primary</span>
            </div>
            <div className="h-px bg-white/[0.08] w-full" />
            <div className="flex items-center justify-between text-zinc-400 text-[11px]">
              <span>Groq Whisper (whisper-large-v3)</span>
              <span className="text-zinc-400">Fallback</span>
            </div>
            <div className="h-px bg-white/[0.08] w-full" />
            <div className="flex items-center justify-between text-zinc-300 text-[11px]">
              <span>Matryoshka 768-dim Vectors</span>
              <span className="text-white">Pinecone</span>
            </div>
          </div>
        </div>

        {/* Act 02 */}
        <div className="p-6 sm:p-8 rounded-3xl apple-glass-card border border-white/[0.08] flex flex-col justify-between gap-6 group hover:border-white/20 transition-all">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-zinc-500 uppercase tracking-wider">Act 02 / Synchronization</span>
              <span className="px-2.5 py-0.5 rounded-full bg-white/[0.05] border border-white/[0.08] text-[10px] font-mono text-zinc-300">
                250ms Polling Tick
              </span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              The Scroll-Sync Engine
            </h3>
            <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
              The transcript stream and YouTube playhead dance in lockstep. As dialogue progresses, the active segment glows and scrolls smoothly into center view. Touch or wheel gestures detach tracking instantly, with a one-click re-center compass.
            </p>
          </div>

          {/* Micro visual preview */}
          <div className="p-4 rounded-2xl bg-black/60 border border-white/[0.06] space-y-2">
            <div className="flex items-center gap-3 p-2 rounded-lg bg-white/[0.06] border border-white/20">
              <span className="w-2 h-2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] animate-pulse" />
              <span className="font-mono text-xs bg-white text-black px-1.5 py-0.5 rounded font-bold">10:02</span>
              <span className="text-xs text-white truncate">"We perform formal verification using TLA+..."</span>
            </div>
            <div className="flex items-center gap-3 p-2 rounded-lg bg-transparent opacity-50">
              <span className="w-1.5 h-1.5 rounded-full bg-zinc-600" />
              <span className="font-mono text-xs text-zinc-400 px-1.5 py-0.5 rounded">11:03</span>
              <span className="text-xs text-zinc-400 truncate">"Checking for flickering or jittering in virtualization..."</span>
            </div>
          </div>
        </div>

        {/* Act 03 */}
        <div className="p-6 sm:p-8 rounded-3xl apple-glass-card border border-white/[0.08] flex flex-col justify-between gap-6 group hover:border-white/20 transition-all">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-zinc-500 uppercase tracking-wider">Act 03 / Grounding</span>
              <span className="px-2.5 py-0.5 rounded-full bg-white/[0.05] border border-white/[0.08] text-[10px] font-mono text-zinc-300">
                Zero Citation Bloat
              </span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Conversational RAG with Inline Pills
            </h3>
            <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
              No hallucinated sources or messy bottom drawers. Every factual assertion is embedded with clickable time capsules like <span className="font-mono text-xs text-white bg-white/10 px-1.5 py-0.5 rounded">▶ 10:02</span>. Clicking any capsule instantly scrubs the video player directly to that second.
            </p>
          </div>

          {/* Micro visual */}
          <div className="p-4 rounded-2xl bg-black/60 border border-white/[0.06] font-sans text-xs space-y-2 text-zinc-300 leading-relaxed">
            <p>
              "The distributed consensus protocol leverages Raft{' '}
              <span className="inline-flex items-center gap-1 font-mono text-[11px] text-zinc-200 bg-white/10 px-1.5 py-0.5 rounded-md border border-white/20">
                ▶ 04:32
              </span>{' '}
              to elect leaders within milliseconds."
            </p>
          </div>
        </div>

        {/* Act 04 */}
        <div className="p-6 sm:p-8 rounded-3xl apple-glass-card border border-white/[0.08] flex flex-col justify-between gap-6 group hover:border-white/20 transition-all">
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-xs text-zinc-500 uppercase tracking-wider">Act 04 / Synthesis</span>
              <span className="px-2.5 py-0.5 rounded-full bg-white/[0.05] border border-white/[0.08] text-[10px] font-mono text-zinc-300">
                Gemini Audio TTS
              </span>
            </div>
            <h3 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Executive Voice Briefings
            </h3>
            <p className="text-xs sm:text-sm text-zinc-400 leading-relaxed">
              Long technical lectures, podcasts, or investor meetings synthesized into crisp executive takeaways. Hit "Generate Audio" to turn key insights into spoken voice briefings with native scrubbing controls.
            </p>
          </div>

          {/* Micro visual */}
          <div className="p-4 rounded-2xl bg-black/60 border border-white/[0.06] flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-white text-black flex items-center justify-center font-bold">
                <Volume2 className="w-4 h-4 text-black" />
              </div>
              <div>
                <p className="text-xs font-medium text-white">Executive Voice Overview</p>
                <p className="text-[10px] font-mono text-zinc-400">Gemini 2.5 Audio • 1m 45s</p>
              </div>
            </div>
            <span className="text-xs font-mono text-zinc-400">24kHz PCM</span>
          </div>
        </div>
      </div>

      {/* Tech Specifications Bento */}
      <div id="architecture" className="p-6 sm:p-10 rounded-3xl bg-zinc-950 border border-white/[0.08] space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-white/[0.08] pb-6">
          <div>
            <h3 className="text-lg sm:text-xl font-bold text-white">Production Stack Architecture</h3>
            <p className="text-xs sm:text-sm text-zinc-400">Engineered for Render Free Tier (&lt;50MB RAM footprint) and universal vector caching.</p>
          </div>
          <button
            onClick={scrollToStudio}
            className="self-start sm:self-auto flex items-center gap-2 px-4 py-2 rounded-full bg-white text-black text-xs font-medium hover:bg-zinc-200 transition-colors cursor-pointer"
          >
            <span>Launch Studio</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] space-y-1.5">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Database className="w-4 h-4 text-zinc-400" />
              <span>Pinecone Serverless</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Global persistent vector index with isolated per-video namespaces and 768-dim Matryoshka embeddings.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] space-y-1.5">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Terminal className="w-4 h-4 text-zinc-400" />
              <span>FastAPI & asyncio</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Lightweight ASGI web service running on Python 3.13. SSE token streaming and offloaded thread pools.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] space-y-1.5">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Zap className="w-4 h-4 text-zinc-400" />
              <span>Groq Whisper Fallback</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Automatic fallback to whisper-large-v3 when videos have subtitles disabled or language barriers.
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white/[0.02] border border-white/[0.05] space-y-1.5">
            <div className="flex items-center gap-2 text-white font-semibold">
              <Sliders className="w-4 h-4 text-zinc-400" />
              <span>React 19 & Tailwind 4</span>
            </div>
            <p className="text-zinc-400 leading-relaxed">
              Pure monochrome design system, fluid spring animations, zero-lag composite-only micro-interactions.
            </p>
          </div>
        </div>
      </div>

      {/* Transition to Studio CTA */}
      <div className="flex flex-col items-center justify-center p-8 sm:p-12 rounded-3xl apple-glass text-center gap-4 border border-white/[0.1]">
        <h3 className="text-xl sm:text-3xl font-bold text-white tracking-tight">
          Ready to experience your video?
        </h3>
        <p className="text-xs sm:text-sm text-zinc-400 max-w-md">
          Scroll down into the Studio workspace below or paste a YouTube URL to begin neural analysis.
        </p>
        <button
          onClick={scrollToStudio}
          className="mt-2 flex items-center gap-2 px-6 py-3 rounded-full bg-white text-black font-medium text-xs sm:text-sm hover:bg-zinc-200 active:scale-95 transition-all shadow-xl shadow-white/10 cursor-pointer"
        >
          <span>Open Interactive Studio</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </section>
  )
}
