import React, { useState, useEffect, useRef, useMemo } from 'react'
import type { TimelineBeat } from '../types'
import { Search, X, Play, Compass, Crosshair, Sparkles } from 'lucide-react'

interface TimelineStreamProps {
  timeline: TimelineBeat[]
  currentTime: number
  isPlaying?: boolean
  onSeek: (seconds: number) => void
  isLoading: boolean
}

export const TimelineStream: React.FC<TimelineStreamProps> = ({
  timeline,
  currentTime,
  isPlaying = false,
  onSeek,
  isLoading,
}) => {
  const [searchQuery, setSearchQuery] = useState('')
  const [isAutoFollow, setIsAutoFollow] = useState(true)

  const containerRef = useRef<HTMLDivElement>(null)
  const beatElementsRef = useRef<Map<number, HTMLDivElement>>(new Map())

  // Find index of currently active beat based on currentTime
  const activeBeatIndex = useMemo(() => {
    if (!timeline.length) return -1
    let matched = -1
    for (let i = 0; i < timeline.length; i++) {
      if (timeline[i].start <= currentTime + 0.5) {
        matched = i
      } else {
        break
      }
    }
    return matched
  }, [timeline, currentTime])

  // Filter beats by search query
  const filteredBeatsWithIndex = useMemo(() => {
    const list = timeline.map((beat, originalIndex) => ({ beat, originalIndex }))
    if (!searchQuery.trim()) return list
    const q = searchQuery.toLowerCase().trim()
    return list.filter(
      ({ beat }) =>
        beat.text.toLowerCase().includes(q) || beat.timestamp.includes(q)
    )
  }, [timeline, searchQuery])

  // Auto-scroll to active beat when activeBeatIndex updates and isAutoFollow is active
  useEffect(() => {
    if (!isAutoFollow || activeBeatIndex < 0) return

    const activeEl = beatElementsRef.current.get(activeBeatIndex)
    if (activeEl && containerRef.current) {
      activeEl.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      })
    }
  }, [activeBeatIndex, isAutoFollow])

  // Detect user manual scroll to detach auto-follow
  const handleUserScroll = () => {
    if (isAutoFollow) {
      setIsAutoFollow(false)
    }
  }

  // Re-center on active beat and re-enable auto-follow
  const handleRecenter = () => {
    setIsAutoFollow(true)
    if (activeBeatIndex >= 0) {
      const activeEl = beatElementsRef.current.get(activeBeatIndex)
      if (activeEl) {
        activeEl.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        })
      }
    }
  }

  // Highlight matching search query text
  const renderHighlightedText = (text: string, query: string) => {
    if (!query.trim()) return text
    const parts = text.split(new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi'))
    return (
      <>
        {parts.map((part, i) =>
          part.toLowerCase() === query.toLowerCase() ? (
            <mark
              key={i}
              className="bg-white/25 text-white font-medium rounded-xs px-0.5 border-b border-white/40"
            >
              {part}
            </mark>
          ) : (
            part
          )
        )}
      </>
    )
  }

  return (
    <div className="flex flex-col h-[650px] apple-glass rounded-2xl border border-white/[0.08] overflow-hidden shadow-2xl relative">
      {/* Sticky Top Control Toolbar */}
      <div className="p-3 sm:p-4 border-b border-white/[0.08] bg-zinc-950/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-3 z-10">
        {/* Search input */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 pointer-events-none" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search dialogue & timestamps..."
            className="w-full pl-8.5 pr-8 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs text-white placeholder-zinc-500 focus:outline-none focus:border-white/30 transition-colors"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-white p-0.5"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Status / Auto-scroll toggle controls */}
        <div className="flex items-center gap-2">
          {isPlaying && (
            <span className="hidden sm:inline-flex items-center gap-1.5 text-[10px] uppercase font-mono tracking-wider text-zinc-400 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08]">
              <span className="w-1.5 h-1.5 rounded-full bg-white shadow-[0_0_6px_rgba(255,255,255,0.8)] animate-pulse" />
              Playing
            </span>
          )}

          <span className="text-[11px] text-zinc-400 font-mono hidden sm:inline">
            {filteredBeatsWithIndex.length} / {timeline.length} beats
          </span>

          <button
            onClick={() => {
              if (!isAutoFollow) {
                handleRecenter()
              } else {
                setIsAutoFollow(false)
              }
            }}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono tracking-tight border transition-colors ${
              isAutoFollow
                ? 'bg-white text-black border-white shadow-sm'
                : 'bg-white/[0.04] text-zinc-400 border-white/[0.08] hover:text-white'
            }`}
            title={isAutoFollow ? 'Click to pause auto-follow' : 'Click to resume playhead auto-follow'}
          >
            <Crosshair className={`w-3 h-3 ${isAutoFollow ? 'text-black' : 'text-zinc-400'}`} />
            <span>{isAutoFollow ? 'Auto-Sync' : 'Free Scroll'}</span>
          </button>
        </div>
      </div>

      {/* Loading Skeleton */}
      {isLoading && (
        <div className="flex-1 p-6 space-y-5 animate-pulse">
          {[1, 2, 3, 4, 5].map((i) => (
            <div key={i} className="flex gap-4 items-start">
              <div className="w-14 h-5 bg-white/[0.06] rounded-md shrink-0" />
              <div className="flex-1 space-y-2">
                <div className="h-3.5 bg-white/[0.06] rounded w-3/4" />
                <div className="h-3 bg-white/[0.04] rounded w-1/2" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && timeline.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-zinc-900 border border-white/10 flex items-center justify-center">
            <Sparkles className="w-5 h-5 text-zinc-400" />
          </div>
          <p className="text-sm font-medium text-white">No transcript beats available</p>
          <p className="text-xs text-zinc-400 max-w-sm">
            Once video processing finishes, chronological dialogue timestamps will appear here for scroll-synchronized scrubbing.
          </p>
        </div>
      )}

      {/* Search has no matches */}
      {!isLoading && timeline.length > 0 && filteredBeatsWithIndex.length === 0 && (
        <div className="flex-1 flex flex-col items-center justify-center p-8 text-center gap-2">
          <p className="text-sm text-zinc-300">No dialogue matching "{searchQuery}"</p>
          <button
            onClick={() => setSearchQuery('')}
            className="text-xs text-zinc-400 hover:text-white underline underline-offset-4"
          >
            Clear filter
          </button>
        </div>
      )}

      {/* Scrollable Timeline Stream */}
      {!isLoading && filteredBeatsWithIndex.length > 0 && (
        <div
          ref={containerRef}
          onWheel={handleUserScroll}
          onTouchMove={handleUserScroll}
          className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-3 sm:space-y-4 scroll-smooth relative"
        >
          {/* Vertical spine guide rail */}
          <div className="absolute left-[2.85rem] sm:left-[3.35rem] top-6 bottom-6 w-px bg-white/[0.06] pointer-events-none" />

          {filteredBeatsWithIndex.map(({ beat, originalIndex }) => {
            const isActive = originalIndex === activeBeatIndex

            return (
              <div
                key={beat.chunk_id || `beat-${originalIndex}`}
                ref={(el) => {
                  if (el) {
                    beatElementsRef.current.set(originalIndex, el)
                  } else {
                    beatElementsRef.current.delete(originalIndex)
                  }
                }}
                onClick={() => onSeek(beat.start)}
                className={`group relative flex items-start gap-3.5 sm:gap-4 p-3 sm:p-3.5 rounded-xl cursor-pointer transition-all duration-150 border ${
                  isActive
                    ? 'bg-white/[0.06] border-white/20 shadow-lg shadow-black/40 scale-[1.005]'
                    : 'bg-transparent border-transparent hover:bg-white/[0.025] hover:border-white/[0.05]'
                }`}
              >
                {/* Timestamp Pill & Active Dot */}
                <div className="flex items-center gap-2 shrink-0 pt-0.5">
                  {/* Status Indicator Dot */}
                  <div className="relative flex items-center justify-center w-3 h-3">
                    {isActive ? (
                      <>
                        <span className="absolute w-3 h-3 rounded-full bg-white/30 animate-ping" />
                        <span className="w-2 h-2 rounded-full bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)]" />
                      </>
                    ) : (
                      <span className="w-1.5 h-1.5 rounded-full bg-zinc-600 group-hover:bg-zinc-400 transition-colors" />
                    )}
                  </div>

                  {/* Timestamp capsule */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      onSeek(beat.start)
                    }}
                    className={`font-mono text-xs px-2 py-0.5 rounded-md flex items-center gap-1 transition-all ${
                      isActive
                        ? 'bg-white text-black font-semibold shadow-xs'
                        : 'bg-white/[0.05] text-zinc-400 group-hover:text-white border border-white/[0.06]'
                    }`}
                  >
                    <Play className={`w-2.5 h-2.5 ${isActive ? 'fill-black text-black' : 'fill-transparent text-zinc-400 group-hover:text-white'}`} />
                    <span>{beat.timestamp}</span>
                  </button>
                </div>

                {/* Beat Text Segment */}
                <div className="flex-1 min-w-0">
                  <p
                    className={`text-xs sm:text-sm leading-relaxed transition-colors ${
                      isActive
                        ? 'text-white font-normal'
                        : 'text-zinc-400 group-hover:text-zinc-200'
                    }`}
                  >
                    {renderHighlightedText(beat.text, searchQuery)}
                  </p>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Detached Auto-Follow Floating Re-center Pill */}
      {!isAutoFollow && activeBeatIndex >= 0 && (
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-20 pointer-events-auto">
          <button
            onClick={handleRecenter}
            className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-zinc-900/95 hover:bg-zinc-800 text-white text-xs font-medium border border-white/20 shadow-2xl backdrop-blur-md active:scale-95 transition-all"
          >
            <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span>Playhead tracking paused — Re-center</span>
            <Compass className="w-3.5 h-3.5 text-zinc-300 ml-0.5" />
          </button>
        </div>
      )}
    </div>
  )
}
