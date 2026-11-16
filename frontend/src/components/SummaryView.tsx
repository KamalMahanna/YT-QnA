import React from 'react'
import { Sparkles, RefreshCw } from 'lucide-react'
import { AudioPlayer } from './AudioPlayer'

interface SummaryViewProps {
  summary: string | null
  isLoadingSummary: boolean
  onGenerateSummary: () => void
  audioBlob: Blob | null
  isLoadingAudio: boolean
  onFetchAudio: () => void
}

export const SummaryView: React.FC<SummaryViewProps> = ({
  summary,
  isLoadingSummary,
  onGenerateSummary,
  audioBlob,
  isLoadingAudio,
  onFetchAudio,
}) => {
  return (
    <div className="flex flex-col gap-4 w-full">
      {/* Audio Player Card */}
      {summary && (
        <AudioPlayer
          audioBlob={audioBlob}
          isLoading={isLoadingAudio}
          onFetchAudio={onFetchAudio}
        />
      )}

      {/* Summary Content Card */}
      <div className="p-6 rounded-2xl apple-glass-card border border-white/[0.08] relative">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <h2 className="text-sm font-semibold text-white tracking-tight">Executive Summary</h2>
          </div>

          {summary && !isLoadingSummary && (
            <button
              onClick={onGenerateSummary}
              className="flex items-center gap-1 text-[11px] text-zinc-400 hover:text-white px-2 py-1 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] transition-colors"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Regenerate</span>
            </button>
          )}
        </div>

        {isLoadingSummary ? (
          <div className="space-y-3 py-2 animate-pulse">
            <div className="h-4 bg-white/[0.08] rounded-md w-3/4" />
            <div className="h-4 bg-white/[0.06] rounded-md w-full" />
            <div className="h-4 bg-white/[0.06] rounded-md w-5/6" />
            <div className="h-4 bg-white/[0.04] rounded-md w-2/3" />
          </div>
        ) : summary ? (
          <div className="prose prose-invert prose-sm max-w-none text-zinc-300 leading-relaxed font-normal whitespace-pre-line text-sm">
            {summary}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-8 text-center gap-3">
            <p className="text-sm text-zinc-400">Generate a concise AI summary of this video's key insights.</p>
            <button
              onClick={onGenerateSummary}
              className="px-4 py-2 rounded-full bg-white text-black font-medium text-xs hover:bg-zinc-200 active:scale-95 transition-all shadow-md shadow-white/10"
            >
              Generate Summary
            </button>
          </div>
        )}
      </div>
    </div>
  )
}
