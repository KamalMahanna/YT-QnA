import React, { useEffect, useRef, useState } from 'react'
import { Play, Pause } from 'lucide-react'

interface AudioPlayerProps {
  audioBlob: Blob | null
  isLoading: boolean
  onFetchAudio: () => void
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioBlob,
  isLoading,
  onFetchAudio,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [playbackRate, setPlaybackRate] = useState(1)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  useEffect(() => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob)
      setAudioUrl(url)
      return () => URL.revokeObjectURL(url)
    } else {
      setAudioUrl(null)
    }
  }, [audioBlob])

  const togglePlay = () => {
    if (!audioUrl) {
      onFetchAudio()
      return
    }

    if (!audioRef.current) return

    if (isPlaying) {
      audioRef.current.pause()
      setIsPlaying(false)
    } else {
      audioRef.current.play()
      setIsPlaying(true)
    }
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value)
    setCurrentTime(time)
    if (audioRef.current) {
      audioRef.current.currentTime = time
    }
  }

  const cycleSpeed = () => {
    const speeds = [1, 1.25, 1.5, 2]
    const nextSpeed = speeds[(speeds.indexOf(playbackRate) + 1) % speeds.length]
    setPlaybackRate(nextSpeed)
    if (audioRef.current) {
      audioRef.current.playbackRate = nextSpeed
    }
  }

  const formatSec = (sec: number) => {
    const mins = Math.floor(sec / 60)
    const secs = Math.floor(sec % 60)
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`
  }

  return (
    <div className="w-full p-3 rounded-2xl apple-glass-card flex flex-col gap-2">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
      )}

      <div className="flex items-center justify-between gap-3">
        {/* Play / Action Button */}
        <button
          onClick={togglePlay}
          disabled={isLoading}
          className="flex items-center justify-center w-10 h-10 rounded-full bg-white text-black hover:bg-zinc-200 active:scale-95 transition-all shadow-md shadow-white/10 shrink-0"
        >
          {isLoading ? (
            <div className="w-4 h-4 border-2 border-black border-t-transparent rounded-full animate-spin" />
          ) : isPlaying ? (
            <Pause className="w-4 h-4 fill-black" />
          ) : (
            <Play className="w-4 h-4 fill-black ml-0.5" />
          )}
        </button>

        {/* Scrub bar & Timeline */}
        <div className="flex-1 flex flex-col justify-center gap-1">
          <div className="flex justify-between items-center text-[11px] text-zinc-400">
            <span className="font-medium text-zinc-300">
              {audioUrl ? 'Gemini Voice Summary' : 'Listen to Audio Summary'}
            </span>
            <span>
              {formatSec(currentTime)} / {formatSec(duration)}
            </span>
          </div>

          <input
            type="range"
            min="0"
            max={duration || 100}
            value={currentTime}
            onChange={handleSeek}
            disabled={!audioUrl}
            className="w-full h-1.5 bg-zinc-700/60 rounded-lg appearance-none cursor-pointer accent-white"
          />
        </div>

        {/* Speed button */}
        {audioUrl && (
          <button
            onClick={cycleSpeed}
            className="px-2 py-1 rounded-lg bg-white/[0.08] hover:bg-white/[0.14] text-[11px] font-semibold text-zinc-300 active:scale-95 transition-all"
          >
            {playbackRate}x
          </button>
        )}
      </div>
    </div>
  )
}
