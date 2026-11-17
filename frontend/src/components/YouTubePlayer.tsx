import React, { useEffect, useRef } from 'react'

interface YouTubePlayerProps {
  videoId: string
  onPlayerReady?: (player: any) => void
  seekTimestamp?: number | null
  onTimeUpdate?: (currentTime: number, isPlaying: boolean) => void
}

declare global {
  interface Window {
    YT: any
    onYouTubeIframeAPIReady: () => void
  }
}

export const YouTubePlayer: React.FC<YouTubePlayerProps> = ({
  videoId,
  onPlayerReady,
  seekTimestamp,
  onTimeUpdate,
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const playerRef = useRef<any>(null)
  const timePollIntervalRef = useRef<any>(null)
  const onTimeUpdateRef = useRef(onTimeUpdate)

  useEffect(() => {
    onTimeUpdateRef.current = onTimeUpdate
  }, [onTimeUpdate])

  const stopTimePolling = () => {
    if (timePollIntervalRef.current) {
      clearInterval(timePollIntervalRef.current)
      timePollIntervalRef.current = null
    }
  }

  const startTimePolling = () => {
    stopTimePolling()
    timePollIntervalRef.current = setInterval(() => {
      if (playerRef.current && typeof playerRef.current.getCurrentTime === 'function') {
        try {
          const time = playerRef.current.getCurrentTime()
          if (typeof time === 'number' && !isNaN(time)) {
            onTimeUpdateRef.current?.(time, true)
          }
        } catch {
          // ignore transient iframe access errors
        }
      }
    }, 250)
  }

  useEffect(() => {
    // Load YouTube Iframe API script if not present
    if (!window.YT) {
      const tag = document.createElement('script')
      tag.src = 'https://www.youtube.com/iframe_api'
      const firstScriptTag = document.getElementsByTagName('script')[0]
      firstScriptTag.parentNode?.insertBefore(tag, firstScriptTag)
    }

    const initPlayer = () => {
      if (!window.YT || !window.YT.Player || !containerRef.current) return

      stopTimePolling()

      // Destroy old instance if any
      if (playerRef.current) {
        try {
          playerRef.current.destroy()
        } catch {
          // ignore
        }
      }

      playerRef.current = new window.YT.Player(containerRef.current, {
        videoId: videoId,
        playerVars: {
          autoplay: 0,
          modestbranding: 1,
          rel: 0,
        },
        events: {
          onReady: (event: any) => {
            if (onPlayerReady) onPlayerReady(event.target)
          },
          onStateChange: (event: any) => {
            // 1: PLAYING, 2: PAUSED, 0: ENDED
            if (event.data === 1) {
              startTimePolling()
            } else {
              stopTimePolling()
              if (playerRef.current && typeof playerRef.current.getCurrentTime === 'function') {
                try {
                  const time = playerRef.current.getCurrentTime()
                  if (typeof time === 'number' && !isNaN(time)) {
                    onTimeUpdateRef.current?.(time, false)
                  }
                } catch {
                  // ignore
                }
              }
            }
          },
        },
      })
    }

    if (window.YT && window.YT.Player) {
      initPlayer()
    } else {
      window.onYouTubeIframeAPIReady = initPlayer
    }

    return () => {
      stopTimePolling()
      if (playerRef.current && playerRef.current.destroy) {
        try {
          playerRef.current.destroy()
        } catch {
          // ignore
        }
      }
    }
  }, [videoId])

  // Scrub when seekTimestamp prop updates
  useEffect(() => {
    if (seekTimestamp !== null && seekTimestamp !== undefined && playerRef.current && playerRef.current.seekTo) {
      try {
        playerRef.current.seekTo(seekTimestamp, true)
        playerRef.current.playVideo()
        onTimeUpdateRef.current?.(seekTimestamp, true)
      } catch {
        // ignore
      }
    }
  }, [seekTimestamp])

  return (
    <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl shadow-black/80 border border-white/[0.08] bg-black">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}
