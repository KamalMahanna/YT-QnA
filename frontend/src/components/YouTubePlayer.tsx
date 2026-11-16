import React, { useEffect, useRef } from 'react'

interface YouTubePlayerProps {
  videoId: string
  onPlayerReady?: (player: any) => void
  seekTimestamp?: number | null
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
}) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const playerRef = useRef<any>(null)

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

      // Destroy old instance if any
      if (playerRef.current) {
        playerRef.current.destroy()
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
        },
      })
    }

    if (window.YT && window.YT.Player) {
      initPlayer()
    } else {
      window.onYouTubeIframeAPIReady = initPlayer
    }

    return () => {
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
      playerRef.current.seekTo(seekTimestamp, true)
      playerRef.current.playVideo()
    }
  }, [seekTimestamp])

  return (
    <div className="relative w-full aspect-video rounded-2xl overflow-hidden shadow-2xl shadow-black/80 border border-white/[0.08] bg-black">
      <div ref={containerRef} className="w-full h-full" />
    </div>
  )
}
