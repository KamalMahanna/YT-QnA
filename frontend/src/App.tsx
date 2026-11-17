import React, { useState, useEffect } from 'react'
import { Navbar } from './components/Navbar'
import { YouTubePlayer } from './components/YouTubePlayer'
import { ChatView } from './components/ChatView'
import { SummaryView } from './components/SummaryView'
import { TimelineStream } from './components/TimelineStream'
import type { ChatMessage, VideoStatus, TimelineBeat } from './types'
import { MessageSquare, FileText, AlertCircle, Sparkles, Clock } from 'lucide-react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ''

const formatSeconds = (sec: number) => {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`
}

export const App: React.FC = () => {
  const [url, setUrl] = useState('')
  const [videoId, setVideoId] = useState<string | null>(null)
  const [forceReindex, setForceReindex] = useState(false)
  const [activeTab, setActiveTab] = useState<'timeline' | 'chat' | 'summary'>('timeline')

  // Backend connection & health
  const [backendStatus, setBackendStatus] = useState<'connected' | 'waking' | 'error'>('waking')
  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  // YouTube playhead sync & seeking
  const [currentTime, setCurrentTime] = useState<number>(0)
  const [isPlaying, setIsPlaying] = useState<boolean>(false)
  const [seekTimestamp, setSeekTimestamp] = useState<number | null>(null)

  // Timeline beats state
  const [timeline, setTimeline] = useState<TimelineBeat[]>([])
  const [isLoadingTimeline, setIsLoadingTimeline] = useState(false)

  // Chat state & persistence
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingTokens, setStreamingTokens] = useState('')

  // Summary & TTS Audio state
  const [summary, setSummary] = useState<string | null>(null)
  const [isLoadingSummary, setIsLoadingSummary] = useState(false)
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null)
  const [isLoadingAudio, setIsLoadingAudio] = useState(false)

  // Health check on mount (wake up Render backend)
  useEffect(() => {
    let isMounted = true

    const checkHealth = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/health`)
        if (res.ok && isMounted) {
          setBackendStatus('connected')
        } else if (isMounted) {
          setBackendStatus('waking')
        }
      } catch {
        if (isMounted) setBackendStatus('waking')
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 15000)
    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [])

  // Load chat & summary cache from localStorage when videoId changes
  useEffect(() => {
    if (!videoId) return

    setTimeline([])
    setCurrentTime(0)
    setIsPlaying(false)

    const storedChat = localStorage.getItem(`yt_chat_${videoId}`)
    if (storedChat) {
      try {
        setMessages(JSON.parse(storedChat))
      } catch {
        setMessages([])
      }
    } else {
      setMessages([])
    }

    const storedSummary = localStorage.getItem(`yt_summary_${videoId}`)
    if (storedSummary) {
      setSummary(storedSummary)
    } else {
      setSummary(null)
    }
    setAudioBlob(null)
  }, [videoId])

  // Fetch chronological timeline when video is ready
  useEffect(() => {
    if (!videoId || videoStatus?.status !== 'ready') return

    const fetchTimeline = async () => {
      setIsLoadingTimeline(true)
      try {
        const res = await fetch(`${BACKEND_URL}/api/video/timeline/${videoId}`)
        if (res.ok) {
          const data = await res.json()
          setTimeline(data.timeline || [])
        }
      } catch {
        // ignore
      } finally {
        setIsLoadingTimeline(false)
      }
    }

    fetchTimeline()
  }, [videoId, videoStatus?.status])

  // Save messages to localStorage
  useEffect(() => {
    if (videoId && messages.length > 0) {
      localStorage.setItem(`yt_chat_${videoId}`, JSON.stringify(messages))
    }
  }, [messages, videoId])

  // Process video
  const handleProcess = async (force: boolean) => {
    if (!url.trim()) return

    setIsProcessing(true)
    setVideoStatus({ status: 'processing', stage: 'contacting_server' })

    try {
      const res = await fetch(`${BACKEND_URL}/api/video/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url.trim(), force }),
      })

      const data = await res.json()

      if (res.status === 200) {
        setVideoId(data.video_id)
        setVideoStatus({ status: 'ready', chunk_count: data.chunk_count })
        setIsProcessing(false)
      } else if (res.status === 202) {
        setVideoId(data.video_id)
        setVideoStatus({ status: 'processing', stage: 'indexing' })
        pollVideoStatus(data.video_id)
      } else {
        setVideoStatus({ status: 'failed', error: data.detail || data.error || 'Failed to process' })
        setIsProcessing(false)
      }
    } catch (err: any) {
      setVideoStatus({ status: 'failed', error: err.message || 'Network error' })
      setIsProcessing(false)
    }
  }

  // Poll video status while indexing
  const pollVideoStatus = async (vid: string) => {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/api/video/status/${vid}`)
        const data = await res.json()

        if (data.status === 'ready') {
          clearInterval(pollInterval)
          setVideoStatus({ status: 'ready', chunk_count: data.chunk_count })
          setIsProcessing(false)
        } else if (data.status === 'failed') {
          clearInterval(pollInterval)
          setVideoStatus({ status: 'failed', error: data.error })
          setIsProcessing(false)
        }
      } catch {
        // Continue polling
      }
    }, 2000)
  }

  // Send conversational question & stream SSE
  const handleSendMessage = async (promptText: string) => {
    if (!videoId || !promptText.trim()) return

    const userMessage: ChatMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      text: promptText,
      createdAt: Date.now(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsStreaming(true)
    setStreamingTokens('')

    try {
      // Build history for backend
      const historyContext = messages.map((m) => ({
        role: m.role,
        text: m.text,
      }))

      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          prompt: promptText,
          history: historyContext,
        }),
      })

      if (!response.body) {
        throw new Error('ReadableStream not supported by browser.')
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let accumulatedText = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        const chunkStr = decoder.decode(value, { stream: true })
        const lines = chunkStr.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(line.slice(6))
              if (eventData.type === 'token') {
                accumulatedText += eventData.token
                setStreamingTokens(accumulatedText)
              }
            } catch {
              // Ignore partial JSON
            }
          }
        }
      }

      // Finalize assistant message
      const assistantMessage: ChatMessage = {
        id: `asst-${Date.now()}`,
        role: 'model',
        text: accumulatedText,
        createdAt: Date.now(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (err: any) {
      const errorMessage: ChatMessage = {
        id: `err-${Date.now()}`,
        role: 'model',
        text: `Error generating response: ${err.message}`,
        createdAt: Date.now(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsStreaming(false)
      setStreamingTokens('')
    }
  }

  // Generate Summary
  const handleGenerateSummary = async () => {
    if (!videoId) return
    setIsLoadingSummary(true)

    try {
      const res = await fetch(`${BACKEND_URL}/api/video/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId }),
      })
      const data = await res.json()
      if (res.ok && data.summary) {
        setSummary(data.summary)
        localStorage.setItem(`yt_summary_${videoId}`, data.summary)
      }
    } catch {
      // ignore
    } finally {
      setIsLoadingSummary(false)
    }
  }

  // Fetch TTS Audio
  const handleFetchAudio = async () => {
    if (!summary) return
    setIsLoadingAudio(true)

    try {
      const res = await fetch(`${BACKEND_URL}/api/video/tts`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: summary }),
      })
      if (res.ok) {
        const blob = await res.blob()
        setAudioBlob(blob)
      }
    } catch {
      // ignore
    } finally {
      setIsLoadingAudio(false)
    }
  }

  const handleClearHistory = () => {
    if (videoId) {
      localStorage.removeItem(`yt_chat_${videoId}`)
      setMessages([])
    }
  }

  const handleSeek = (seconds: number) => {
    setSeekTimestamp(seconds)
  }

  return (
    <div className="min-h-screen bg-black text-[#ededed] flex flex-col font-sans selection:bg-white/20 selection:text-white">
      {/* Top Navigation */}
      <Navbar
        url={url}
        setUrl={setUrl}
        onProcess={handleProcess}
        isProcessing={isProcessing}
        backendStatus={backendStatus}
        forceReindex={forceReindex}
        setForceReindex={setForceReindex}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 sm:py-6 flex flex-col gap-4 sm:gap-6">
        {/* Error notification if video processing failed */}
        {videoStatus?.status === 'failed' && (
          <div className="flex items-center gap-3 p-3.5 sm:p-4 rounded-2xl bg-zinc-900/90 border border-white/15 text-zinc-200 text-sm shadow-lg">
            <AlertCircle className="w-5 h-5 text-zinc-400 shrink-0" />
            <span className="leading-snug">{videoStatus.error || 'Failed to process video.'}</span>
          </div>
        )}

        {/* Video Processing State Banner */}
        {isProcessing && (
          <div className="flex items-center justify-between p-3.5 sm:p-4 rounded-2xl apple-glass border border-white/20 bg-white/[0.02]">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin shrink-0" />
              <div>
                <p className="text-sm font-medium text-white">Extracting & indexing transcript...</p>
                <p className="text-xs text-zinc-400">Partitioning Pinecone vectors & caching chunks</p>
              </div>
            </div>
          </div>
        )}

        {/* Empty State when no video is loaded yet */}
        {!videoId && !isProcessing && (
          <div className="flex-1 flex flex-col items-center justify-center min-h-[450px] text-center gap-4 p-6 sm:p-8">
            <div className="w-14 h-14 sm:w-16 sm:h-16 rounded-3xl bg-zinc-900 border border-white/10 flex items-center justify-center shadow-2xl">
              <Sparkles className="w-7 h-7 sm:w-8 sm:h-8 text-zinc-300" />
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">Experience Video Intelligence</h2>
            <p className="text-xs sm:text-sm text-zinc-400 max-w-md leading-relaxed">
              Paste any YouTube link to extract transcripts, generate executive voice summaries, and chat with grounded timeline citations.
            </p>
          </div>
        )}

        {/* Loaded Video View: Responsive Split Layout */}
        {videoId && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 lg:gap-8 items-start">
            {/* Left Column: Synchronized YouTube Player (Sticky on Desktop) */}
            <div className="lg:col-span-5 xl:col-span-5 lg:sticky lg:top-24 flex flex-col gap-3.5">
              <YouTubePlayer
                videoId={videoId}
                seekTimestamp={seekTimestamp}
                onTimeUpdate={(time, playing) => {
                  setCurrentTime(time)
                  setIsPlaying(playing)
                }}
              />

              <div className="p-3 sm:p-4 rounded-2xl apple-glass-card text-xs text-zinc-400 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-white shadow-[0_0_8px_rgba(255,255,255,0.8)] animate-pulse' : 'bg-zinc-600'}`} />
                  <span className="font-mono text-zinc-200">{formatSeconds(currentTime)}</span>
                </div>
                <span>ID: <span className="font-mono text-zinc-200">{videoId}</span></span>
                {videoStatus?.chunk_count && (
                  <span className="hidden sm:inline">{videoStatus.chunk_count} beats</span>
                )}
              </div>
            </div>

            {/* Right Column: Apple Segmented Tabs & Content */}
            <div className="lg:col-span-7 xl:col-span-7 flex flex-col gap-4">
              {/* Segmented Pill Tabs */}
              <div className="flex p-1 rounded-full bg-zinc-950 border border-white/[0.08] w-fit">
                <button
                  onClick={() => setActiveTab('timeline')}
                  className={`flex items-center gap-1.5 px-3.5 sm:px-4 py-1.5 rounded-full text-xs font-medium transition-transform duration-100 ${
                    activeTab === 'timeline'
                      ? 'apple-pill-active'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <Clock className="w-3.5 h-3.5" />
                  <span>Storyline Beats</span>
                </button>
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`flex items-center gap-1.5 px-3.5 sm:px-4 py-1.5 rounded-full text-xs font-medium transition-transform duration-100 ${
                    activeTab === 'chat'
                      ? 'apple-pill-active'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <MessageSquare className="w-3.5 h-3.5" />
                  <span>Interactive Q&A</span>
                </button>
                <button
                  onClick={() => setActiveTab('summary')}
                  className={`flex items-center gap-1.5 px-3.5 sm:px-4 py-1.5 rounded-full text-xs font-medium transition-transform duration-100 ${
                    activeTab === 'summary'
                      ? 'apple-pill-active'
                      : 'text-zinc-400 hover:text-white'
                  }`}
                >
                  <FileText className="w-3.5 h-3.5" />
                  <span>Summary & Audio</span>
                </button>
              </div>

              {/* Tab Panes */}
              {activeTab === 'timeline' && (
                <TimelineStream
                  timeline={timeline}
                  currentTime={currentTime}
                  isPlaying={isPlaying}
                  onSeek={handleSeek}
                  isLoading={isLoadingTimeline}
                />
              )}

              {activeTab === 'chat' && (
                <ChatView
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isStreaming={isStreaming}
                  streamingTokens={streamingTokens}
                  onSeek={handleSeek}
                  onClearHistory={handleClearHistory}
                />
              )}

              {activeTab === 'summary' && (
                <SummaryView
                  summary={summary}
                  isLoadingSummary={isLoadingSummary}
                  onGenerateSummary={handleGenerateSummary}
                  audioBlob={audioBlob}
                  isLoadingAudio={isLoadingAudio}
                  onFetchAudio={handleFetchAudio}
                  onSeek={handleSeek}
                />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App
