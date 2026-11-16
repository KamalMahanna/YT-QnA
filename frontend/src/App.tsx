import React, { useState, useEffect } from 'react'
import { Navbar } from './components/Navbar'
import { YouTubePlayer } from './components/YouTubePlayer'
import { ChatView } from './components/ChatView'
import { SummaryView } from './components/SummaryView'
import type { ChatMessage, VideoStatus } from './types'
import { MessageSquare, FileText, AlertCircle, Sparkles } from 'lucide-react'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || ''

export const App: React.FC = () => {
  const [url, setUrl] = useState('')
  const [videoId, setVideoId] = useState<string | null>(null)
  const [forceReindex, setForceReindex] = useState(false)
  const [activeTab, setActiveTab] = useState<'chat' | 'summary'>('chat')

  // Backend connection & health
  const [backendStatus, setBackendStatus] = useState<'connected' | 'waking' | 'error'>('waking')
  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)

  // YouTube seeking
  const [seekTimestamp, setSeekTimestamp] = useState<number | null>(null)

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
    <div className="min-h-screen bg-black text-[#f5f5f7] flex flex-col font-sans selection:bg-blue-500/30">
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
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 flex flex-col gap-6">
        {/* Error notification if video processing failed */}
        {videoStatus?.status === 'failed' && (
          <div className="flex items-center gap-2 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm">
            <AlertCircle className="w-5 h-5 shrink-0" />
            <span>{videoStatus.error || 'Failed to process video.'}</span>
          </div>
        )}

        {/* Video Processing State Banner */}
        {isProcessing && (
          <div className="flex items-center justify-between p-4 rounded-2xl apple-glass border border-blue-500/30 bg-blue-500/5">
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
              <div>
                <p className="text-sm font-medium text-white">Extracting & indexing transcript...</p>
                <p className="text-xs text-zinc-400">Partitioning Pinecone vectors & caching chunks</p>
              </div>
            </div>
          </div>
        )}

        {/* Empty State when no video is loaded yet */}
        {!videoId && !isProcessing && (
          <div className="flex-1 flex flex-col items-center justify-center min-h-[500px] text-center gap-4 p-8">
            <div className="w-16 h-16 rounded-3xl bg-gradient-to-tr from-zinc-800 to-zinc-900 border border-white/10 flex items-center justify-center shadow-2xl">
              <Sparkles className="w-8 h-8 text-white/80" />
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-white">Experience Video Intelligence</h2>
            <p className="text-sm text-zinc-400 max-w-md">
              Paste any YouTube video link to extract transcripts, generate executive voice summaries, and chat with grounded timestamp citations.
            </p>
          </div>
        )}

        {/* Loaded Video View: Split Layout */}
        {videoId && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left Column: Synchronized YouTube Player */}
            <div className="lg:col-span-6 flex flex-col gap-4">
              <YouTubePlayer videoId={videoId} seekTimestamp={seekTimestamp} />

              <div className="p-4 rounded-2xl apple-glass-card text-xs text-zinc-400 flex items-center justify-between">
                <span>Video ID: <span className="font-mono text-zinc-200">{videoId}</span></span>
                {videoStatus?.chunk_count && (
                  <span>{videoStatus.chunk_count} searchable chunks indexed</span>
                )}
              </div>
            </div>

            {/* Right Column: Apple Segmented Tabs & Content */}
            <div className="lg:col-span-6 flex flex-col gap-4">
              {/* Segmented Pill Tabs */}
              <div className="flex p-1 rounded-full bg-zinc-900 border border-white/[0.08] w-fit">
                <button
                  onClick={() => setActiveTab('chat')}
                  className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-medium transition-all ${
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
                  className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-medium transition-all ${
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
