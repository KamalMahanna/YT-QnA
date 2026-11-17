import React, { useState, useRef, useEffect } from 'react'
import { ArrowUp, Trash2, Bot } from 'lucide-react'
import type { ChatMessage } from '../types'
import { MarkdownRenderer } from './MarkdownRenderer'

interface ChatViewProps {
  messages: ChatMessage[]
  onSendMessage: (text: string) => void
  isStreaming: boolean
  streamingTokens: string
  onSeek: (seconds: number) => void
  onClearHistory: () => void
}

export const ChatView: React.FC<ChatViewProps> = ({
  messages,
  onSendMessage,
  isStreaming,
  streamingTokens,
  onSeek,
  onClearHistory,
}) => {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingTokens])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim() && !isStreaming) {
      onSendMessage(input.trim())
      setInput('')
    }
  }

  return (
    <div className="flex flex-col h-[500px] sm:h-[560px] lg:h-[620px] w-full rounded-2xl apple-glass-card border border-white/[0.08] overflow-hidden shadow-2xl shadow-black/80">
      {/* Header with Clear Action */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06] bg-white/[0.015]">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-zinc-400" />
          <span className="text-xs font-semibold text-zinc-200 tracking-tight">Video Knowledge Chat</span>
        </div>
        {messages.length > 0 && (
          <button
            onClick={onClearHistory}
            className="flex items-center gap-1.5 text-[11px] text-zinc-500 hover:text-zinc-300 active:scale-95 transition-all p-1"
            title="Clear conversation"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Clear Chat</span>
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-3.5 sm:p-5 space-y-4">
        {messages.length === 0 && !isStreaming && (
          <div className="h-full flex flex-col items-center justify-center text-center p-4 sm:p-6 gap-3">
            <div className="w-11 h-11 sm:w-12 sm:h-12 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-zinc-400 shadow-inner">
              <Bot className="w-5 h-5 sm:w-6 sm:h-6" />
            </div>
            <p className="text-sm font-medium text-zinc-200">Ask anything about this video</p>
            <p className="text-xs text-zinc-500 max-w-xs leading-relaxed">
              Every answer is verified against exact timestamps from the video transcript.
            </p>

            <div className="flex flex-wrap gap-2 justify-center mt-3 max-w-md">
              {[
                'What are the core conclusions?',
                'Explain the main argument in detail',
                'What was discussed in the beginning?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => onSendMessage(suggestion)}
                  className="text-xs px-3 py-1.5 rounded-full bg-white/[0.03] hover:bg-white/[0.08] border border-white/[0.06] text-zinc-300 active:scale-95 transition-transform duration-100 text-left"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}
          >
            <div
              className={`max-w-[88%] sm:max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-white text-black font-medium shadow-md shadow-white/10'
                  : 'bg-white/[0.035] border border-white/[0.08] text-zinc-200 shadow-sm'
              }`}
            >
              {msg.role === 'user' ? (
                <div className="whitespace-pre-wrap">{msg.text}</div>
              ) : (
                <MarkdownRenderer content={msg.text} onSeek={onSeek} />
              )}
            </div>
          </div>
        ))}

        {/* Live Streaming Response Bubble */}
        {isStreaming && (
          <div className="flex flex-col items-start">
            <div className="max-w-[88%] sm:max-w-[82%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-white/[0.035] border border-white/[0.08] text-zinc-200 shadow-sm">
              {streamingTokens ? (
                <>
                  <MarkdownRenderer content={streamingTokens} onSeek={onSeek} />
                  <span className="inline-block w-1.5 h-3.5 ml-1 bg-white/70 animate-pulse align-middle" />
                </>
              ) : (
                <div className="flex items-center gap-1.5 py-1 px-1">
                  <span className="w-2 h-2 rounded-full bg-white/60 animate-bounce [animation-delay:-0.3s]" />
                  <span className="w-2 h-2 rounded-full bg-white/60 animate-bounce [animation-delay:-0.15s]" />
                  <span className="w-2 h-2 rounded-full bg-white/60 animate-bounce" />
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Bottom Floating Pill Input */}
      <form onSubmit={handleSubmit} className="p-2.5 sm:p-3 bg-white/[0.015] border-t border-white/[0.06]">
        <div className="flex items-center gap-2 bg-zinc-900/90 border border-white/[0.1] rounded-full px-3 py-1.5 focus-within:border-white/30 focus-within:ring-1 focus-within:ring-white/20 transition-all duration-150">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
            placeholder={isStreaming ? 'Gemini is thinking...' : 'Ask a question about this video...'}
            className="flex-1 bg-transparent text-base sm:text-sm text-zinc-100 placeholder:text-zinc-500 outline-none px-1"
          />

          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-transform duration-100 shrink-0 ${
              input.trim() && !isStreaming
                ? 'bg-white text-black hover:bg-zinc-200 active:scale-90 shadow-md shadow-white/20'
                : 'bg-white/[0.04] text-zinc-600 cursor-not-allowed'
            }`}
            title="Send query"
          >
            <ArrowUp className="w-4 h-4 stroke-[2.5]" />
          </button>
        </div>
      </form>
    </div>
  )
}
