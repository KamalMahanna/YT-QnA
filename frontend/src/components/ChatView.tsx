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
    <div className="flex flex-col h-[600px] w-full rounded-2xl apple-glass-card border border-white/[0.08] overflow-hidden">
      {/* Header with Clear Action */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-white/[0.06] bg-white/[0.02]">
        <div className="flex items-center gap-2">
          <Bot className="w-4 h-4 text-blue-400" />
          <span className="text-xs font-semibold text-zinc-200">Video Knowledge Chat</span>
        </div>
        {messages.length > 0 && (
          <button
            onClick={onClearHistory}
            className="flex items-center gap-1 text-[11px] text-zinc-500 hover:text-zinc-300 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
            <span>Clear Chat</span>
          </button>
        )}
      </div>

      {/* Messages Scroll Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && !isStreaming && (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 gap-3">
            <div className="w-12 h-12 rounded-2xl bg-white/[0.04] border border-white/[0.08] flex items-center justify-center text-blue-400">
              <Bot className="w-6 h-6" />
            </div>
            <p className="text-sm font-medium text-zinc-300">Ask anything about this video</p>
            <p className="text-xs text-zinc-500 max-w-xs">
              Every answer is grounded in exact timestamps from the video transcript.
            </p>

            <div className="flex flex-wrap gap-2 justify-center mt-2 max-w-sm">
              {[
                'What are the core conclusions?',
                'Explain the main argument in detail',
                'What was discussed in the beginning?',
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => onSendMessage(suggestion)}
                  className="text-xs px-3 py-1.5 rounded-full bg-white/[0.04] hover:bg-white/[0.08] border border-white/[0.06] text-zinc-300 active:scale-95 transition-all text-left"
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
              className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white shadow-lg shadow-blue-600/10'
                  : 'bg-white/[0.05] border border-white/[0.08] text-zinc-200'
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
            <div className="max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed bg-white/[0.05] border border-white/[0.08] text-zinc-200">
              <MarkdownRenderer content={streamingTokens} onSeek={onSeek} />
              <span className="inline-block w-1.5 h-4 ml-1 bg-blue-400 animate-pulse align-middle" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Bottom Floating Pill Input */}
      <form onSubmit={handleSubmit} className="p-3 bg-white/[0.02] border-t border-white/[0.06]">
        <div className="flex items-center gap-2 bg-zinc-900/80 border border-white/[0.1] rounded-full px-3 py-1.5 shadow-inner">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isStreaming}
            placeholder={isStreaming ? 'Gemini is thinking...' : 'Ask a question about this video...'}
            className="flex-1 bg-transparent text-sm text-zinc-100 placeholder:text-zinc-500 outline-none px-2"
          />

          <button
            type="submit"
            disabled={!input.trim() || isStreaming}
            className={`w-8 h-8 rounded-full flex items-center justify-center transition-all ${
              input.trim() && !isStreaming
                ? 'bg-blue-600 text-white hover:bg-blue-500 active:scale-90 shadow-md shadow-blue-500/20'
                : 'bg-white/[0.05] text-zinc-600 cursor-not-allowed'
            }`}
          >
            <ArrowUp className="w-4 h-4" />
          </button>
        </div>
      </form>
    </div>
  )
}
