import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Play } from 'lucide-react'

interface MarkdownRendererProps {
  content: string
  onSeek: (seconds: number) => void
  className?: string
}

function formatTimestampsToLinks(text: string): string {
  // Matches [MM:SS] or [HH:MM:SS], optionally followed by (url)
  const regex = /\[(\d{1,2}:\d{2}(?::\d{2})?)\](?:\([^)]*\))?/g
  return text.replace(regex, (_match, ts) => {
    const parts = ts.split(':').map(Number)
    let sec = 0
    if (parts.length === 2) {
      sec = parts[0] * 60 + parts[1]
    } else if (parts.length === 3) {
      sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    }
    return `[${ts}](#t=${sec})`
  })
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  onSeek,
  className = '',
}) => {
  const processedContent = formatTimestampsToLinks(content)

  return (
    <div className={`text-sm leading-relaxed text-zinc-200 ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            if (href?.startsWith('#t=')) {
              const seconds = parseInt(href.replace('#t=', ''), 10) || 0
              return (
                <button
                  type="button"
                  onClick={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    onSeek(seconds)
                  }}
                  className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-full bg-blue-500/15 hover:bg-blue-500/25 border border-blue-400/30 text-blue-300 text-[11px] font-mono font-medium align-middle active:scale-95 transition-all cursor-pointer select-none"
                  title={`Jump to video at ${children}`}
                >
                  <Play className="w-2.5 h-2.5 fill-blue-300 shrink-0" />
                  <span>{children}</span>
                </button>
              )
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300 underline underline-offset-2"
              >
                {children}
              </a>
            )
          },
          p: ({ children }) => <p className="mb-2.5 last:mb-0 leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 mb-2.5 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2.5 space-y-1">{children}</ol>,
          li: ({ children }) => <li className="text-zinc-200 leading-relaxed">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
          em: ({ children }) => <em className="italic text-zinc-300">{children}</em>,
          code: ({ children }) => (
            <code className="bg-white/[0.08] text-blue-300 font-mono text-xs px-1.5 py-0.5 rounded">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="bg-zinc-900 border border-white/10 p-3 rounded-xl overflow-x-auto my-2 text-xs font-mono text-zinc-200">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-blue-400/50 pl-3 my-2 text-zinc-400 italic">
              {children}
            </blockquote>
          ),
        }}
      >
        {processedContent}
      </ReactMarkdown>
    </div>
  )
}
