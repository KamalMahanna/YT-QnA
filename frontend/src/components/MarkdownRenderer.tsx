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
    <div className={`text-sm leading-relaxed text-zinc-300 font-normal ${className}`}>
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
                  className="inline-flex items-center gap-1 mx-1 px-2.5 py-0.5 rounded-full bg-white/[0.08] hover:bg-white/[0.18] border border-white/[0.15] text-white text-[11px] font-mono font-medium align-middle active:scale-95 transition-all cursor-pointer select-none shadow-sm shadow-black/40"
                  title={`Jump to video at ${children}`}
                >
                  <Play className="w-2.5 h-2.5 fill-white text-white shrink-0" />
                  <span>{children}</span>
                </button>
              )
            }
            return (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="text-zinc-200 underline decoration-zinc-500 hover:text-white hover:decoration-white transition-colors"
              >
                {children}
              </a>
            )
          },
          p: ({ children }) => <p className="mb-2.5 last:mb-0 leading-relaxed text-zinc-300">{children}</p>,
          ul: ({ children }) => <ul className="list-disc pl-5 mb-2.5 space-y-1 text-zinc-300">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal pl-5 mb-2.5 space-y-1 text-zinc-300">{children}</ol>,
          li: ({ children }) => <li className="text-zinc-300 leading-relaxed">{children}</li>,
          strong: ({ children }) => <strong className="font-semibold text-white tracking-tight">{children}</strong>,
          em: ({ children }) => <em className="italic text-zinc-400">{children}</em>,
          code: ({ children }) => (
            <code className="bg-white/[0.07] text-zinc-200 font-mono text-xs px-1.5 py-0.5 rounded border border-white/[0.06]">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="bg-zinc-950 border border-white/[0.08] p-3.5 rounded-xl overflow-x-auto my-2 text-xs font-mono text-zinc-300">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-zinc-600 pl-3 my-2 text-zinc-400 italic">
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
