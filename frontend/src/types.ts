export interface SourceCitation {
  source_idx: number
  start: number
  timestamp: string
  url: string
  snippet: string
}

export interface ChatMessage {
  id: string
  role: 'user' | 'model'
  text: string
  sources?: SourceCitation[]
  createdAt: number
}

export interface VideoStatus {
  status: 'not_indexed' | 'processing' | 'ready' | 'failed'
  stage?: string
  error?: string
  chunk_count?: number
  cached?: boolean
}
