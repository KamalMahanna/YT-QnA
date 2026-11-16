# YouTube QnA - System Architecture

This document outlines the architecture, data flow, and components of the **YouTube QnA** application.

---

## 1. High-Level Architecture Overview

```mermaid
flowchart TD
    User(["User (Browser)"]) <--> UI["Streamlit Web App (main.py)"]

    subgraph Ingestion ["Ingestion & Transcription Pipeline"]
        direction TB
        URLParser["URL Parser (get_video_id)"]
        YTAPI["YouTube Transcript API"]
        AudioDL["Audio Downloader (pytubefix + pydub)"]
        GroqWhisper["Groq STT (whisper-large-v3-turbo)"]
        Chunker["Chunking & Timestamp Alignment"]
    end

    subgraph Storage ["Vector Storage (ChromaDB)"]
        direction TB
        Chroma[("ChromaDB Persistent Store")]
        GeminiEmbed["Gemini Embeddings (gemini-embedding-2)"]
    end

    subgraph Intelligence ["LLM & Audio Services (Google Gemini)"]
        direction TB
        SummarizerEngine["Hierarchical Summarizer"]
        RAGGeneration["Chat Generation (gemini-2.5-flash)"]
        TTSService["TTS Audio Synthesis (gemini-3.8-flash-tts)"]
    end

    UI --> URLParser
    URLParser --> YTAPI
    YTAPI -- "Fallback if captions unavailable" --> AudioDL
    AudioDL --> GroqWhisper
    YTAPI --> Chunker
    GroqWhisper --> Chunker
    Chunker --> GeminiEmbed
    GeminiEmbed --> Chroma

    Chroma --> SummarizerEngine
    SummarizerEngine --> TTSService
    TTSService --> UI

    UI -- "User Query" --> Chroma
    Chroma -- "Top K Relevant Chunks" --> RAGGeneration
    RAGGeneration -- "Answer + Timestamp Links" --> UI
```

---

## 2. Ingestion & Transcription Pipeline

When a user submits a YouTube URL, the application extracts the transcript through a fault-tolerant multi-tier pipeline:

```mermaid
flowchart TD
    Start(["Input YouTube URL"]) --> ParseID["Extract Video ID"]
    ParseID --> CheckDB{"Already indexed in ChromaDB?"}

    CheckDB -- "Yes" --> Ready(["Ready for Q&A / Summary"])
    CheckDB -- "No" --> TryYTAPI["Try youtube-transcript-api"]

    TryYTAPI -- "Captions available" --> GetYTTranscript["Fetch English Subtitles"]
    TryYTAPI -- "Failed / Disabled" --> DL["Download Audio Stream (pytubefix)"]

    DL --> CheckSize{"Audio size > 15MB?"}
    CheckSize -- "Yes" --> SplitAudio["Split Audio with pydub / ffmpeg"]
    CheckSize -- "No" --> Transcribe["Send to Groq API"]
    SplitAudio --> Transcribe

    Transcribe --> WhisperSTT["Transcribe with whisper-large-v3-turbo"]

    GetYTTranscript --> TextChunker["Chunk Text (~1000 chars) & Map Timestamps"]
    WhisperSTT --> TextChunker

    TextChunker --> BatchEmbed["Batch Embed with gemini-embedding-2"]
    BatchEmbed --> SaveChroma[("Store in ChromaDB with metadata")]
    SaveChroma --> Ready
```

---

## 3. RAG Q&A Interaction Flow

The conversational question-answering workflow uses Retrieval-Augmented Generation (RAG) to ensure grounded answers with hyperlinked timestamp citations:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit Frontend
    participant VectorStore as ChromaDB Vector Store
    participant Gemini as Gemini API (Text & Embedding)

    User->>Streamlit: Submit question (prompt)
    Streamlit->>VectorStore: retrieve_documents(prompt, video_id, n_results=5)
    VectorStore->>Gemini: embed_query(prompt) via gemini-embedding-2
    Gemini-->>VectorStore: 3072-dim Query Vector
    VectorStore-->>Streamlit: Return top 5 chunks + start timestamps

    Note over Streamlit: Construct System Prompt with Context & Citation Rules
    Streamlit->>Gemini: TextLLM(system_instruction, chat_history, prompt)
    Gemini-->>Streamlit: Grounded answer with markdown links [start](https://youtube.com/watch?v=...&t=...)
    Streamlit-->>User: Display response in chat UI
```

---

## 4. Summarization & Voice Synthesis Flow

```mermaid
flowchart LR
    subgraph Summarization ["Hierarchical Summarizer"]
        direction TB
        RawChunks["All Video Chunks"] --> LengthCheck{"Total Characters > 100,000?"}
        LengthCheck -- "Yes" --> SplitSentences["Sentence-level Chunks"]
        SplitSentences --> MapSummaries["Individual Chunk Summaries (gemini-2.5-flash)"]
        MapSummaries --> ReduceSummary["Merge Summary (summary_of_summaries)"]
        LengthCheck -- "No" --> DirectSummary["Single-pass Chunk Summary"]
    end

    subgraph TTS ["Audio Synthesis"]
        direction TB
        FinalSummary["Final Text Summary"] --> GeminiTTS["Gemini TTS (gemini-3.8-flash-tts / Voice: Zephyr)"]
        GeminiTTS --> WaveConvert["PCM to WAV (wave_bytesio)"]
        WaveConvert --> AudioPlayer["Autoplay Audio in Streamlit"]
    end

    ReduceSummary --> FinalSummary
    DirectSummary --> FinalSummary
```

---

## 5. Component & Technology Matrix

| Subsystem | Components / Files | Technologies / Models | Role |
| :--- | :--- | :--- | :--- |
| **Frontend** | [main.py](file:///home/zoro/Desktop/YT-QnA/main.py) | Streamlit | Chat interface, status indicators, audio playback, sidebar configuration |
| **Transcription** | [Transcript.py](file:///home/zoro/Desktop/YT-QnA/utils/Transcript.py), [AudioDownloader.py](file:///home/zoro/Desktop/YT-QnA/utils/AudioDownloader.py) | `youtube-transcript-api`, `pytubefix`, `pydub`, `ffmpeg`, Groq (`whisper-large-v3-turbo`) | Extracts captions directly or downloads audio and transcribes via Whisper fallback |
| **Preprocessing** | [HelperFunctions.py](file:///home/zoro/Desktop/YT-QnA/utils/HelperFunctions.py) | `nltk`, `re`, `wave` | Sentence tokenization, character-bounded chunking, timestamp alignment, audio byte stream formatting |
| **Vector Database** | [VectorStore.py](file:///home/zoro/Desktop/YT-QnA/DataBases/VectorStore.py) | ChromaDB | Local persistent vector storage, metadata filtering (`youtube_id`), semantic retrieval |
| **Embeddings** | [LLM.py](file:///home/zoro/Desktop/YT-QnA/utils/LLM.py) (`GeminiLLM`) | Google Gemini (`gemini-embedding-2`) | 3072-dimensional document and query embeddings using multimodal `types.Content` architecture |
| **Summarizer** | [Summarizer.py](file:///home/zoro/Desktop/YT-QnA/utils/Summarizer.py) | Google Gemini (`gemini-2.5-flash`) | Map-reduce hierarchical summarization preserving factual content |
| **Text-to-Speech** | [LLM.py](file:///home/zoro/Desktop/YT-QnA/utils/LLM.py) (`TTS`) | Google Gemini (`gemini-3.8-flash-tts`) | Converts summaries into natural voice audio with the "Zephyr" prebuilt voice |
| **Q&A LLM** | [LLM.py](file:///home/zoro/Desktop/YT-QnA/utils/LLM.py) (`TextLLM`) | Google Gemini (`gemini-2.5-flash`) | Context-grounded multi-turn conversational chat with timestamp citation links |
