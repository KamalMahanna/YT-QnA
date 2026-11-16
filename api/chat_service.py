import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator

from google import genai
from google.genai import types

from DataBases.PineconeStore import PineconeStore
from utils.HelperFunctions import format_timestamp, wave_bytesio
from utils.LLM import GeminiLLM
from utils.Summarizer import Summarizer

logger = logging.getLogger(__name__)


class ChatService:
    """Manages conversational RAG generation with citations, streaming SSE, summarization, and TTS audio."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        store: PineconeStore | None = None,
    ):
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for ChatService.")

        self.store = store
        self.gemini_client = genai.Client(
            api_key=self.gemini_api_key,
            http_options={"api_version": "v1beta"},
        )
        self.llm = GeminiLLM(self.gemini_api_key)
        self.summarizer = Summarizer(self.gemini_api_key)

    def build_rag_context(
        self,
        chunks: list[dict[str, Any]],
        video_id: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Format retrieved context chunks and structured source citations."""
        formatted_context_list = []
        sources = []

        for idx, chunk in enumerate(chunks, start=1):
            start_sec = int(chunk.get("start", 0))
            ts = format_timestamp(start_sec)
            url = f"https://www.youtube.com/watch?v={video_id}&t={start_sec}s"
            text_snippet = chunk.get("text", "").strip()

            formatted_context_list.append(
                f"Source {idx} Citation: [{ts}]({url})\nContent: {text_snippet}"
            )
            sources.append({
                "source_idx": idx,
                "start": start_sec,
                "timestamp": ts,
                "url": url,
                "snippet": text_snippet[:150] + "..." if len(text_snippet) > 150 else text_snippet,
            })

        context_str = "\n\n".join(formatted_context_list)
        return context_str, sources

    async def stream_rag_answer(
        self,
        query: str,
        video_id: str,
        history: list[dict[str, Any]] | None = None,
        n_results: int = 5,
    ) -> AsyncGenerator[str, None]:
        """Stream conversational RAG response formatted as Server-Sent Events (SSE)."""
        if not self.store:
            raise RuntimeError("PineconeStore is not configured in ChatService.")

        history = history or []

        # 1. Retrieve top matching chunks from Pinecone
        chunks = await asyncio.to_thread(
            self.store.retrieve_documents, query, video_id, n_results
        )
        context_str, sources = self.build_rag_context(chunks, video_id)

        # 2. Emit sources first so the frontend can immediately render citations
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

        system_instruction = f"""Instructions:
- Be helpful and answer questions concisely based on the context provided.
- Ground all facts in the context. Every time you mention a fact or information from a source, cite it with the exact markdown link provided in the Source Citation header (e.g. [MM:SS](URL)).
- Never fabricate timestamps. Use strictly the citation links provided in the Context headers.
- If the context does not contain the answer, say 'I don't know based on the video.'

Context:
{context_str}
"""

        # 3. Build contents list with previous conversation history
        contents: list[types.Content] = []
        for msg in history:
            role = "user" if msg.get("role") in ["user", "human"] else "model"
            text_val = msg.get("text", "")
            if not text_val and "parts" in msg:
                parts = msg.get("parts") or []
                text_val = parts[0].get("text", "") if parts else ""
            if text_val:
                contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text_val)]))

        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=query)]))

        # 4. Stream response from Gemini
        try:
            stream = await asyncio.to_thread(
                self.gemini_client.models.generate_content_stream,
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                ),
            )

            for response_chunk in stream:
                token = response_chunk.text or ""
                if token:
                    yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as err:
            logger.exception("Error during Gemini stream: %s", err)
            yield f"data: {json.dumps({'type': 'error', 'error': str(err)})}\n\n"

    async def generate_summary(self, transcript_chunks: list[str]) -> str:
        """Generate hierarchical summary of transcript chunks."""
        if not transcript_chunks:
            return "No transcript content available for summarization."
        return await asyncio.to_thread(self.summarizer.summarize_transcript, transcript_chunks)

    async def generate_tts(self, text: str) -> bytes:
        """Synthesize text to WAV audio bytes."""
        pcm_bytes = await asyncio.to_thread(self.llm.TTS, text)
        wav_buffer = wave_bytesio(pcm_bytes)
        return wav_buffer.getvalue()
