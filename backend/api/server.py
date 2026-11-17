import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from api.chat_service import ChatService
from api.services import (
    STATUS_FAILED,
    STATUS_NOT_INDEXED,
    STATUS_PROCESSING,
    STATUS_READY,
    VideoTaskCoordinator,
)
from DataBases.PineconeStore import PineconeStore
from utils.HelperFunctions import get_video_id

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("yt_qna_api")

# Pydantic Schemas
class ProcessVideoRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or 11-char video ID")
    force: bool = Field(False, description="Force re-indexing even if already cached")


class ChatStreamRequest(BaseModel):
    video_id: str
    prompt: str
    history: list[dict[str, Any]] = []


class SummaryRequest(BaseModel):
    video_id: str


class TTSRequest(BaseModel):
    text: str


def create_app(
    store: PineconeStore | None = None,
    coordinator: VideoTaskCoordinator | None = None,
    chat_service: ChatService | None = None,
) -> FastAPI:
    """Application factory for dependency injection and testing."""

    # Initialize components if not provided
    pinecone_key = os.getenv("PINECONE_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if store is None and pinecone_key and gemini_key:
        try:
            store = PineconeStore(pinecone_api_key=pinecone_key, gemini_api_key=gemini_key)
            logger.info("PineconeStore initialized successfully")
        except Exception as e:
            logger.warning("Could not initialize PineconeStore at startup: %s", e)

    if coordinator is None:
        coordinator = VideoTaskCoordinator(store=store)

    if chat_service is None and gemini_key:
        try:
            chat_service = ChatService(gemini_api_key=gemini_key, store=store)
            logger.info("ChatService initialized successfully")
        except Exception as e:
            logger.warning("Could not initialize ChatService at startup: %s", e)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Starting YouTube QnA API server...")
        yield
        logger.info("Shutting down YouTube QnA API server...")

    app = FastAPI(
        title="YouTube QnA API",
        version="1.0.0",
        description="Stateless backend for YouTube QnA with Pinecone vector caching and Gemini streaming.",
        lifespan=lifespan,
    )

    # Enable CORS for local development and Render/Vercel frontends
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach instances to app state
    app.state.store = store
    app.state.coordinator = coordinator
    app.state.chat_service = chat_service

    @app.get("/api/health")
    async def health_check():
        """Health check and cold-start wake endpoint."""
        return {
            "status": "ok",
            "service": "yt-qna-backend",
            "version": "1.0.0",
            "pinecone_ready": app.state.store is not None,
        }

    @app.post("/api/video/process")
    async def process_video(req: ProcessVideoRequest):
        """Process YouTube video transcript and index in Pinecone."""
        video_id = get_video_id(req.url)
        if not video_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid YouTube URL. Could not extract valid 11-character video ID.",
            )

        if not app.state.coordinator.store and app.state.store:
            app.state.coordinator.store = app.state.store

        result = await app.state.coordinator.process_video(video_id, force=req.force)
        res_status = result.get("status")

        if res_status == STATUS_PROCESSING:
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content=result,
            )
        elif res_status == STATUS_READY:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content=result,
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=result,
            )

    @app.get("/api/video/status/{video_id}")
    async def get_video_status(video_id: str):
        """Poll processing status of a video."""
        status_info = app.state.coordinator.get_status(video_id)
        return status_info

    @app.get("/api/video/timeline/{video_id}")
    async def get_video_timeline(video_id: str):
        """Retrieve chronological transcript timeline beats for scroll-driven synchronization."""
        timeline = app.state.coordinator.get_video_timeline(video_id)
        return {"video_id": video_id, "timeline": timeline}

    @app.post("/api/chat/stream")
    async def chat_stream(req: ChatStreamRequest):
        """Stream conversational RAG response via Server-Sent Events (SSE)."""
        if not app.state.chat_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChatService is not initialized. Ensure GEMINI_API_KEY is configured.",
            )

        # Make sure chat_service has the active store
        if not app.state.chat_service.store and app.state.store:
            app.state.chat_service.store = app.state.store

        stream = app.state.chat_service.stream_rag_answer(
            query=req.prompt,
            video_id=req.video_id,
            history=req.history,
        )

        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/video/summary")
    async def get_video_summary(req: SummaryRequest):
        """Generate or retrieve video summary."""
        if not app.state.chat_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChatService is not initialized.",
            )

        # Fetch chunks from coordinator cache or store
        chunks = app.state.coordinator.get_video_chunks(req.video_id)
        if not chunks and app.state.store:
            chunks = app.state.store.get_video_chunks(req.video_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No transcript chunks found for video {req.video_id}. Please process the video first.",
            )

        summary_text = await app.state.chat_service.generate_summary(chunks)
        return {"video_id": req.video_id, "summary": summary_text}

    @app.post("/api/video/tts")
    async def get_video_tts(req: TTSRequest):
        """Convert summary or text to WAV audio."""
        if not app.state.chat_service:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ChatService is not initialized.",
            )

        if not req.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty.",
            )

        try:
            wav_bytes = await app.state.chat_service.generate_tts(req.text)
            return Response(content=wav_bytes, media_type="audio/wav")
        except Exception as e:
            logger.exception("TTS audio synthesis failed: %s", e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"TTS synthesis failed: {e}",
            )

    return app


# Default app instance for uvicorn
app = create_app()
