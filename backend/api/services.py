import asyncio
import logging
import os
from typing import Any

from DataBases.PineconeStore import PineconeStore
from utils.HelperFunctions import create_chunks_with_timestamps, format_timestamp
from utils.Transcript import Transcript

logger = logging.getLogger(__name__)

STATUS_READY = "ready"
STATUS_PROCESSING = "processing"
STATUS_FAILED = "failed"
STATUS_NOT_INDEXED = "not_indexed"


class VideoTaskCoordinator:
    """Coordinates video ingestion, transcript fetching, and Pinecone indexing with in-memory single-flight locking."""

    def __init__(
        self,
        store: PineconeStore | None = None,
        transcript_service: Transcript | None = None,
        groq_api_key: str | None = None,
    ):
        self.store = store
        self.transcript_service = transcript_service or Transcript()
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        # In-memory single-flight mutex per video
        self._locks: dict[str, asyncio.Lock] = {}
        # Status registry for active/completed jobs
        self._statuses: dict[str, dict[str, Any]] = {}
        # In-memory chunk cache for fast summary retrieval
        self._chunks_cache: dict[str, list[str]] = {}
        # In-memory timeline segments cache
        self._timeline_cache: dict[str, list[dict[str, Any]]] = {}

    def _get_lock(self, video_id: str) -> asyncio.Lock:
        if video_id not in self._locks:
            self._locks[video_id] = asyncio.Lock()
        return self._locks[video_id]

    def get_status(self, video_id: str) -> dict[str, Any]:
        """Check status of a video from memory or Pinecone."""
        cached_status = self._statuses.get(video_id)
        if cached_status:
            return {"video_id": video_id, **cached_status}

        if self.store and self.store.video_exists(video_id):
            self._statuses[video_id] = {"status": STATUS_READY}
            return {"video_id": video_id, "status": STATUS_READY, "cached": True}

        return {"video_id": video_id, "status": STATUS_NOT_INDEXED}

    async def process_video(self, video_id: str, force: bool = False) -> dict[str, Any]:
        """Process video transcript and index into Pinecone, safely handling concurrent hits."""
        if not self.store:
            raise RuntimeError("PineconeStore is not configured in VideoTaskCoordinator.")

        # If already indexed and not force, return ready immediately
        if not force and self.store.video_exists(video_id):
            self._statuses[video_id] = {"status": STATUS_READY}
            return {"status": STATUS_READY, "video_id": video_id, "cached": True}

        lock = self._get_lock(video_id)

        # If already locked by another concurrent request, inform caller without reprocessing
        if lock.locked():
            return {
                "status": STATUS_PROCESSING,
                "video_id": video_id,
                "message": "Video is already being processed by another task.",
                "retry_after": 2,
            }

        async with lock:
            # Re-check inside lock
            if not force and self.store.video_exists(video_id):
                self._statuses[video_id] = {"status": STATUS_READY}
                return {"status": STATUS_READY, "video_id": video_id, "cached": True}

            self._statuses[video_id] = {
                "status": STATUS_PROCESSING,
                "stage": "fetching_transcript",
            }
            logger.info("Starting ingestion pipeline for video %s", video_id)

            try:
                # 1. Fetch transcript (run in worker thread to prevent event loop blocking)
                transcript_list: list[dict[str, Any]] = []
                try:
                    transcript_list = await asyncio.to_thread(self.transcript_service.with_youtube_api, video_id)
                    logger.info("Retrieved transcript via YouTube Subtitles API for %s", video_id)
                except Exception as yt_err:
                    logger.warning("YouTube API captions failed for %s: %s; falling back to Whisper", video_id, yt_err)
                    if self.groq_api_key:
                        transcript_list = await asyncio.to_thread(
                            self.transcript_service.with_whisper, self.groq_api_key, video_id
                        )
                        logger.info("Transcribed via Whisper for %s", video_id)
                    else:
                        raise RuntimeError(f"YouTube captions unavailable and GROQ_API_KEY not configured: {yt_err}") from yt_err

                if not transcript_list:
                    raise RuntimeError(f"Could not retrieve transcript for video {video_id}.")

                # 2. Chunking & timestamps
                self._statuses[video_id]["stage"] = "chunking"
                chunks, timestamps = create_chunks_with_timestamps(transcript_list)
                logger.info("Created %d chunks for video %s", len(chunks), video_id)

                # 3. Upsert to Pinecone (run in thread)
                self._statuses[video_id]["stage"] = "embedding_and_indexing"
                if force:
                    await asyncio.to_thread(self.store.delete_video, video_id)
                await asyncio.to_thread(self.store.add_documents, chunks, timestamps, video_id)

                # 4. Cache chunks and timeline for summarization and scrollytelling
                self._chunks_cache[video_id] = chunks
                self._timeline_cache[video_id] = [
                    {
                        "chunk_id": f"{video_id}#chunk_{i}",
                        "start": float(ts),
                        "timestamp": format_timestamp(float(ts)),
                        "text": doc,
                    }
                    for i, (doc, ts) in enumerate(zip(chunks, timestamps))
                ]
                result = {
                    "status": STATUS_READY,
                    "video_id": video_id,
                    "chunk_count": len(chunks),
                }
                self._statuses[video_id] = result
                logger.info("Ingestion completed for video %s", video_id)
                return result

            except Exception as e:
                logger.exception("Failed processing video %s: %s", video_id, e)
                error_result = {
                    "status": STATUS_FAILED,
                    "video_id": video_id,
                    "error": str(e),
                }
                self._statuses[video_id] = error_result
                return error_result

    def get_video_chunks(self, video_id: str) -> list[str]:
        """Retrieve video chunks from memory cache or PineconeStore."""
        if video_id in self._chunks_cache and self._chunks_cache[video_id]:
            return self._chunks_cache[video_id]

        if self.store:
            chunks = self.store.get_video_chunks(video_id)
            if chunks:
                self._chunks_cache[video_id] = chunks
                return chunks

        return []

    def get_video_timeline(self, video_id: str) -> list[dict[str, Any]]:
        """Retrieve chronological timeline beats from memory cache or PineconeStore."""
        if video_id in self._timeline_cache and self._timeline_cache[video_id]:
            return self._timeline_cache[video_id]

        if self.store and hasattr(self.store, "get_video_timeline"):
            timeline = self.store.get_video_timeline(video_id)
            if timeline:
                self._timeline_cache[video_id] = timeline
                return timeline

        return []
