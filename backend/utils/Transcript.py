import logging
from typing import Any

from youtube_transcript_api import YouTubeTranscriptApi

from .AudioDownloader import AudioDownloader
from .LLM import GroqLLM

logger = logging.getLogger(__name__)


class Transcript:
    """Handles fetching YouTube transcripts via YouTube Transcript API with Whisper STT fallback."""

    def with_youtube_api(self, video_id: str) -> list[dict[str, Any]]:
        """Fetch transcript directly from YouTube using the official subtitles API."""
        logger.info("Attempting transcript retrieval via YouTube Transcript API for %s", video_id)
        transcript_api = YouTubeTranscriptApi()
        fetched = transcript_api.fetch(video_id, languages=["en"])
        results = [{"text": snippet.text, "start": float(snippet.start)} for snippet in fetched]
        logger.info("Successfully fetched %d transcript segments via YouTube API", len(results))
        return results

    def with_whisper(
        self,
        api_key: str,
        video_id: str,
        model_name: str = "whisper-large-v3-turbo",
    ) -> list[dict[str, Any]]:
        """Download audio and transcribe using Groq-hosted Whisper model."""
        logger.info("Falling back to Whisper STT for video %s (model: %s)", video_id, model_name)
        downloader = AudioDownloader()
        audio_paths = downloader.download_audio(video_id)
        groq_llm = GroqLLM(api_key)
        return groq_llm.AudioLLM(audio_paths, model_name=model_name)
