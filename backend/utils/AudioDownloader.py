import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from pydub import AudioSegment
from pytubefix import YouTube

logger = logging.getLogger(__name__)

# Groq Whisper free tier file limit is 25 MB; chunk at 15 MB for safety margin
MAX_AUDIO_CHUNK_BYTES = 15 * 1024 * 1024


class AudioDownloader:
    """Downloads and segments YouTube audio streams for speech transcription."""

    def __init__(self, max_chunk_bytes: int = MAX_AUDIO_CHUNK_BYTES):
        self.max_chunk_bytes = max_chunk_bytes

    def download_audio(self, video_id: str) -> list[str]:
        """Download audio for a YouTube video ID and split into chunks if exceeding size limit.

        Returns a list of local file paths to the audio file(s).
        """
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info("Initializing audio stream download for video ID: %s", video_id)

        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True, abr="128kbps").first()
        if not audio_stream:
            logger.warning("128kbps audio stream not found for %s, falling back to first available audio stream", video_id)
            audio_stream = yt.streams.filter(only_audio=True).first()

        if not audio_stream:
            raise RuntimeError(f"No audio stream found for YouTube video {video_id}")

        with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_audio:
            temp_path = Path(temp_audio.name)

        logger.debug("Downloading audio stream to %s", temp_path)
        audio_stream.download(output_path=str(temp_path.parent), filename=temp_path.name)
        file_size = temp_path.stat().st_size
        logger.info("Downloaded audio file size: %0.2f MB", file_size / (1024 * 1024))

        if file_size > self.max_chunk_bytes:
            logger.info("Audio size exceeds %0.2f MB threshold; splitting into segments", self.max_chunk_bytes / (1024 * 1024))
            audio = AudioSegment.from_file(str(temp_path))
            num_chunks = int(np.ceil(file_size / self.max_chunk_bytes))
            split_points = np.linspace(0, len(audio), num_chunks + 1, dtype=int)

            chunk_paths: list[str] = []
            for start_ms, end_ms in zip(split_points[:-1], split_points[1:]):
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as chunk_file:
                    chunk_path = chunk_file.name

                audio[start_ms:end_ms].export(chunk_path, format="mp3")
                chunk_paths.append(chunk_path)

            # Clean up the original large audio file
            if temp_path.exists():
                temp_path.unlink()

            logger.info("Split audio into %d segments", len(chunk_paths))
            return chunk_paths

        return [str(temp_path)]
