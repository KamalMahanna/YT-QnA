import io
import logging
import re
import wave
from typing import Any

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    """Format total seconds into HH:MM:SS or MM:SS string."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def split_text(txt: str, pattern: str) -> list[str]:
    """Split text by pattern, preserving delimiter at the end of each segment."""
    splits = txt.split(pattern)
    lst_ele = splits[-1].strip()
    splits = [i.strip() + pattern for i in splits[:-1] if i.strip()]
    if lst_ele:
        if txt.strip().endswith(pattern[0]):
            splits.append(lst_ele + pattern)
        else:
            splits.append(lst_ele + " ")
    return splits


def splitter(txt: str) -> list[str]:
    """Recursively split text by standard punctuation delimiters."""
    splits = split_text(txt + " ", ". ")
    for delimiter in (", ", "! ", "? "):
        splits = [sub for i in splits if i.strip() for sub in split_text(i, delimiter) if sub.strip()]
    return splits


def create_chunks_with_timestamps(
    transcript_list: list[Any],
    chunk_size: int = 1000,
) -> tuple[list[str], list[float]]:
    """Group transcript entries into text chunks of roughly chunk_size characters with accurate start timestamps.

    Supports both object snippets (with .text/.start) and dictionary representations.
    """
    chunks: list[str] = []
    timestamps: list[float] = []

    current_chunk_texts: list[str] = []
    current_chunk_start: float | None = None
    current_length = 0

    for item in transcript_list:
        text = (getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else "") or "").strip()
        start = getattr(item, "start", None) or (item.get("start") if isinstance(item, dict) else 0.0) or 0.0

        if not text:
            continue

        text_len = len(text)

        # If a single snippet is exceptionally large, flush accumulator and split it
        if text_len > chunk_size:
            if current_chunk_texts:
                chunks.append(" ".join(current_chunk_texts))
                timestamps.append(current_chunk_start if current_chunk_start is not None else 0.0)
                current_chunk_texts = []
                current_chunk_start = None
                current_length = 0

            sub_chunks = chunk_by_sentences(text, max_chars=chunk_size)
            for sc in sub_chunks:
                if sc.strip():
                    chunks.append(sc.strip())
                    timestamps.append(float(start))
            continue

        # If adding this snippet exceeds the target chunk size, finalize current chunk
        if current_chunk_texts and (current_length + text_len + 1 > chunk_size):
            chunks.append(" ".join(current_chunk_texts))
            timestamps.append(current_chunk_start if current_chunk_start is not None else 0.0)
            current_chunk_texts = []
            current_chunk_start = None
            current_length = 0

        # Start a new chunk: record the snippet's exact start timestamp
        if current_chunk_start is None:
            try:
                current_chunk_start = float(start)
            except (ValueError, TypeError):
                current_chunk_start = 0.0

        current_chunk_texts.append(text)
        current_length += text_len + 1

    # Append any remaining accumulated snippet(s)
    if current_chunk_texts:
        chunks.append(" ".join(current_chunk_texts))
        timestamps.append(current_chunk_start if current_chunk_start is not None else 0.0)

    logger.info("Created %d chunks with timestamps spanning %0.1fs to %0.1fs",
                len(chunks), timestamps[0] if timestamps else 0.0, timestamps[-1] if timestamps else 0.0)
    return chunks, timestamps


def chunk_by_sentences(text: str, max_chars: int = 100_000) -> list[str]:
    """Chunk text by sentence boundaries up to max_chars per chunk."""
    try:
        sentences = sent_tokenize(text)
    except Exception:
        # Fallback to regex sentence splitting if nltk tokenizers are missing
        sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def get_video_id(url: str) -> str:
    """Extract the 11-character YouTube video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|\/embed\/|\/shorts\/|youtu\.be\/|\/v\/)([\w-]{11})",
        r"^([\w-]{11})$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url.strip())
        if match:
            return match.group(1)
    return ""


def wave_bytesio(pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> io.BytesIO:
    """Convert raw PCM audio bytes to a WAV in-memory buffer."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)
    buffer.seek(0)
    return buffer
