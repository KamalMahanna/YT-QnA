import io
import logging
import re
import wave
from typing import Any

from nltk.tokenize import sent_tokenize

logger = logging.getLogger(__name__)


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
    """Group transcript entries into text chunks of roughly chunk_size characters with start timestamps.

    Supports both object snippets (with .text/.start) and dictionary representations.
    """
    chunks: list[str] = []
    timestamps: list[float] = []

    chunk_stack = ""
    timestamp_stack = 0.0

    for i in transcript_list:
        the_text = (getattr(i, "text", None) or (i.get("text") if isinstance(i, dict) else "") or "").strip()
        the_timestamp = getattr(i, "start", None) or (i.get("start") if isinstance(i, dict) else 0.0) or 0.0

        if not the_text:
            continue

        if len(the_text) >= chunk_size:
            if chunk_stack:
                chunks.append(chunk_stack)
                timestamps.append(timestamp_stack)
                chunk_stack = ""
                timestamp_stack = 0.0

            chunks.append(the_text)
            timestamps.append(the_timestamp)
        else:
            if chunk_stack:
                chunk_stack += " " + the_text
            else:
                chunk_stack = the_text
                timestamp_stack = the_timestamp

            if len(chunk_stack) > chunk_size:
                splits = splitter(chunk_stack)
                if len(splits) == 1:
                    splits = [f"{word} " for word in splits[0].split() if word.strip()]

                temp_chunk_stack = ""
                while len("".join(splits)) > chunk_size and splits:
                    temp_chunk_stack = splits.pop() + temp_chunk_stack

                resolved_chunk = "".join(splits).strip()
                if resolved_chunk:
                    chunks.append(resolved_chunk)
                    timestamps.append(timestamp_stack)

                chunk_stack = temp_chunk_stack.strip()

    # Append any remaining chunk stack so trailing transcript content is preserved
    if chunk_stack.strip():
        chunks.append(chunk_stack.strip())
        timestamps.append(timestamp_stack)

    logger.debug("Created %d chunks from %d transcript items", len(chunks), len(transcript_list))
    return chunks, timestamps


def chunk_by_sentences(text: str, max_chars: int = 100_000) -> list[str]:
    """Chunk text by sentence boundaries up to max_chars per chunk."""
    sentences = sent_tokenize(text)
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
