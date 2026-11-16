import logging
import os
from typing import Any, Union

from google import genai
from google.genai import types
from groq import Groq

logger = logging.getLogger(__name__)


class GroqLLM:
    """Wrapper for Groq Audio transcription services."""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def AudioLLM(
        self,
        audio_paths: list[str],
        model_name: str = "whisper-large-v3-turbo",
    ) -> list[dict[str, Any]]:
        """Transcribe a list of audio files and merge segment timestamps."""
        all_segments: list[dict[str, Any]] = []
        cumulative_start_time = 0.0

        for audio_path in audio_paths:
            logger.info("Transcribing audio file: %s using model %s", audio_path, model_name)
            try:
                with open(audio_path, "rb") as file_handle:
                    transcription = self.client.audio.transcriptions.create(
                        file=(os.path.basename(audio_path), file_handle.read()),
                        language="en",
                        model=model_name,
                        response_format="verbose_json",
                    )

                segments = transcription.segments or []
                for seg in segments:
                    text = seg.get("text", "").strip() if isinstance(seg, dict) else seg.text.strip()
                    start = seg.get("start", 0.0) if isinstance(seg, dict) else seg.start
                    if text:
                        all_segments.append({
                            "text": text,
                            "start": float(start) + cumulative_start_time,
                        })

                if segments:
                    last_seg = segments[-1]
                    last_end = last_seg.get("end", 0.0) if isinstance(last_seg, dict) else getattr(last_seg, "end", 0.0)
                    cumulative_start_time += float(last_end)
            except Exception as e:
                logger.exception("Error transcribing %s: %s", audio_path, e)
                raise
            finally:
                if os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except OSError as err:
                        logger.warning("Failed to remove temporary audio file %s: %s", audio_path, err)

        logger.info("Transcription completed with %d total segments", len(all_segments))
        return all_segments

    # Alias for pythonic naming
    transcribe_audio = AudioLLM


class GeminiLLM:
    """Wrapper for Google Gemini models: Text Generation, TTS, and Multimodal Embeddings."""

    def __init__(self, api_key: str):
        # Use v1beta endpoint to support system_instruction, multimodal gemini-embedding-2, and TTS speech_config
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1beta"})

    def TextLLM(
        self,
        system_instruction: str,
        history: list[dict[str, Any]],
        query: str,
        model_name: str = "gemini-2.5-flash",
    ) -> str:
        """Generate conversational text response with given instructions and history."""
        logger.debug("Generating response using %s (query length: %d chars)", model_name, len(query))
        chat = self.client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
            history=history,
        )
        response = chat.send_message(query)
        return response.text or ""

    # Alias for pythonic naming
    generate_text = TextLLM

    def TTS(self, texts: str, voice_name: str = "Zephyr") -> bytes:
        """Synthesize text into raw PCM audio bytes using Gemini TTS."""
        logger.info("Synthesizing TTS audio (input length: %d chars, voice: %s)", len(texts), voice_name)
        response = self.client.models.generate_content(
            model="gemini-3.8-flash-tts",
            contents=f"Read aloud in an energetic and friendly tone: {texts}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )
        return response.candidates[0].content.parts[0].inline_data.data

    def __call__(self, input: Union[str, list[str]]) -> list[list[float]]:
        """Embed document texts for ChromaDB indexing using gemini-embedding-2."""
        items = [input] if isinstance(input, str) else list(input)
        if not items:
            return []

        logger.debug("Embedding %d documents with gemini-embedding-2", len(items))
        embeddings: list[list[float]] = []
        batch_size = 50

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in batch
            ]
            response = self.client.models.embed_content(
                model="gemini-embedding-2",
                contents=contents,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            embeddings.extend([e.values for e in response.embeddings])

        return embeddings

    def embed_query(self, input: Union[str, list[str]]) -> list[list[float]]:
        """Embed search query texts using gemini-embedding-2."""
        items = [input] if isinstance(input, str) else list(input)
        if not items:
            return []

        logger.debug("Embedding %d query text(s) with gemini-embedding-2", len(items))
        embeddings: list[list[float]] = []
        batch_size = 50

        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            contents = [
                types.Content(parts=[types.Part.from_text(text=text)])
                for text in batch
            ]
            response = self.client.models.embed_content(
                model="gemini-embedding-2",
                contents=contents,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            embeddings.extend([e.values for e in response.embeddings])

        return embeddings

    def name(self) -> str:
        return "gemini_embedding"
