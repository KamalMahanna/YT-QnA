import logging

from .HelperFunctions import chunk_by_sentences
from .LLM import GeminiLLM

logger = logging.getLogger(__name__)

CHUNK_SYSTEM_PROMPT_TEMPLATE = (
    "You are a precise and faithful summarizer. "
    "Below is a transcript chunk from a YouTube video. "
    "Your task is to summarize the content of this chunk only. "
    "Do not add any information that is not explicitly present in the chunk. "
    "Do not attempt to predict or infer anything beyond the given text. "
    "Stay within the scope of this chunk. "
    "Focus on preserving the speaker's intent, and ensure the summary is clear and concise while reflecting the actual content. "
    "If the chunk includes multiple topics, reflect that in your summary as well. "
    "Please do not include ads, marketing, or promotions in the summary.\n\n"
    "Transcript Chunk:\n---\n{text}\n---"
)

MERGE_SYSTEM_PROMPT_TEMPLATE = (
    "You are a summarizer responsible for combining multiple independent summaries into a final coherent summary. "
    "Each summary corresponds to a chunk of a longer transcript. "
    "Your task is to merge these summaries into one comprehensive, faithful summary of the full content, "
    "without introducing any new assumptions or interpretations. Do not hallucinate or infer beyond what the summaries provide. "
    "Ensure that the final output is cohesive and preserves the structure, themes, and key information reflected across the chunk summaries.\n\n"
    "Chunk Summaries:\n---\n{summaries}\n---"
)


class Summarizer:
    """Performs hierarchical map-reduce summarization on video transcripts."""

    def __init__(self, api_key: str):
        self.llm = GeminiLLM(api_key)

    def summarize_chunk(self, text: str) -> str:
        """Summarize an individual transcript segment."""
        logger.debug("Summarizing transcript chunk (%d characters)", len(text))
        prompt = CHUNK_SYSTEM_PROMPT_TEMPLATE.format(text=text)
        return self.llm.TextLLM(
            system_instruction=prompt,
            history=[],
            query="Provide your summary below: ",
        )

    def summary_of_summaries(self, summaries: list[str]) -> str:
        """Merge multiple chunk summaries into a cohesive document summary."""
        combined_text = "\n\n".join(summaries)
        logger.info("Merging %d chunk summaries into final summary", len(summaries))
        prompt = MERGE_SYSTEM_PROMPT_TEMPLATE.format(summaries=combined_text)
        return self.llm.TextLLM(
            system_instruction=prompt,
            history=[],
            query="Provide the final summary below: ",
        )

    def summarize_transcript(self, transcript_text_list: list[str], max_chars: int = 100_000) -> str:
        """Summarize full transcript using map-reduce if length exceeds max_chars."""
        big_text = " ".join(transcript_text_list).strip()
        if not big_text:
            return ""

        logger.info("Starting transcript summarization (total characters: %d, threshold: %d)", len(big_text), max_chars)

        if len(big_text) > max_chars:
            chunks = chunk_by_sentences(big_text, max_chars=max_chars)
            logger.info("Split large transcript into %d chunks for map-reduce summarization", len(chunks))
            chunk_summaries = [self.summarize_chunk(chunk) for chunk in chunks]
            return self.summary_of_summaries(chunk_summaries)

        return self.summarize_chunk(big_text)
