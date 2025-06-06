import logging

from .HelperFunctions import chunk_by_sentences
from .LLM import GeminiLLM

# Set up logger for this module
logger = logging.getLogger(__name__)


class Summarizer:
    """
    A class responsible for summarizing YouTube video transcripts using a Gemini LLM.
    It can summarize individual chunks and combine multiple summaries into a final one.
    """
    def __init__(self, api_key: str):
        """
        Initializes the Summarizer with a GeminiLLM instance.

        Args:
            api_key (str): API key for Gemini LLM.
        """
        self.llm = GeminiLLM(api_key)
        logger.info("Summarizer initialized.")

    def summarize_chunk(self, text: str) -> str:
        """
        Summarizes a single chunk of text using the Gemini LLM.

        Args:
            text (str): The text chunk to be summarized.

        Returns:
            str: The summarized text of the chunk.
        """
        self.text = text
        logger.info("Summarizing a single text chunk.")
        logger.debug(f"Chunk text (first 100 chars): {self.text[:100]}...")

        self.system_prompt = (
            "You are a precise and faithful summarizer. "
            "Below is a transcript chunk from a YouTube video. "
            "Your task is to summarize the content of this chunk only. "
            "Do not add any information that is not explicitly present in the chunk. "
            "Do not attempt to predict or infer anything beyond the given text. "
            "Stay within the scope of this chunk. "
            "Focus on preserving the speaker's intent, "
            "and ensure the summary is clear and concise while reflecting the actual content. "
            "If the chunk includes multiple topics, reflect that in your summary as well. "
            "Please do not include Ads or markettings or promotions in the summary. "
            f"""Transcript Chunk: 
            ---
            {self.text}
            ---
            """
        )

        self.history = []
        
        try:
            self.model = self.llm.TextLLM(
                system_instruction=self.system_prompt,
                history=self.history,
                query="Provide your summary below: ",
            )
            logger.info("Chunk summarization successful.")
            return self.model
        except Exception as e:
            logger.error(f"Error summarizing chunk: {e}")
            return "Error: Could not summarize chunk."

    def summary_of_summaries(self, summaries: list) -> str:
        """
        Combines multiple individual chunk summaries into one comprehensive final summary.

        Args:
            summaries (list): A list of strings, where each string is a summary of a chunk.

        Returns:
            str: The combined, comprehensive summary.
        """
        self.summaries = summaries
        logger.info("Combining multiple chunk summaries into a final summary.")
        logger.debug(f"Number of summaries to combine: {len(self.summaries)}")

        self.system_prompt = (
            "You are a summarizer responsible for combining multiple independent "
            "summaries into a final coherent summary. Each summary corresponds "
            "to a chunk of a longer transcript. "
            "Your task is to merge these summaries into one comprehensive, "
            "faithful summary of the full content, without introducing any new "
            "assumptions or interpretations. Do not hallucinate or infer beyond "
            "what the summaries provide. "
            "Ensure that the final output is cohesive and preserves the structure, "
            "themes, and key information reflected across the chunk summaries."
            f"""Chunk Summaries: 
            ---
            {self.summaries}
            ---
            """
        )

        self.history = []
        
        try:
            self.model = self.llm.TextLLM(
                system_instruction=self.system_prompt,
                history=self.history,
                query="Provide the final summary below: ",
            )
            logger.info("Summary of summaries successful.")
            return self.model
        except Exception as e:
            logger.error(f"Error combining summaries: {e}")
            return "Error: Could not combine summaries."


    def summarize_transcript(self, transcript_text_list: list, max_chars: int = 100_000) -> str:
        """
        Summarizes an entire transcript, handling large texts by chunking and
        then combining individual chunk summaries.

        Args:
            transcript_text_list (list): A list of strings, where each string is a
                                         segment of the transcript.
            max_chars (int, optional): The maximum character limit for a single chunk
                                       before it's split. Defaults to 100,000.

        Returns:
            str: The final comprehensive summary of the transcript.
        """
        self.max_chars = max_chars
        self.transcript_text_list = transcript_text_list
        self.big_text = " ".join(self.transcript_text_list)
        logger.info(f"Starting transcript summarization. Total text length: {len(self.big_text)}")

        if len(self.big_text) > self.max_chars:
            logger.info(f"Transcript too large ({len(self.big_text)} chars), chunking for summarization.")
            self.chunks = chunk_by_sentences(self.big_text, self.max_chars)
            logger.debug(f"Transcript split into {len(self.chunks)} chunks.")

            self.summaries = []
            for i, each_chunk in enumerate(self.chunks):
                logger.debug(f"Summarizing chunk {i+1}/{len(self.chunks)}.")
                self.summaries.append(
                    self.summarize_chunk(each_chunk)
                )
            logger.info("All chunks summarized. Combining summaries.")
            return self.summary_of_summaries(self.summaries)
        else:
            logger.info("Transcript size is manageable, summarizing as a single chunk.")
            return self.summarize_chunk(self.big_text)


if __name__ == "__main__":
    # Example usage for testing purposes
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        summarizer = Summarizer(api_key)
        test_text = "Others who use this device won’t see your activity, so you can browse more privately. This won't change how data is collected by websites that you visit and the services that they use, including Google. Downloads, bookmarks and reading list items will be saved."
        logger.info(f"Testing summarizer with text: {test_text[:50]}...")
        print(summarizer.summarize_transcript(test_text))
    else:
        logger.warning("GEMINI_API_KEY not found in .env for testing Summarizer.py.")
        print("GEMINI_API_KEY not found. Please set it in your .env file.")
