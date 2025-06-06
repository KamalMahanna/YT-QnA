import logging
import os

from google import genai
from google.genai import types
from groq import Groq

# Set up logger for this module
logger = logging.getLogger(__name__)


class GroqLLM:
    """
    A class to interact with Groq's Large Language Models for audio transcription.
    """
    def __init__(self, api_key: str):
        """
        Initializes the GroqLLM client.

        Args:
            api_key (str): Your Groq API key.
        """
        self.llm = Groq(api_key=api_key)
        logger.info("GroqLLM initialized.")

    def AudioLLM(self, audio_paths: list, model_name: str = "whisper-large-v3-turbo") -> list:
        """
        Transcribes audio files using Groq's Whisper model.

        Args:
            audio_paths (list): A list of paths to the audio files to transcribe.
            model_name (str, optional): The name of the Whisper model to use.
                                        Defaults to "whisper-large-v3-turbo".

        Returns:
            list: A list of dictionaries, where each dictionary represents a segment
                  of the transcription with 'text' and 'start' time.
        """
        self.audio_paths = audio_paths
        self.model_name = model_name
        self.transcription_lists = []
        logger.info(f"Starting audio transcription with GroqLLM using model: {self.model_name}")

        for audio_path in self.audio_paths:
            logger.debug(f"Processing audio file: {audio_path}")
            try:
                with open(audio_path, "rb") as file:
                    self.transcription = self.llm.audio.transcriptions.create(
                        file=(audio_path, file.read()),
                        language="en",
                        model=self.model_name,
                        response_format="verbose_json",
                    )
                logger.info(f"Transcription successful for {audio_path}.")
            except Exception as e:
                logger.error(f"Error transcribing audio file {audio_path}: {e}")
                continue # Continue to next audio file if one fails

            # Clean up temporary audio file
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.debug(f"Removed temporary audio file: {audio_path}")

            self.transcription_segments = self.transcription.segments
            self.transcription_list = [
                {
                    "text": each_transcription_segment["text"],
                    "start": each_transcription_segment["start"],
                }
                for each_transcription_segment in self.transcription_segments
            ]
            self.transcription_lists.extend(self.transcription_list)
            logger.debug(f"Added {len(self.transcription_list)} segments to transcription list.")

        logger.info("Finished audio transcription.")
        return self.transcription_lists


class GeminiLLM:
    """
    A class to interact with Google's Gemini Large Language Models for text generation
    and embeddings.
    """
    def __init__(self, api_key: str):
        """
        Initializes the GeminiLLM client.

        Args:
            api_key (str): Your Gemini API key.
        """
        self.llm = genai.Client(api_key=api_key)
        logger.info("GeminiLLM initialized.")

    def TextLLM(
        self, system_instruction: str, history: list, query: str, model_name: str = "gemini-2.5-flash-preview-05-20"
    ) -> str:
        """
        Generates a text response using a Gemini chat model.

        Args:
            system_instruction (str): System-level instructions for the model.
            history (list): A list of previous chat messages to maintain context.
            query (str): The user's current query.
            model_name (str, optional): The name of the Gemini text model to use.
                                        Defaults to "gemini-2.5-flash-preview-05-20".

        Returns:
            str: The generated text response from the LLM.
        """
        self.history = history
        self.system_instruction = system_instruction
        self.model_name = model_name
        logger.info(f"Calling Gemini TextLLM with model: {self.model_name}")
        logger.debug(f"System Instruction: {self.system_instruction[:100]}...")
        logger.debug(f"Query: {query[:50]}...")

        try:
            self.model = self.llm.chats.create(
                model=self.model_name,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_instruction
                ),
                history=self.history,
            )
            self.response = self.model.send_message(query)
            logger.info("Gemini TextLLM call successful.")
            return self.response.text
        except Exception as e:
            logger.error(f"Error calling Gemini TextLLM: {e}")
            return "An error occurred while generating the response."

    def __call__(self, input: str) -> list:
        """
        Generates embeddings for the given input text using a Gemini embedding model.
        This method makes the class instance callable.

        Args:
            input (str): The text for which to generate embeddings.

        Returns:
            list: A list of embedding values.
        """
        self.input = input
        logger.info("Generating embeddings using Gemini embedding model.")
        logger.debug(f"Embedding input: {self.input[:50]}...")

        try:
            self.embedding = self.llm.models.embed_content(
                model="models/text-embedding-004",
                contents=self.input,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            logger.info("Embedding generation successful.")
            return [each_embedding.values for each_embedding in self.embedding.embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    def name(self) -> str:
        """
        Returns the name of the embedding function.

        Returns:
            str: The string "gemini_embedding".
        """
        return "gemini_embedding"


if __name__ == "__main__":
    # Example usage for testing purposes
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if api_key:
        llm = GeminiLLM(api_key)
        logger.info("Testing GeminiLLM.TextLLM with a simple query.")
        print(llm.TextLLM("answer shortly", [], "Hello"))
    else:
        logger.warning("GEMINI_API_KEY not found in .env for testing LLM.py.")
        print("GEMINI_API_KEY not found. Please set it in your .env file.")

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_llm = GroqLLM(groq_api_key)
        logger.info("Testing GroqLLM.AudioLLM (requires a dummy audio file).")
        # This part would require a dummy audio file for a full test
        # For now, just log that it's ready to test
        print("GroqLLM initialized. To test AudioLLM, provide a valid audio file path.")
    else:
        logger.warning("GROQ_API_KEY not found in .env for testing LLM.py.")
        print("GROQ_API_KEY not found. Please set it in your .env file.")
