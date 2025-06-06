import logging
import os

from youtube_transcript_api import YouTubeTranscriptApi

from .AudioDownloader import AudioDownloader
from .LLM import GroqLLM

# Set up logger for this module
logger = logging.getLogger(__name__)


class Transcript:
    """
    Handles fetching YouTube video transcripts using either the YouTube Transcript API
    or by transcribing audio with Whisper (via GroqLLM).
    """
    def __init__(self):
        """
        Initializes the Transcript handler.
        """
        logger.info("Transcript handler initialized.")
        pass

    def with_whisper(self, api_key: str, video_id: str, model_name: str = "whisper-large-v3-turbo") -> list:
        """
        Fetches the transcript of a YouTube video by downloading its audio
        and transcribing it using Groq's Whisper model.

        Args:
            api_key (str): Groq API key for the Whisper model.
            video_id (str): The ID of the YouTube video.
            model_name (str, optional): The name of the Whisper model to use.
                                        Defaults to "whisper-large-v3-turbo".

        Returns:
            list: A list of dictionaries, where each dictionary represents a segment
                  of the transcription with 'text' and 'start' time.
                  Returns an empty list if transcription fails.
        """
        self.video_id = video_id
        self.model_name = model_name
        logger.info(f"Attempting to get transcript for video ID {self.video_id} using Whisper.")

        try:
            self.audio_downloader = AudioDownloader()
            self.audio_paths = self.audio_downloader.download_audio(self.video_id)
            
            if not self.audio_paths:
                logger.error(f"Failed to download audio for video ID: {self.video_id}. Cannot proceed with Whisper transcription.")
                return []

            self.llm = GroqLLM(api_key)
            self.transcript_list = self.llm.AudioLLM(self.audio_paths, self.model_name)
            logger.info(f"Successfully transcribed audio for video ID {self.video_id} with Whisper.")
            return self.transcript_list
        except Exception as e:
            logger.error(f"Error during Whisper transcription for video ID {self.video_id}: {e}")
            return []

    def with_youtube_api(self, video_id: str) -> list:
        """
        Fetches the transcript of a YouTube video using the official YouTube Transcript API.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            list: A list of dictionaries, where each dictionary represents a segment
                  of the transcript with 'text' and 'start' time.
                  Returns an empty list if transcript retrieval fails.
        """
        self.video_id = video_id
        self.transcript_api = YouTubeTranscriptApi()
        logger.info(f"Attempting to get transcript for video ID {self.video_id} using YouTube API.")

        try:
            self.transcript_list = self.transcript_api.get_transcript(
                self.video_id, languages=["en"]
            )
            logger.info(f"Successfully retrieved transcript for video ID {self.video_id} with YouTube API.")
            return self.transcript_list
        except Exception as e:
            logger.error(f"Error retrieving transcript for video ID {self.video_id} with YouTube API: {e}")
            return []


if __name__ == "__main__":
    # Example usage for testing purposes
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    groq_api_key = os.getenv("GROQ_API_KEY")

    transcript = Transcript()

    # Example video ID for testing
    test_video_id = "2nNGN72eYiU" # Replace with a valid video ID for testing

    if groq_api_key:
        logger.info(f"Testing with_whisper for video ID: {test_video_id}")
        whisper_transcript = transcript.with_whisper(groq_api_key, test_video_id)
        if whisper_transcript:
            print(f"Whisper Transcript (first 3 segments): {whisper_transcript[:3]}")
        else:
            print("Whisper transcription failed or returned empty.")
    else:
        logger.warning("GROQ_API_KEY not found in .env for testing Transcript.py with Whisper.")
        print("GROQ_API_KEY not found. Cannot test Whisper transcription.")

    logger.info(f"Testing with_youtube_api for video ID: {test_video_id}")
    youtube_transcript = transcript.with_youtube_api(test_video_id)
    if youtube_transcript:
        print(f"YouTube API Transcript (first 3 segments): {youtube_transcript[:3]}")
    else:
        print("YouTube API transcription failed or returned empty.")
