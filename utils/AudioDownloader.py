import logging
import os
import tempfile

import numpy as np
from pydub import AudioSegment
from pytubefix import YouTube

# Set up logger for this module
logger = logging.getLogger(__name__)


class AudioDownloader:
    """
    Handles downloading audio from YouTube videos and splitting them into manageable chunks.
    """
    def __init__(self):
        """
        Initializes the AudioDownloader.
        """
        logger.info("AudioDownloader initialized.")
        pass
    
    def download_audio(self, video_id: str) -> list:
        """
        Downloads the audio from a given YouTube video ID and splits it into
        chunks if the file size exceeds 15MB.

        Args:
            video_id (str): The ID of the YouTube video.

        Returns:
            list: A list of paths to the temporary audio files (chunks).
        """
        self.video_id = video_id
        self.url = f"https://www.youtube.com/watch?v={self.video_id}"
        logger.info(f"Attempting to download audio for video ID: {self.video_id}")

        try:
            self.yt = YouTube(self.url)
            self.audio_stream = self.yt.streams.filter(only_audio=True, abr="128kbps").first()
            if not self.audio_stream:
                logger.error(f"No audio stream found for video ID: {self.video_id}")
                return []
            logger.debug(f"Found audio stream: {self.audio_stream.title}")
        except Exception as e:
            logger.error(f"Error accessing YouTube video {self.video_id}: {e}")
            return []

        with tempfile.NamedTemporaryFile(
            suffix=".m4a", delete=False
        ) as self.temp_audio:
            try:
                # Download the audio stream
                self.audio_directory = self.temp_audio.name.split("/")
                self.audio_stream.download(
                    filename=self.audio_directory[-1],
                    output_path="/" + self.audio_directory[1],
                )
                logger.info(f"Audio downloaded to {self.temp_audio.name}")
            except Exception as e:
                logger.error(f"Error downloading audio for video ID {self.video_id}: {e}")
                return []

            # Check if the downloaded audio file size exceeds 15MB
            if os.path.getsize(self.temp_audio.name) > (15 * 1024**2):
                logger.info("Audio file size exceeds 15MB, splitting into chunks.")
                try:
                    self.audio = AudioSegment.from_file(self.temp_audio.name)
                    self.temp_audio_paths = []
                    # Calculate number of required chunks
                    self.no_of_required_chunks = int(np.ceil(os.path.getsize(self.temp_audio.name) / (15 * 1024**2)))
                    # Define segments for splitting
                    self.segments = np.linspace(0, len(self.audio), self.no_of_required_chunks + 1)

                    for i, (start_pt, end_pt) in enumerate(zip(self.segments[:-1], self.segments[1:])):
                        with tempfile.NamedTemporaryFile(
                            suffix=".mp3", delete=False
                        ) as self.temp_audio_chunk:
                            # Export each segment as an MP3 chunk
                            self.audio[start_pt:end_pt].export(self.temp_audio_chunk.name, format="mp3")
                            self.temp_audio_paths.append(self.temp_audio_chunk.name)
                            self.temp_audio_chunk.flush()
                            logger.debug(f"Created audio chunk {i+1}: {self.temp_audio_chunk.name}")
                    
                    logger.info(f"Successfully split audio into {len(self.temp_audio_paths)} chunks.")
                    return self.temp_audio_paths
                except Exception as e:
                    logger.error(f"Error splitting audio file {self.temp_audio.name}: {e}")
                    return []
            else:
                logger.info("Audio file size is within limits, returning single file.")
                self.temp_audio.flush()
                return [self.temp_audio.name]
