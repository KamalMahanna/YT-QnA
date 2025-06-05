from pytubefix import YouTube
import tempfile
import os
from pydub import AudioSegment
import numpy as np


class AudioDownloader:
    def __init__(self):
        pass
    
    def download_audio(self, video_id):
        self.video_id = video_id
        self.url = "https://www.youtube.com/watch?v=" + self.video_id
        self.yt = YouTube(self.url)
        self.audio_stream = self.yt.streams.filter(only_audio=True, abr="128kbps")

        with tempfile.NamedTemporaryFile(
            suffix=".m4a", delete=False
        ) as self.temp_audio:

            self.audio_directory = self.temp_audio.name.split("/")
            self.audio_stream.first().download(
                filename=self.audio_directory[-1],
                output_path="/" + self.audio_directory[1],
            )

            if os.path.getsize(self.temp_audio.name) > (15 * 1024**2):
                self.audio = AudioSegment.from_file(self.temp_audio.name)
                self.temp_audio_paths = []
                self.no_of_required_chunks = int(np.ceil(os.path.getsize(self.temp_audio.name) / (15 * 1024**2)))
                self.segments = np.linspace(0, os.path.getsize(self.temp_audio.name)+1, self.no_of_required_chunks + 1)

                for start_pt, end_pt in zip(self.segments[:-1], self.segments[1:]):
                    with tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    ) as self.temp_audio_chunk:
                        self.audio[start_pt:end_pt].export(self.temp_audio_chunk.name, format="mp3")
                        self.temp_audio_paths.append(self.temp_audio_chunk.name)
                    
                        self.temp_audio_chunk.flush()
                
                return self.temp_audio_paths
                
            else:
                self.temp_audio.flush()
                return [self.temp_audio.name]

        