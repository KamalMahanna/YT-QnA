from pytubefix import YouTube
import tempfile
import os

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

            if os.path.getsize(self.temp_audio.name) >= (19 * 1024**2):
                raise ValueError(
                    "Video length crossed 19 minute. Please try with a smaller video. But you can install YT-QnA ( https://github.com/KamalMahanna/YT-QnA.git ) locally to process lengthy video"
                )

            else:
                self.temp_audio.flush()
                return [self.temp_audio.name]

        # fix timestamps for multiple audio chunks
