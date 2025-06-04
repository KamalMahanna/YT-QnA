from youtube_transcript_api import YouTubeTranscriptApi
from .AudioDownloader import AudioDownloader
from .LLM import GroqLLM


class Transcript:
    def __init__(self):
        pass

    def with_whisper(self, api_key, video_id, model_name="whisper-large-v3-turbo"):
        self.video_id = video_id
        self.model_name = model_name

        self.audio_downloader = AudioDownloader()
        self.audio_path = self.audio_downloader.download_audio(self.video_id)
        self.llm = GroqLLM(api_key)
        self.transcript_list = self.llm.AudioLLM(self.audio_path, self.model_name)
        return self.transcript_list

    def with_youtube_api(self, video_id):
        self.video_id = video_id
        self.transcript_api = YouTubeTranscriptApi()
        self.transcript_list = self.transcript_api.get_transcript(
            self.video_id, languages=["en"]
        )
        return self.transcript_list


if __name__ == "__main__":
    transcript = Transcript()

    # print(transcript.with_whisper("0zhPaRZev8w"))
    print(transcript.with_whisper("2nNGN72eYiU"))

    # if os.path.exists('/tmp/tmphr_7t9ae.mp4'):
    #     print(transcript.with_whisper("/tmp/tmphr_7t9ae.mp4"))
    # else:
    #     print("Not exist")
