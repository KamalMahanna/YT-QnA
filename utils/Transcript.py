from youtube_transcript_api import YouTubeTranscriptApi



class Transcript:
    def __init__(self):
        pass

    def with_youtube_api(self, video_id):
        self.video_id = video_id
        self.transcript_api = YouTubeTranscriptApi()
        self.transcript_list = self.transcript_api.fetch(
            self.video_id, languages=["en"]
        )
        return self.transcript_list
