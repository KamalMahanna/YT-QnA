from groq import Groq
from google import genai
from google.genai import types
import os


class GroqLLM:
    def __init__(self, api_key):
        self.llm = Groq(api_key=api_key)

    def AudioLLM(self, audio_path, model_name="whisper-large-v3-turbo"):
        self.audio_path = audio_path
        self.model_name = model_name

        with open(self.audio_path, "rb") as file:
            self.transcription = self.llm.audio.transcriptions.create(
                file=(self.audio_path, file.read()),
                language="en",
                model=self.model_name,
                response_format="verbose_json",
            )

        if os.path.exists(self.audio_path):
            os.remove(self.audio_path)

        self.transcription_segments = self.transcription.segments
        self.transcription_list = [
            {
                "text": each_transcription_segment["text"],
                "start": each_transcription_segment["start"],
            }
            for each_transcription_segment in self.transcription_segments
        ]

        return self.transcription_list


class GeminiLLM:
    def __init__(self, api_key):
        self.llm = genai.Client(api_key=api_key)

    def TextLLM(
        self, system_instruction, history, query, model_name="gemini-2.5-flash-preview-05-20"
    ):
        self.history = history
        self.system_instruction = system_instruction
        self.model_name = model_name

        self.model = self.llm.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction
            ),
            history=self.history,
        )
        self.response = self.model.send_message(query)
        return self.response.text

    def __call__(self, input):
        self.input = input

        self.embedding = self.llm.models.embed_content(
            model="models/text-embedding-004",
            contents=self.input,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        return [each_embedding.values for each_embedding in self.embedding.embeddings]

    def name(self):
        return "gemini_embedding"


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    api_key = os.getenv("GROQ_API_KEY")
    llm = GroqLLM(api_key)
    print(llm.AudioLLM("/tmp/tmp6g5jvppl.m4a"))
