from groq import Groq
from google import genai
from google.genai import types
import os


class GroqLLM:
    def __init__(self, api_key):
        self.llm = Groq(api_key=api_key)

    def AudioLLM(self, audio_paths, model_name="whisper-large-v3-turbo"):
        self.audio_paths = audio_paths
        self.model_name = model_name
        self.transcription_lists = []

        for audio_path in self.audio_paths:
            with open(audio_path, "rb") as file:
                self.transcription = self.llm.audio.transcriptions.create(
                    file=(audio_path, file.read()),
                    language="en",
                    model=self.model_name,
                    response_format="verbose_json",
                )

            if os.path.exists(audio_path):
                os.remove(audio_path)

            self.transcription_segments = self.transcription.segments
            self.transcription_list = [
                {
                    "text": each_transcription_segment["text"],
                    "start": each_transcription_segment["start"],
                }
                for each_transcription_segment in self.transcription_segments
            ]
            self.transcription_lists.extend(self.transcription_list)

        return self.transcription_lists


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
    api_key = os.getenv("GEMINI_API_KEY")
    llm = GeminiLLM(api_key)
    print(llm.TextLLM("answer shorltly", [], "Hello"))
