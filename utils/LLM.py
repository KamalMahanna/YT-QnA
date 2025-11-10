from google import genai
from google.genai import types
import os


class GeminiLLM:
    def __init__(self, api_key):
        self.llm = genai.Client(api_key=api_key)

    def TextLLM(
        self,
        system_instruction,
        history,
        query,
        model_name="gemini-2.5-flash",
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

    def TTS(self, texts):
        self.response = self.llm.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=f"Read aloud in a energetic and friendly tone: {texts}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Zephyr",
                        )
                    )
                ),
            ),
        )
        return self.response.candidates[0].content.parts[0].inline_data.data

    def __call__(self, input):
        self.input = input

        self.embedding = self.llm.models.embed_content(
            model="gemini-embedding-001",
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
