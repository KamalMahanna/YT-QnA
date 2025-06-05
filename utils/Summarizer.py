from .LLM import GeminiLLM
from .HelperFunctions import chunk_by_sentences


class Summarizer:
    def __init__(self, api_key):
        self.llm = GeminiLLM(api_key)

    def summarize_chunk(self, text:str):
        self.text = text

        self.system_prompt = (
            "You are a precise and faithful summarizer. "
            "Below is a transcript chunk from a YouTube video. "
            "Your task is to summarize the content of this chunk only. "
            "Do not add any information that is not explicitly present in the chunk. "
            "Do not attempt to predict or infer anything beyond the given text. "
            "Stay within the scope of this chunk. "
            "Focus on preserving the speaker's intent, "
            "and ensure the summary is clear and concise while reflecting the actual content. "
            "If the chunk includes multiple topics, reflect that in your summary as well. "
            "Please do not include Ads or markettings or promotions in the summary. "
            f"""Transcript Chunk: 
            ---
            {self.text}
            ---
            """
        )

        self.history = []
        
        self.model = self.llm.TextLLM(
            system_instruction=self.system_prompt,
            history=self.history,
            query="Provide your summary below: ",
        )

        return self.model

    def summary_of_summaries(self, summaries):
        self.summaries = summaries

        self.system_prompt = (
            "You are a summarizer responsible for combining multiple independent "
            "summaries into a final coherent summary. Each summary corresponds "
            "to a chunk of a longer transcript. "
            "Your task is to merge these summaries into one comprehensive, "
            "faithful summary of the full content, without introducing any new "
            "assumptions or interpretations. Do not hallucinate or infer beyond "
            "what the summaries provide. "
            "Ensure that the final output is cohesive and preserves the structure, "
            "themes, and key information reflected across the chunk summaries."
            f"""Chunk Summaries: 
            ---
            {self.summaries}
            ---
            """
        )

        self.history = []
        
        self.model = self.llm.TextLLM(
            system_instruction=self.system_prompt,
            history=self.history,
            query="Provide the final summary below: ",
        )

        return self.model


    def summarize_transcript(self, transcript_text_list, max_chars=100_000):
        self.max_chars = max_chars
        self.transcript_text_list = transcript_text_list
        self.big_text = " ".join(self.transcript_text_list)

        if len(self.big_text) > self.max_chars:
            self.chunks = chunk_by_sentences(self.big_text, self.max_chars)

            self.summaries = []
            for each_chunk in self.chunks:

                self.summaries.append(
                    self.summarize_chunk(each_chunk)
                )

            return self.summary_of_summaries(self.summaries)
        else:
            return self.summarize_chunk(self.big_text)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path="../.env")
    api_key = os.getenv("GEMINI_API_KEY")

    summarizer = Summarizer(api_key)
    print(summarizer.summarize_transcript("Others who use this device won’t see your activity, so you can browse more privately. This won't change how data is collected by websites that you visit and the services that they use, including Google. Downloads, bookmarks and reading list items will be saved"))
