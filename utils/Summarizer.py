from .LLM import GeminiLLM
from nltk.tokenize import sent_tokenize


class Summarizer:
    def __init__(self, api_key):
        self.llm = GeminiLLM(api_key)

    def chunk_by_sentences(self, text, max_chars=100_000):
        self.text = text
        self.max_chars = max_chars

        self.sentences = sent_tokenize(self.text)
        self.chunks, self.current_chunk = [], ""
        for sentence in self.sentences:
            if len(self.current_chunk) + len(sentence) <= self.max_chars:
                self.current_chunk += " " + sentence
            else:
                self.chunks.append(self.current_chunk.strip())
                self.current_chunk = sentence
        if self.current_chunk:
            self.chunks.append(self.current_chunk.strip())
        return self.chunks

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
            self.chunks = self.chunk_by_sentences(self.big_text, self.max_chars)

            self.summaries = []
            for each_chunk in self.chunks:

                self.summaries.append(
                    self.summarize_chunk(each_chunk)
                )

            self.summary = self.summary_of_summaries(self.summaries)
            return self.summary
        else:
            return self.summarize_chunk(self.big_text)