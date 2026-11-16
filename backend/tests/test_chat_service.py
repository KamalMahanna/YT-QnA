import json
import unittest
from unittest.mock import MagicMock, patch

from api.chat_service import ChatService


class TestChatService(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_store = MagicMock()

    @patch("api.chat_service.genai.Client")
    @patch("api.chat_service.GeminiLLM")
    @patch("api.chat_service.Summarizer")
    def test_build_rag_context(self, mock_sum_cls, mock_llm_cls, mock_genai_client_cls):
        service = ChatService(gemini_api_key="fake_key", store=self.mock_store)

        chunks = [
            {"start": 30, "text": "This is segment 1."},
            {"start": 125, "text": "This is segment 2."},
        ]

        context_str, sources = service.build_rag_context(chunks, "v123")
        self.assertIn("[00:30](https://www.youtube.com/watch?v=v123&t=30s)", context_str)
        self.assertIn("[02:05](https://www.youtube.com/watch?v=v123&t=125s)", context_str)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["timestamp"], "00:30")
        self.assertEqual(sources[1]["timestamp"], "02:05")

    @patch("api.chat_service.genai.Client")
    @patch("api.chat_service.GeminiLLM")
    @patch("api.chat_service.Summarizer")
    async def test_stream_rag_answer(self, mock_sum_cls, mock_llm_cls, mock_genai_client_cls):
        mock_client = MagicMock()
        mock_genai_client_cls.return_value = mock_client

        # Mock chunk stream response
        chunk1 = MagicMock(text="Hello ")
        chunk2 = MagicMock(text="world!")
        mock_client.models.generate_content_stream.return_value = [chunk1, chunk2]

        self.mock_store.retrieve_documents.return_value = [
            {"start": 10, "text": "Relevant snippet."}
        ]

        service = ChatService(gemini_api_key="fake_key", store=self.mock_store)

        events = []
        async for sse_event in service.stream_rag_answer("What is this?", "v123"):
            events.append(sse_event)

        # First event is sources
        self.assertTrue(events[0].startswith("data: "))
        evt0 = json.loads(events[0][6:])
        self.assertEqual(evt0["type"], "sources")
        self.assertEqual(len(evt0["sources"]), 1)

        # Tokens
        tokens = [json.loads(e[6:])["token"] for e in events if json.loads(e[6:])["type"] == "token"]
        self.assertEqual("".join(tokens), "Hello world!")

        # Done event
        last_evt = json.loads(events[-1][6:])
        self.assertEqual(last_evt["type"], "done")

    @patch("api.chat_service.genai.Client")
    @patch("api.chat_service.GeminiLLM")
    @patch("api.chat_service.Summarizer")
    async def test_generate_summary(self, mock_sum_cls, mock_llm_cls, mock_genai_client_cls):
        mock_sum = MagicMock()
        mock_sum.summarize_transcript.return_value = "Video summary result."
        mock_sum_cls.return_value = mock_sum

        service = ChatService(gemini_api_key="fake_key", store=self.mock_store)
        summary = await service.generate_summary(["chunk1", "chunk2"])
        self.assertEqual(summary, "Video summary result.")


if __name__ == "__main__":
    unittest.main()
