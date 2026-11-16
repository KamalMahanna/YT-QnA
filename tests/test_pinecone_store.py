import os
import unittest
from unittest.mock import MagicMock, patch

from DataBases.PineconeStore import PineconeStore


class TestPineconeStore(unittest.TestCase):
    def setUp(self):
        self.mock_env = {
            "PINECONE_API_KEY": "fake_pinecone_key",
            "GEMINI_API_KEY": "fake_gemini_key",
            "PINECONE_INDEX_NAME": "test-index",
        }

    @patch("DataBases.PineconeStore.Pinecone")
    @patch("DataBases.PineconeStore.GeminiLLM")
    def test_initialization_and_ensure_index(self, mock_llm_cls, mock_pinecone_cls):
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc
        # Mock index list (index doesn't exist yet)
        mock_pc.list_indexes.return_value = []

        store = PineconeStore(
            pinecone_api_key="key123",
            gemini_api_key="gem123",
            index_name="test-index",
            dimension=768,
        )

        mock_pc.create_index.assert_called_once()
        mock_pc.Index.assert_called_with("test-index")
        self.assertEqual(store.dimension, 768)

    @patch("DataBases.PineconeStore.Pinecone")
    @patch("DataBases.PineconeStore.GeminiLLM")
    def test_video_exists_true_and_false(self, mock_llm_cls, mock_pinecone_cls):
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]

        store = PineconeStore(
            pinecone_api_key="key123",
            gemini_api_key="gem123",
            index_name="test-index",
        )

        # Case 1: Video exists
        mock_stats = MagicMock()
        mock_stats.namespaces = {"v123": MagicMock(vector_count=10)}
        mock_index.describe_index_stats.return_value = mock_stats
        self.assertTrue(store.video_exists("v123"))

        # Case 2: Video does not exist
        self.assertFalse(store.video_exists("v999"))

    @patch("DataBases.PineconeStore.Pinecone")
    @patch("DataBases.PineconeStore.GeminiLLM")
    def test_add_documents_upsert(self, mock_llm_cls, mock_pinecone_cls):
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]

        mock_llm = MagicMock()
        mock_llm.return_value = [[0.1] * 768, [0.2] * 768]
        mock_llm_cls.return_value = mock_llm

        store = PineconeStore(
            pinecone_api_key="key123",
            gemini_api_key="gem123",
            index_name="test-index",
        )

        store.add_documents(
            documents=["chunk 1", "chunk 2"],
            timestamps=[10.5, 25.0],
            video_id="v123",
            batch_size=50,
        )

        mock_index.upsert.assert_called_once()
        call_kwargs = mock_index.upsert.call_args.kwargs
        self.assertEqual(call_kwargs["namespace"], "v123")
        self.assertEqual(len(call_kwargs["vectors"]), 2)
        self.assertEqual(call_kwargs["vectors"][0]["id"], "v123#chunk_0")
        self.assertEqual(call_kwargs["vectors"][0]["metadata"]["start"], 10.5)

    @patch("DataBases.PineconeStore.Pinecone")
    @patch("DataBases.PineconeStore.GeminiLLM")
    def test_retrieve_documents(self, mock_llm_cls, mock_pinecone_cls):
        mock_pc = MagicMock()
        mock_pinecone_cls.return_value = mock_pc
        mock_index = MagicMock()
        mock_pc.Index.return_value = mock_index
        mock_pc.list_indexes.return_value = [MagicMock(name="test-index")]

        mock_llm = MagicMock()
        mock_llm.embed_query.return_value = [[0.05] * 768]
        mock_llm_cls.return_value = mock_llm

        mock_match = MagicMock()
        mock_match.metadata = {"text": "hello world", "start": 12.0}
        mock_match.score = 0.95
        mock_index.query.return_value = MagicMock(matches=[mock_match])

        store = PineconeStore(
            pinecone_api_key="key123",
            gemini_api_key="gem123",
            index_name="test-index",
        )

        results = store.retrieve_documents("hello", "v123", n_results=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["text"], "hello world")
        self.assertEqual(results[0]["start"], 12)


if __name__ == "__main__":
    unittest.main()
