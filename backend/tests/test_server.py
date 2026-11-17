import unittest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.server import create_app
from api.services import STATUS_PROCESSING, STATUS_READY


class TestServerEndpoints(unittest.TestCase):
    def setUp(self):
        self.mock_store = MagicMock()
        self.mock_coordinator = MagicMock()
        self.mock_chat_service = MagicMock()

        self.app = create_app(
            store=self.mock_store,
            coordinator=self.mock_coordinator,
            chat_service=self.mock_chat_service,
        )
        self.client = TestClient(self.app)

    def test_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "yt-qna-backend")

    def test_process_video_invalid_url(self):
        response = self.client.post("/api/video/process", json={"url": "not-a-valid-url"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid YouTube URL", response.json()["detail"])

    def test_process_video_ready(self):
        self.mock_coordinator.process_video = AsyncMock(
            return_value={"status": STATUS_READY, "video_id": "dQw4w9WgXcQ", "chunk_count": 12}
        )

        response = self.client.post(
            "/api/video/process",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], STATUS_READY)
        self.assertEqual(data["video_id"], "dQw4w9WgXcQ")

    def test_process_video_processing_accepted(self):
        self.mock_coordinator.process_video = AsyncMock(
            return_value={
                "status": STATUS_PROCESSING,
                "video_id": "dQw4w9WgXcQ",
                "retry_after": 2,
            }
        )

        response = self.client.post(
            "/api/video/process",
            json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"},
        )
        self.assertEqual(response.status_code, 202)
        data = response.json()
        self.assertEqual(data["status"], STATUS_PROCESSING)

    def test_get_video_status(self):
        self.mock_coordinator.get_status.return_value = {
            "video_id": "abc12345678",
            "status": STATUS_READY,
        }

        response = self.client.get("/api/video/status/abc12345678")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], STATUS_READY)

    def test_get_video_timeline(self):
        self.mock_coordinator.get_video_timeline.return_value = [
            {"chunk_id": "chunk_0", "start": 0.0, "timestamp": "00:00", "text": "Intro"}
        ]

        response = self.client.get("/api/video/timeline/abc12345678")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["video_id"], "abc12345678")
        self.assertEqual(len(data["timeline"]), 1)
        self.assertEqual(data["timeline"][0]["timestamp"], "00:00")

    def test_chat_stream(self):
        async def fake_stream(*args, **kwargs):
            yield 'data: {"type": "sources", "sources": []}\n\n'
            yield 'data: {"type": "token", "token": "Apple"}\n\n'
            yield 'data: {"type": "done"}\n\n'

        self.mock_chat_service.stream_rag_answer = fake_stream

        response = self.client.post(
            "/api/chat/stream",
            json={"video_id": "dQw4w9WgXcQ", "prompt": "Hello", "history": []},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        body = response.text
        self.assertIn('"type": "sources"', body)
        self.assertIn('"token": "Apple"', body)

    def test_video_summary_success(self):
        self.mock_coordinator.get_video_chunks.return_value = ["chunk 1", "chunk 2"]
        self.mock_chat_service.generate_summary = AsyncMock(return_value="Summary text.")

        response = self.client.post("/api/video/summary", json={"video_id": "dQw4w9WgXcQ"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["summary"], "Summary text.")

    def test_video_tts_success(self):
        self.mock_chat_service.generate_tts = AsyncMock(return_value=b"RIFFdummywavdata")

        response = self.client.post("/api/video/tts", json={"text": "Hello world audio"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "audio/wav")
        self.assertEqual(response.content, b"RIFFdummywavdata")


if __name__ == "__main__":
    unittest.main()
