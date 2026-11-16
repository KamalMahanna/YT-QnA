import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from api.services import STATUS_FAILED, STATUS_PROCESSING, STATUS_READY, VideoTaskCoordinator


class TestVideoTaskCoordinator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.mock_store = MagicMock()
        self.mock_transcript = MagicMock()
        self.coordinator = VideoTaskCoordinator(
            store=self.mock_store,
            transcript_service=self.mock_transcript,
            groq_api_key="test_groq_key",
        )

    async def test_video_already_exists(self):
        self.mock_store.video_exists.return_value = True
        result = await self.coordinator.process_video("v123", force=False)
        self.assertEqual(result["status"], STATUS_READY)
        self.assertTrue(result.get("cached"))
        self.mock_transcript.with_youtube_api.assert_not_called()

    async def test_process_video_success(self):
        self.mock_store.video_exists.return_value = False
        self.mock_transcript.with_youtube_api.return_value = [
            {"text": "Intro text segment", "start": 0.0},
            {"text": "Second text segment", "start": 5.0},
        ]

        result = await self.coordinator.process_video("v123", force=False)
        self.assertEqual(result["status"], STATUS_READY)
        self.mock_store.add_documents.assert_called_once()
        chunks = self.coordinator.get_video_chunks("v123")
        self.assertTrue(len(chunks) > 0)

    async def test_concurrent_processing_lock(self):
        """Verify that a second concurrent request while first is running gets 'processing' status."""
        self.mock_store.video_exists.return_value = False

        # Simulate a slow transcript fetch
        async def slow_fetch(vid):
            await asyncio.sleep(0.1)
            return [{"text": "Hello world", "start": 0.0}]

        # Wrap in regular method that blocks inside asyncio thread or mock
        loop = asyncio.get_running_loop()

        def blocking_fetch(vid):
            import time
            time.sleep(0.05)
            return [{"text": "Hello world", "start": 0.0}]

        self.mock_transcript.with_youtube_api.side_effect = blocking_fetch

        # Launch two tasks in parallel
        t1 = asyncio.create_task(self.coordinator.process_video("v_concurrent"))
        # Tiny yield to let t1 start and acquire the lock
        await asyncio.sleep(0.01)
        t2 = asyncio.create_task(self.coordinator.process_video("v_concurrent"))

        res1, res2 = await asyncio.gather(t1, t2)

        # One should be READY, and the other should get PROCESSING
        statuses = {res1["status"], res2["status"]}
        self.assertIn(STATUS_READY, statuses)
        self.assertIn(STATUS_PROCESSING, statuses)

    async def test_process_video_failure(self):
        self.mock_store.video_exists.return_value = False
        self.mock_transcript.with_youtube_api.side_effect = Exception("Captions disabled")
        self.mock_transcript.with_whisper.side_effect = Exception("Whisper failed")

        result = await self.coordinator.process_video("v_fail")
        self.assertEqual(result["status"], STATUS_FAILED)
        self.assertIn("error", result)


if __name__ == "__main__":
    unittest.main()
