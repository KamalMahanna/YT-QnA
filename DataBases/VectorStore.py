import logging
import os
import uuid
from typing import Any

import chromadb

from utils.LLM import GeminiLLM

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME = "yt_transcripts"


class VectorStore:
    """Manages persistent ChromaDB vector storage and semantic document retrieval."""

    def __init__(self, api_key: str, collection_name: str = DEFAULT_COLLECTION_NAME):
        # Allow overriding database storage path via env var (defaults to project's DataBases directory)
        self.db_path = os.getenv(
            "CHROMA_DB_PATH",
            os.path.join(os.getcwd(), "DataBases", "my_chroma_db"),
        )
        os.makedirs(self.db_path, exist_ok=True)

        logger.info("Connecting to ChromaDB at: %s", self.db_path)
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        self.collection_name = collection_name
        self.embedding_fn = GeminiLLM(api_key)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
        )

    def add_documents(
        self,
        documents: list[str],
        timestamps: list[float],
        video_id: str,
        batch_size: int = 100,
    ) -> None:
        """Add transcript chunks and their associated timestamps to the vector store."""
        logger.info("Adding %d documents to collection '%s' for video ID %s", len(documents), self.collection_name, video_id)

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            batch_times = timestamps[i : i + batch_size]

            self.collection.add(
                documents=batch_docs,
                ids=[str(uuid.uuid4()) for _ in batch_docs],
                metadatas=[
                    {
                        "start": t,
                        "youtube_id": video_id,
                    }
                    for t in batch_times
                ],
            )
            logger.debug("Indexed batch %d to %d of %d documents", i + 1, min(i + batch_size, len(documents)), len(documents))

        logger.info("Successfully added all %d documents for video %s", len(documents), video_id)

    def retrieve_documents(
        self,
        query: str,
        video_id: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top relevant transcript chunks for a query filtered by video ID."""
        logger.debug("Querying vector store for video %s: '%s' (top %d)", video_id, query, n_results)
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"youtube_id": video_id},
            include=["documents", "metadatas"],
        )

        matched_docs = results.get("documents", [[]])[0]
        matched_metas = results.get("metadatas", [[]])[0]

        if not matched_docs:
            logger.info("No matching documents found for video %s", video_id)
            return []

        retrieved = [
            {"text": text, "start": int(float(meta.get("start", 0)))}
            for text, meta in zip(matched_docs, matched_metas)
        ]
        logger.info("Retrieved %d relevant chunks for query", len(retrieved))
        return retrieved

    def delete_video(self, video_id: str) -> None:
        """Delete all indexed chunks for a specific YouTube video."""
        logger.info("Deleting documents for video ID: %s", video_id)
        try:
            self.collection.delete(where={"youtube_id": video_id})
            logger.info("Successfully deleted chunks for video ID: %s", video_id)
        except Exception as e:
            logger.exception("Error deleting video %s: %s", video_id, e)

    def delete_collection(self) -> None:
        """Delete the current collection and reset state."""
        logger.warning("Deleting ChromaDB collection: %s", self.collection_name)
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            logger.info("Collection '%s' deleted successfully", self.collection_name)
        except Exception as e:
            logger.exception("Error deleting collection '%s': %s", self.collection_name, e)
            raise
