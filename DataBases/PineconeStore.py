import logging
import os
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from utils.LLM import GeminiLLM

logger = logging.getLogger(__name__)

DEFAULT_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "yt-qna")
DEFAULT_DIMENSION = 768


class PineconeStore:
    """Manages cloud Pinecone vector storage and semantic retrieval partitioned by video_id."""

    def __init__(
        self,
        pinecone_api_key: str | None = None,
        gemini_api_key: str | None = None,
        index_name: str = DEFAULT_INDEX_NAME,
        dimension: int = DEFAULT_DIMENSION,
    ):
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.index_name = index_name
        self.dimension = dimension

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required to initialize PineconeStore.")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for generating embeddings.")

        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.llm = GeminiLLM(self.gemini_api_key, embedding_dimension=self.dimension)
        self._ensure_index()
        self.index = self.pc.Index(self.index_name)

    def _ensure_index(self) -> None:
        """Create Pinecone serverless index if it does not already exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            logger.info("Creating Pinecone serverless index '%s' (dim: %d)", self.index_name, self.dimension)
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            logger.info("Pinecone index '%s' created successfully", self.index_name)
        else:
            logger.debug("Pinecone index '%s' already exists", self.index_name)

    def video_exists(self, video_id: str) -> bool:
        """Check if transcript vectors for a given video_id exist in its namespace."""
        try:
            stats = self.index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", {}) or {}
            if video_id in namespaces:
                ns_stat = namespaces[video_id]
                count = getattr(ns_stat, "vector_count", 0) if not isinstance(ns_stat, dict) else ns_stat.get("vector_count", 0)
                return count > 0
            return False
        except Exception as e:
            logger.warning("Error checking index stats for video %s: %s", video_id, e)
            return False

    def add_documents(
        self,
        documents: list[str],
        timestamps: list[float],
        video_id: str,
        batch_size: int = 50,
    ) -> None:
        """Embed and upsert transcript chunks to Pinecone under namespace=video_id."""
        if not documents:
            logger.warning("No documents to index for video %s", video_id)
            return

        logger.info("Generating embeddings and upserting %d chunks for video %s", len(documents), video_id)
        embeddings = self.llm(documents)

        vectors = []
        for i, (doc, ts, emb) in enumerate(zip(documents, timestamps, embeddings)):
            chunk_id = f"{video_id}#chunk_{i}"
            vectors.append({
                "id": chunk_id,
                "values": emb,
                "metadata": {
                    "text": doc,
                    "start": float(ts),
                    "video_id": video_id,
                },
            })

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self.index.upsert(vectors=batch, namespace=video_id)
            logger.debug("Upserted batch %d..%d for video %s", i, min(i + batch_size, len(vectors)), video_id)

        logger.info("Successfully indexed %d chunks for video %s", len(vectors), video_id)

    def retrieve_documents(
        self,
        query: str,
        video_id: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve top relevant transcript chunks for a query from namespace=video_id."""
        query_embs = self.llm.embed_query(query)
        if not query_embs:
            return []

        res = self.index.query(
            vector=query_embs[0],
            top_k=n_results,
            namespace=video_id,
            include_metadata=True,
        )

        matches = getattr(res, "matches", []) or []
        results = []
        for m in matches:
            meta = getattr(m, "metadata", {}) or {}
            results.append({
                "text": meta.get("text", ""),
                "start": int(float(meta.get("start", 0))),
                "score": getattr(m, "score", 0.0),
            })

        logger.info("Retrieved %d relevant chunks for query on video %s", len(results), video_id)
        return results

    def get_video_chunks(self, video_id: str, limit: int = 100) -> list[str]:
        """Fetch indexed chunks for video summarization."""
        # Query with zero vector to pull chunks within this namespace
        dummy_vector = [0.0] * self.dimension
        res = self.index.query(
            vector=dummy_vector,
            top_k=limit,
            namespace=video_id,
            include_metadata=True,
        )
        matches = getattr(res, "matches", []) or []
        # Sort chunks by id or start time to preserve chronological order
        sorted_matches = sorted(
            matches,
            key=lambda m: float((getattr(m, "metadata", {}) or {}).get("start", 0)),
        )
        return [(getattr(m, "metadata", {}) or {}).get("text", "") for m in sorted_matches]

    def delete_video(self, video_id: str) -> None:
        """Delete all vectors in namespace=video_id."""
        logger.info("Deleting namespace for video %s", video_id)
        try:
            self.index.delete(delete_all=True, namespace=video_id)
        except Exception as e:
            logger.warning("Failed to delete namespace %s: %s", video_id, e)
