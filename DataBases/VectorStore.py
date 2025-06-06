import chromadb
import sys
import os
import uuid
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

# Add parent directory to sys.path for module imports
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from utils.LLM import GeminiLLM


class VectorStore:
    """
    Manages interactions with the ChromaDB vector store for storing and retrieving
    YouTube transcript chunks.
    """
    def __init__(self, api_key: str):
        """
        Initializes the VectorStore with a ChromaDB client and a collection.

        Args:
            api_key (str): API key for Gemini LLM, used for embedding function.
        """
        try:
            self.chroma_client = chromadb.PersistentClient(path="DataBases/my_chroma_db")
            self.collection = self.chroma_client.get_or_create_collection(
                "yt_transcripts", embedding_function=GeminiLLM(api_key)
            )
            logger.info("ChromaDB client and collection initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing VectorStore: {e}")
            raise

    def add_documents(self, documents: list, timestamps: list, video_id: str):
        """
        Adds a list of document chunks, their timestamps, and associated video ID
        to the ChromaDB collection. Documents are added in batches of 100.

        Args:
            documents (list): A list of text chunks (strings) to be added.
            timestamps (list): A list of corresponding timestamps for each chunk.
            video_id (str): The YouTube video ID associated with these documents.
        """
        self.documents = documents
        self.timestamps = timestamps
        self.video_id = video_id

        self.chunks_len = len(self.documents)
        logger.info(f"Adding {self.chunks_len} documents for video ID: {self.video_id}")

        # Process documents in batches to avoid overwhelming the database
        for i in range(0, self.chunks_len, 100):
            batch_documents = self.documents[i : i + 100]
            batch_timestamps = self.timestamps[i : i + 100]
            batch_size = len(batch_documents)

            try:
                self.collection.add(
                    documents=batch_documents,
                    ids=[str(uuid.uuid4()) for _ in range(batch_size)],
                    metadatas=[
                        {
                            "start": t,
                            "youtube_id": self.video_id,
                        }
                        for t in batch_timestamps
                    ],
                )
                logger.debug(f"Added batch of {batch_size} documents starting from index {i}.")
            except Exception as e:
                logger.error(f"Error adding documents batch starting at index {i}: {e}")
                raise

    def retrieve_documents(self, query: str, video_id: str, n_results: int = 5) -> list:
        """
        Retrieves relevant document chunks from the ChromaDB collection based on a query.

        Args:
            query (str): The query string to search for.
            video_id (str): The YouTube video ID to filter results by.
            n_results (int, optional): The maximum number of results to retrieve. Defaults to 5.

        Returns:
            list: A list of dictionaries, where each dictionary contains 'text' and 'start'
                  of the retrieved document chunks. Returns an empty list if no results are found.
        """
        self.query = query
        self.n_results = n_results
        self.video_id = video_id
        logger.info(f"Retrieving documents for query: '{self.query}' (video ID: {self.video_id})")

        try:
            self.results = self.collection.query(
                query_texts=self.query,
                n_results=self.n_results,
                where={"youtube_id": self.video_id},
                include=["documents", "metadatas"],
            )
            logger.debug(f"Query returned {len(self.results['documents'][0]) if self.results['documents'] else 0} results.")
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{self.query}': {e}")
            return []

        if self.results["documents"] and self.results["documents"][0]:
            # Combine retrieved texts and their corresponding start timestamps
            retrieved_data = [
                {"text": texts, "start": int(meta["start"])}
                for texts, meta in zip(
                    self.results["documents"][0],
                    self.results["metadatas"][0],
                )
            ]
            logger.debug(f"Successfully formatted {len(retrieved_data)} retrieved documents.")
            return retrieved_data
        else:
            logger.info("No documents found for the given query and video ID.")
            return []
