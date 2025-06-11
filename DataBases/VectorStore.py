import chromadb
import sys
import os
import uuid

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

from utils.LLM import GeminiLLM


class VectorStore:
    def __init__(self, api_key):
        self.chroma_client = chromadb.PersistentClient(path="DataBases/my_chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            "yt_transcripts", embedding_function=GeminiLLM(api_key)
        )

    def add_documents(self, documents, timestamps, video_id):
        self.documents = documents
        self.timestamps = timestamps
        self.video_id = video_id

        self.chunks_len = len(self.documents)
        for i in range(0, self.chunks_len, 100):

            self.collection.add(
                documents=self.documents[i : i + 100],
                ids=[
                    str(uuid.uuid4())
                    for _ in range(
                        100 if self.chunks_len >= 100 else self.chunks_len % 100
                    )
                ],
                metadatas=[
                    {
                        "start": t,
                        "youtube_id": self.video_id,
                    }
                    for t in self.timestamps[i : i + 100]
                ],
            )
            self.chunks_len -= 100

    def retrieve_documents(self, query, video_id, n_results=5):
        self.query = query
        self.n_results = n_results
        self.video_id = video_id

        self.results = self.collection.query(
            query_texts=self.query,
            n_results=self.n_results,
            where={"youtube_id": self.video_id},
            include=["documents", "metadatas"],
        )

        if self.results["documents"][0]:
            return [
                {"text": texts, "start": start}
                for texts, start in zip(
                    self.results["documents"][0],
                    [int(i["start"]) for i in self.results["metadatas"][0]],
                )
            ]
            

        else:
            return []
