import re
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.schemas.query import Source
from app.services.embeddings import EmbeddingService


@dataclass
class RetrievedChunk:
    source: Source
    content: str


class RetrievalService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = EmbeddingService(
            settings.embedding_backend, settings.embedding_model, settings.hashing_dimensions
        )
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=20
            )
        else:
            store_path = Path(settings.qdrant_path).resolve()
            store_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(store_path))
        if not self.client.collection_exists(settings.qdrant_collection):
            raise RuntimeError(
                f"Qdrant collection '{settings.qdrant_collection}' does not exist. "
                "Run notebooks/rag_pipeline.ipynb first."
            )

    def search(self, question: str, limit: int) -> list[RetrievedChunk]:
        vector = self.model.encode([question])[0].tolist()
        candidate_limit = min(50, max(limit, limit * 10))
        points = self.client.query_points(
            collection_name=self.settings.qdrant_collection,
            query=vector,
            limit=candidate_limit,
            with_payload=True,
        ).points
        points.sort(
            key=lambda point: float(point.score)
            + self._question_bonus(question, str((point.payload or {}).get("content", ""))),
            reverse=True,
        )
        retrieved: list[RetrievedChunk] = []
        for point in points:
            payload = point.payload or {}
            content = str(payload.get("content", ""))
            bonus = self._question_bonus(question, content)
            if float(point.score) < self.settings.min_relevance_score and bonus == 0:
                continue
            retrieved.append(
                RetrievedChunk(
                    source=Source(
                        document=str(payload.get("book_name", payload.get("document", "Unknown"))),
                        page=int(payload.get("page_number", 0)),
                        chunk_id=str(payload.get("chunk_id", point.id)),
                        score=round(float(point.score), 4),
                        excerpt=content[:240].strip(),
                    ),
                    content=content,
                )
            )
            if len(retrieved) == limit:
                break
        return retrieved

    @staticmethod
    def _question_bonus(question: str, content: str) -> float:
        lowered = question.lower().strip()
        prefixes = ("what is ", "what are ", "who is ", "who was ")
        prefix = next((item for item in prefixes if lowered.startswith(item)), None)
        if prefix is None:
            return 0.0
        subject = re.sub(r"[^a-z0-9' -]", "", lowered[len(prefix) :]).strip()
        if not subject:
            return 0.0
        text = content.lower()
        bonus = 0.0
        if re.search(rf"\b{re.escape(subject)}\s+(?:is|are|was|were|means)\b", text):
            bonus += 1.0
        if f"point of a {subject} is" in text or f"point of the {subject} is" in text:
            bonus += 0.75
        return bonus
