import httpx
from app.core.config import Settings
from app.schemas.query import Source
from app.services.generation import GenerationService
from app.services.retrieval import RetrievedChunk


class OfflineClient:
    def chat(self, **kwargs):
        raise httpx.ConnectError("offline")


def test_generation_falls_back_to_cited_extraction() -> None:
    service = GenerationService(Settings(allow_extractive_fallback=True))
    service.client = OfflineClient()
    chunk = RetrievedChunk(
        Source(
            document="Half-Blood Prince",
            page=2710,
            chunk_id="x",
            score=0.8,
            excerpt="A Horcrux contains a fragment of a soul.",
        ),
        "A Horcrux is an object in which a person has concealed part of their soul.",
    )
    answer = service.answer("What is a Horcrux?", [chunk])
    assert "[Half-Blood Prince, p. 2710]" in answer
