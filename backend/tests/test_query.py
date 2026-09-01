import os

os.environ["SKIP_PIPELINE_STARTUP"] = "true"

from fastapi.testclient import TestClient

from app.api.routes.query import get_pipeline
from app.main import app
from app.schemas.query import QueryResponse, Source


class FakePipeline:
    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        return QueryResponse(
            question=question,
            route="retrieve",
            answer="A Horcrux contains a fragment of a soul [Half-Blood Prince, p. 2710].",
            sources=[
                Source(
                    document="Half-Blood Prince",
                    page=2710,
                    chunk_id="test-chunk",
                    score=0.91,
                    excerpt="A Horcrux was the word used for an object...",
                )
            ],
        )


def override_pipeline() -> FakePipeline:
    return FakePipeline()


app.dependency_overrides[get_pipeline] = override_pipeline


def test_query_happy_path() -> None:
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "What is a Horcrux?", "top_k": 3})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "retrieve"
    assert body["sources"][0]["page"] == 2710


def test_query_rejects_invalid_input() -> None:
    with TestClient(app) as client:
        response = client.post("/query", json={"question": "  "})
    assert response.status_code == 422
