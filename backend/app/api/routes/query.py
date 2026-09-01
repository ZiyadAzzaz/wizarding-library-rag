from fastapi import APIRouter, Depends, HTTPException, Request

from app.core.config import Settings, get_settings
from app.schemas.query import HealthResponse, QueryRequest, QueryResponse
from app.services.pipeline import RagPipeline

router = APIRouter(tags=["RAG"])


def get_pipeline(request: Request) -> RagPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        detail = getattr(request.app.state, "startup_error", "RAG pipeline is unavailable")
        raise HTTPException(status_code=503, detail=detail)
    return pipeline


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    ready = getattr(request.app.state, "pipeline", None) is not None
    return HealthResponse(
        status="ok" if ready else "degraded",
        vector_store="ready" if ready else "unavailable",
        collection=settings.qdrant_collection,
        model=settings.ollama_model,
    )


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, pipeline: RagPipeline = Depends(get_pipeline)) -> QueryResponse:
    try:
        return pipeline.query(payload.question, payload.top_k)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail="The RAG service could not answer the query"
        ) from exc
