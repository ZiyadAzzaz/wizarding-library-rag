import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router
from app.core.config import get_settings
from app.services.pipeline import RagPipeline
from app.utils.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    app.state.startup_error = None
    if settings.skip_pipeline_startup:
        yield
        return
    try:
        app.state.pipeline = RagPipeline(settings)
        logger.info("RAG pipeline loaded")
    except Exception as exc:
        app.state.startup_error = str(exc)
        logger.warning("Pipeline unavailable at startup: %s", exc)
    yield
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline and hasattr(pipeline.retriever.client, "close"):
        pipeline.retriever.client.close()


def create_app() -> FastAPI:
    application = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Grounded question answering across the seven Harry Potter books.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(router)

    @application.get("/", tags=["Meta"])
    def root() -> dict[str, str]:
        return {"name": settings.app_name, "docs": "/docs", "health": "/health"}

    return application


app = create_app()
