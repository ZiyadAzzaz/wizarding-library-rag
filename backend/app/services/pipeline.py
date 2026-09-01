import re

from app.core.config import Settings
from app.schemas.query import QueryResponse
from app.services.generation import GenerationService
from app.services.retrieval import RetrievalService

CHITCHAT = re.compile(r"^(hi|hello|hey|thanks|thank you|good (morning|evening))[!. ]*$", re.I)


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.retriever = RetrievalService(settings)
        self.generator = GenerationService(settings)

    @staticmethod
    def classify(question: str) -> str:
        if CHITCHAT.match(question):
            return "chitchat"
        return "retrieve"

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        route = self.classify(question)
        if route == "chitchat":
            return QueryResponse(
                question=question,
                route=route,
                answer="Hello! Ask me anything grounded in the seven Harry Potter books.",
                sources=[],
            )
        chunks = self.retriever.search(question, top_k or self.settings.top_k)
        answer = self.generator.answer(question, chunks)
        return QueryResponse(
            question=question,
            route=route,
            answer=answer,
            sources=[chunk.source for chunk in chunks],
        )
