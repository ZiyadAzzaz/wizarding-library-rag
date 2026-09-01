from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=1000, examples=["What is a Horcrux?"])
    top_k: int | None = Field(default=None, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must contain non-whitespace characters")
        return value


class Source(BaseModel):
    document: str
    page: int
    chunk_id: str
    score: float = Field(ge=-1, le=1)
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    route: str
    answer: str
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    vector_store: str
    collection: str
    model: str
