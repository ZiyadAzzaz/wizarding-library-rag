import pytest
from app.services.documents import book_for_page, clean_text, sliding_chunks
from app.services.retrieval import RetrievalService


def test_clean_text_and_book_mapping() -> None:
    assert clean_text("mag-\nical   text") == "magical text"
    assert book_for_page(301) == "Harry Potter and the Chamber of Secrets"
    assert book_for_page(1) is None


def test_chunk_validation() -> None:
    with pytest.raises(ValueError):
        list(sliding_chunks("text", size=10, overlap=10))


def test_chunks_preserve_overlap() -> None:
    chunks = list(sliding_chunks("one two three four five six seven", size=16, overlap=4))
    assert len(chunks) >= 2
    assert all(chunks)


def test_definition_reranking_bonus() -> None:
    definition = "The point of a Horcrux is to keep part of the soul hidden."
    unrelated = "Harry searched for the Horcrux beside the lake."
    assert RetrievalService._question_bonus("What is a Horcrux?", definition) > 0
    assert RetrievalService._question_bonus("What is a Horcrux?", unrelated) == 0
