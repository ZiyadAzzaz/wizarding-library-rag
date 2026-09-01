import hashlib
import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz

BOOK_RANGES = (
    ("Harry Potter and the Sorcerer's Stone", 12, 274),
    ("Harry Potter and the Chamber of Secrets", 282, 565),
    ("Harry Potter and the Prisoner of Azkaban", 573, 939),
    ("Harry Potter and the Goblet of Fire", 949, 1560),
    ("Harry Potter and the Order of the Phoenix", 1570, 2406),
    ("Harry Potter and the Half-Blood Prince", 2409, 2964),
    ("Harry Potter and the Deathly Hallows", 2974, 3622),
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    book_name: str
    page_number: int
    content: str

    def payload(self) -> dict[str, str | int]:
        return asdict(self)


def book_for_page(page_number: int) -> str | None:
    return next((name for name, start, end in BOOK_RANGES if start <= page_number <= end), None)


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", " ")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def sliding_chunks(text: str, size: int = 900, overlap: int = 150) -> Iterator[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > overlap >= 0")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        value = text[start:end].strip()
        if value:
            yield value
        if end >= len(text):
            break
        start = end - overlap


def extract_pdf_chunks(
    pdf_path: Path, chunk_size: int = 900, overlap: int = 150
) -> tuple[list[Chunk], dict[str, int | str]]:
    chunks: list[Chunk] = []
    empty_pages = 0
    with fitz.open(pdf_path) as document:
        page_count = len(document)
        for page_index, page in enumerate(document):
            page_number = page_index + 1
            book_name = book_for_page(page_number)
            if not book_name:
                continue
            text = clean_text(page.get_text("text"))
            if not text:
                empty_pages += 1
                continue
            for ordinal, content in enumerate(sliding_chunks(text, chunk_size, overlap)):
                raw_id = f"{pdf_path.name}:{page_number}:{ordinal}:{content[:80]}"
                chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
                chunks.append(Chunk(chunk_id, pdf_path.name, book_name, page_number, content))
    report: dict[str, int | str] = {
        "document": pdf_path.name,
        "pages": page_count,
        "chunks": len(chunks),
        "empty_content_pages": empty_pages,
        "ocr_required": "no" if chunks else "yes",
    }
    return chunks, report
