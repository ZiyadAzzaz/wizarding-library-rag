import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient, models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.services.documents import extract_pdf_chunks  # noqa: E402
from app.services.embeddings import EmbeddingService  # noqa: E402


def build_index(
    pdf: Path,
    store: Path,
    collection: str,
    model_name: str,
    backend: str = "hashing",
    dimensions: int = 768,
) -> dict:
    chunks, report = extract_pdf_chunks(pdf)
    if not chunks:
        raise RuntimeError("No text chunks were extracted")
    encoder = EmbeddingService(backend, model_name, dimensions)
    vectors = encoder.encode([chunk.content for chunk in chunks], show_progress=True)
    store.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(store))
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE),
    )
    batch_size = 2048
    for start in range(0, len(chunks), batch_size):
        client.upsert(
            collection,
            points=[
                models.PointStruct(id=start + offset, vector=vector, payload=chunk.payload())
                for offset, (chunk, vector) in enumerate(
                    zip(
                        chunks[start : start + batch_size],
                        vectors[start : start + batch_size],
                        strict=True,
                    )
                )
            ],
        )
    client.close()
    config = {
        **report,
        "collection": collection,
        "embedding_backend": backend,
        "embedding_model": model_name,
        "chunk_size": 900,
        "chunk_overlap": 150,
        "vector_dimensions": int(vectors.shape[1]),
    }
    (store / "index_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the persisted Harry Potter Qdrant index")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT.parent / "harrypotter.pdf")
    parser.add_argument("--store", type=Path, default=PROJECT_ROOT / "backend/data/vector_store")
    parser.add_argument("--collection", default="harry_potter_books")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument(
        "--backend", choices=["hashing", "sentence-transformers"], default="hashing"
    )
    parser.add_argument("--dimensions", type=int, default=768)
    args = parser.parse_args()
    print(
        json.dumps(
            build_index(
                args.pdf, args.store, args.collection, args.model, args.backend, args.dimensions
            ),
            indent=2,
        )
    )
