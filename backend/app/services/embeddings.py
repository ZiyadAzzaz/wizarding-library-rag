from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class EmbeddingService:
    """Stateless embeddings with an offline default and optional transformer backend."""

    def __init__(
        self,
        backend: str = "hashing",
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimensions: int = 768,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.dimensions = dimensions
        if backend == "hashing":
            self.encoder = HashingVectorizer(
                n_features=dimensions,
                alternate_sign=False,
                norm="l2",
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
            )
        elif backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name)
            self.dimensions = self.encoder.get_sentence_embedding_dimension()
        else:
            raise ValueError("embedding_backend must be 'hashing' or 'sentence-transformers'")

    def encode(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        if self.backend == "hashing":
            return self.encoder.transform(texts).toarray().astype(np.float32)
        return self.encoder.encode(
            list(texts), normalize_embeddings=True, batch_size=32, show_progress_bar=show_progress
        )
