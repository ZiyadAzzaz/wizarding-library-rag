import numpy as np
from app.services.embeddings import EmbeddingService


def test_hashing_embeddings_are_stable_and_normalized() -> None:
    encoder = EmbeddingService("hashing", dimensions=256)
    first = encoder.encode(["Harry found the hidden chamber"])[0]
    second = encoder.encode(["Harry found the hidden chamber"])[0]
    assert np.array_equal(first, second)
    assert np.isclose(np.linalg.norm(first), 1.0)
