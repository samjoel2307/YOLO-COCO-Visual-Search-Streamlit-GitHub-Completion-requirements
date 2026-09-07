"""Utility functions for visual similarity search.

The Streamlit app currently keeps gallery images in memory so the repository
does not require a large dataset. This module is kept as a supporting file
for future extensions such as FAISS or a persistent image index.
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F


def cosine_similarity(query_vector: torch.Tensor,
                      gallery_vectors: torch.Tensor) -> torch.Tensor:
    """Return cosine similarity between one query and a gallery matrix."""
    query_vector = F.normalize(query_vector, dim=-1)
    gallery_vectors = F.normalize(gallery_vectors, dim=-1)
    return gallery_vectors @ query_vector


def rank_scores(scores: torch.Tensor) -> List[Tuple[int, float]]:
    """Return gallery indices and scores in descending order."""
    values, indices = torch.sort(scores, descending=True)
    return [(int(i), float(v)) for i, v in zip(indices, values)]
