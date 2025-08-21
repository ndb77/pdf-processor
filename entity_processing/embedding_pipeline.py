"""embedding_pipeline.py

Utility classes for computing embeddings for biomedical entities using
SciBERT (allenai/scibert_scivocab_uncased).

The embedder is intentionally generic so it can be dropped into other
projects (including EntityProcessor) without modification.

Key features
------------
1. Lazy GPU/CPU selection – automatically uses CUDA when available.
2. Mini-batch processing for throughput (default 32).
3. Mean-pooling with optional L2 normalisation – a common practice for
   sentence-level embeddings.
4. Stateless API: `encode(texts)` for arbitrary lists and
   `encode_entity(name, type, description)` for the specific entity
   concatenation requested by the user.
5. Returns `numpy.ndarray` to play nicely with FAISS / Chroma / etc.

Example usage
-------------
>>> from embedding_pipeline import scibertEmbedder
>>> embedder = scibertEmbedder()
>>> vec = embedder.encode_entity("Aspirin", "Pharmacologic Substance", "A common non-steroidal anti-inflammatory drug.")
>>> print(vec.shape)  # (768,)

You can then store the vector in your favourite vector store or pass it
back to upstream code.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class scibertEmbedder:
    """Compute embeddings using SciBERT.

    The model is loaded with `AutoModel` so we get the base transformer
    without the masked-LM head. If you prefer to keep the head, swap to
    `AutoModelForMaskedLM`, but it is not needed for embedding.
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        batch_size: int = 32,
        device: str | None = None,
        normalize: bool = True,
    ):  # noqa: D401 – simple property
        self.batch_size = batch_size
        self.normalize = normalize
        self.model_name = model_name

        # Auto-select device if none provided
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load tokenizer + model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:  # shape (N, D)
        """Encode a list/sequence of texts into dense vectors."""
        all_vecs: list[np.ndarray] = []
        with torch.no_grad():
            for idx in range(0, len(texts), self.batch_size):
                batch_texts = texts[idx : idx + self.batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                # Forward pass – we only need the last hidden state
                outputs = self.model(**encoded)
                token_embeds = outputs.last_hidden_state  # (B, T, H)

                vecs = self._mean_pool(token_embeds, encoded["attention_mask"])
                if self.normalize:
                    vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)

                all_vecs.append(vecs.cpu().numpy())

        return np.vstack(all_vecs)

    def encode_entity(self, entity_name: str, entity_type: str, entity_description: str) -> np.ndarray:
        """Helper to build the concatenated text and embed a single entity."""
        text = (
            f"entity_name: {entity_name} "
            f"entity_type: {entity_type} "
            f"entity_description: {entity_description}"
        )
        return self.encode([text])[0]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_pool(token_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Perform mean pooling on the CLS-less token embeddings."""
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
        sum_embeds = (token_embeds * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeds / sum_mask
