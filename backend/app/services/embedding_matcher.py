from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.schemas.resume_schema import ResumeBullet


@dataclass
class SemanticMatch:
    query_text: str
    bullet_id: str
    bullet_text: str
    score: float


class EmbeddingMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def cosine_similarity_matrix(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        if a.size == 0 or b.size == 0:
            return np.array([[]])
        return np.matmul(a, b.T)

    def top_k_bullet_matches(
        self,
        query_text: str,
        bullets: List[ResumeBullet],
        top_k: int = 3,
    ) -> List[SemanticMatch]:
        if not bullets:
            return []

        bullet_texts = [
            self._bullet_to_match_text(bullet)
            for bullet in bullets
        ]

        query_emb = self.encode_texts([query_text])
        bullet_embs = self.encode_texts(bullet_texts)

        sims = self.cosine_similarity_matrix(query_emb, bullet_embs)[0]
        ranked_indices = np.argsort(-sims)[:top_k]

        results: List[SemanticMatch] = []
        for idx in ranked_indices:
            results.append(
                SemanticMatch(
                    query_text=query_text,
                    bullet_id=bullets[idx].id,
                    bullet_text=bullets[idx].text,
                    score=float(sims[idx]),
                )
            )
        return results

    def best_match_score(
        self,
        query_text: str,
        bullets: List[ResumeBullet],
    ) -> float:
        matches = self.top_k_bullet_matches(query_text, bullets, top_k=1)
        if not matches:
            return 0.0
        return matches[0].score

    @staticmethod
    def _bullet_to_match_text(bullet: ResumeBullet) -> str:
        parts = [bullet.text]
        if bullet.technologies:
            parts.append("Technologies: " + ", ".join(bullet.technologies))
        if bullet.claims:
            parts.append("Claims: " + "; ".join(bullet.claims))
        return " | ".join(parts)