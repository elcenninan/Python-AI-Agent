from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    table: str
    text: str


class RAGStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self._matrix = None

    def add(self, table: str, text: str) -> None:
        self._chunks.append(Chunk(table=table, text=text))

    def build(self) -> None:
        corpus = [chunk.text for chunk in self._chunks]
        if not corpus:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []
        q = self._vectorizer.transform([query])
        sims = cosine_similarity(q, self._matrix)[0]
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self._chunks[idx], float(score)) for idx, score in ranked if score > 0]
