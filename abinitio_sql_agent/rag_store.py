from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    key: str
    text: str
    payload: dict[str, str]


class RAGStore:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
        self._matrix = None

    def add(self, key: str, text: str, payload: dict[str, str]) -> None:
        self._chunks.append(Chunk(key=key, text=text, payload=payload))

    def build(self) -> None:
        corpus = [chunk.text for chunk in self._chunks]
        if not corpus:
            self._matrix = None
            return
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self._chunks:
            return []

        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._matrix)[0]
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self._chunks[index], float(score)) for index, score in ranked]
