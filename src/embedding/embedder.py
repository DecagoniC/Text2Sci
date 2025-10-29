"""Модуль для создания эмбеддингов текста с использованием SentenceTransformer."""

from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """Класс для кодирования текстов в векторные эмбеддинги."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[+] Модель эмбеддингов: {model_name}")
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dim: int | None = None

    def _load_model(self) -> None:
        """Ленивая загрузка модели."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
            self._dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Кодирует тексты в эмбеддинги."""
        self._load_model()
        assert self._model is not None  # ← mypy: теперь уверен

        if len(texts) == 0:
            dim = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dim), dtype=np.float32)

        return self._model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def get_embedding_dim(self) -> int:
        """Возвращает размерность."""
        self._load_model()
        assert self._dim is not None  # ← mypy: теперь int
        return self._dim