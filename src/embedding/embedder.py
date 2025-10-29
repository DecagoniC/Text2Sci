from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class TextEmbedder:
    def __init__(self, model_name: str = "sberbank-ai/sbert_large_nlu_ru"):
        print(f"[+] Загружается модель эмбеддингов: {model_name}")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Возвращает матрицу эмбеддингов для списка текстов.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings