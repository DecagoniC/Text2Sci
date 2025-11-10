import faiss
import numpy as np
import pickle
from typing import List, Tuple

class VectorRetriever:
    """
    Хранение эмбеддингов и быстрый поиск с помощью FAISS (HNSW).
    Оригинальные тексты не меняются и хранятся отдельно.
    """

    def __init__(self, dim: int, m: int = 32):
        self.dim = dim
        self.m = m
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.texts: List[str] = []  # Оригинальные тексты, соответствующие эмбеддингам

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        """
        embeddings: np.ndarray, форма (N, dim)
        texts: список оригинальных текстовых фрагментов, длина N
        """
        assert embeddings.shape[1] == self.dim, "Неверная размерность эмбеддингов!"
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Поиск ближайших векторов.
        Возвращает список: [(оригинальный текст, расстояние), ...]
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], dist))
        return results

    def save(self, index_path: str, texts_path: str):
        """
        Сохраняет FAISS индекс и оригинальные тексты на диск.
        """
        faiss.write_index(self.index, index_path)
        with open(texts_path, "wb") as f:
            pickle.dump(self.texts, f)
        print(f"[+] Индекс сохранён: {index_path}")
        print(f"[+] Тексты сохранены: {texts_path}")

    @classmethod
    def load(cls, index_path: str, texts_path: str):
        """
        Загружает индекс и оригинальные тексты с диска.
        """
        index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)

        dim = index.d
        retriever = cls(dim=dim)
        retriever.index = index
        retriever.texts = texts

        print(f"[+] Индекс загружен: {index_path}")
        print(f"[+] Загружено {len(texts)} текстов")
        return retriever
