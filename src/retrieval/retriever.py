import faiss
import numpy as np
from typing import List, Tuple


class VectorRetriever:
    """
    Класс для хранения эмбеддингов и поиска ближайших по смыслу.
    Использует FAISS с индексом HNSW.
    """

    def __init__(self, dim: int, m: int = 32) -> None:
        """
        dim — размерность эмбеддингов (например, 768)
        m — параметр графа (число связей между вершинами, влияет на скорость/точность)
        """
        self.index: faiss.IndexHNSWFlat = faiss.IndexHNSWFlat(dim, m)
        self.texts: List[str] = []  # сопоставляем каждому вектору исходный текст
        self.dim: int = dim

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """
        Добавляет новые эмбеддинги и тексты в индекс.
        """
        assert embeddings.shape[1] == self.dim, "Неверная размерность эмбеддингов!"
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Ищет top_k наиболее близких фрагментов по эмбеддингу запроса.
        Возвращает список (текст, расстояние).
        """
        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(dist)))
        return results
