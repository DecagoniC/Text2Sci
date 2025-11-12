from typing import List
import faiss
import numpy as np
import pickle

# ------------------ Класс для хранения информации о фрагменте ------------------
class Chunk:
    def __init__(self, text: str, file_path: str, title: str = None, authors: List[str] = None):
        self.text = text
        self.file_path = file_path
        self.title = title
        self.authors = authors or []

    def to_dict(self):
        """Возвращает словарь для совместимости с сохранением/выводом."""
        return {
            "text": self.text,
            "file_path": self.file_path,
            "title": self.title,
            "authors": self.authors
        }

# ------------------ Ретривер ------------------
class VectorRetriever:
    """
    Хранение эмбеддингов и быстрый поиск с помощью FAISS (HNSW).
    Вместо словарей используется коллекция объектов Chunk.
    """
    def __init__(self, dim: int, m: int = 32):
        self.dim = dim
        self.m = m
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.collector: List[Chunk] = []

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Chunk]):
        print('\n\n', embeddings.shape[1], self.dim, '\n\n')
        assert embeddings.shape[1] == self.dim, "Неверная размерность эмбеддингов!"
        self.index.add(embeddings.astype("float32"))
        self.collector.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[dict]:
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.collector):
                chunk = self.collector[idx]
                entry = chunk.to_dict()
                entry["distance"] = dist
                results.append(entry)
        return results

    def save(self, index_path: str, collector_path: str):
        faiss.write_index(self.index, index_path)
        with open(collector_path, "wb") as f:
            pickle.dump(self.collector, f)
        print(f"[+] Индекс сохранён: {index_path}")
        print(f"[+] Collector сохранён: {collector_path}")

    @classmethod
    def load(cls, index_path: str, collector_path: str):
        index = faiss.read_index(index_path)
        with open(collector_path, "rb") as f:
            collector = pickle.load(f)

        dim = index.d
        retriever = cls(dim=dim)
        retriever.index = index
        retriever.collector = collector

        print(f"[+] Индекс загружен: {index_path}")
        print(f"[+] Collector загружен ({len(collector)} элементов)")
        return retriever
