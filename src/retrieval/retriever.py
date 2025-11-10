import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple


class VectorRetriever:
    def __init__(self, dim: int, m: int = 32):
        self.index = faiss.IndexHNSWFlat(dim, m)
        self.texts = []  # Сопоставляем каждому вектору исходный текст
        self.dim = dim
        self.m = m

    def add_embeddings(self, embeddings: np.ndarray, texts: List[str]):
        assert embeddings.shape[1] == self.dim, "Неверная размерность эмбеддингов!"
        self.index.add(embeddings.astype("float32"))
        self.texts.extend(texts)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        distances, indices = self.index.search(query_vector.astype("float32"), top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], dist))
        return results

    def save(self, base_name: str):
        """Сохраняет индекс и тексты в папку data/, используя базовое имя."""
        os.makedirs("data", exist_ok=True)

        index_path = os.path.join("data", f"{base_name}.index")
        texts_path = os.path.join("data", f"{base_name}_texts.pkl")

        faiss.write_index(self.index, index_path)
        with open(texts_path, "wb") as f:
            pickle.dump(self.texts, f)

        print(f"[+] Индекс сохранён: {index_path}")
        print(f"[+] Тексты сохранены: {texts_path}")

    @classmethod
    def load(cls, base_name: str):
        """Загружает индекс и тексты из папки data/ по базовому имени."""
        index_path = os.path.join("data", f"{base_name}.index")
        texts_path = os.path.join("data", f"{base_name}_texts.pkl")

        index = faiss.read_index(index_path)
        with open(texts_path, "rb") as f:
            texts = pickle.load(f)

        dim = index.d
        retriever = cls(dim=dim)
        retriever.index = index
        retriever.texts = texts

        print(f"[+] Загружен индекс '{base_name}' ({len(texts)} текстов)")
        return retriever
