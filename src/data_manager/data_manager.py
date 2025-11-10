import os
import shutil
import pickle
from typing import List, Dict, Tuple
import numpy as np

from embedding import TextEmbedder
from retrieval import VectorRetriever

class DatabaseManager:
    """
    Менеджер базы данных статей:
    - хранение исходных файлов
    - обработанные текстовые чанки
    - FAISS индекс для поиска
    - метаданные для связи чанков и файлов
    """
    def __init__(self, data_path: str = "data", dim: int = 768):
        self.raw_path = os.path.join(data_path, "articles_raw")
        self.texts_path = os.path.join(data_path, "articles_texts.pkl")
        self.index_path = os.path.join(data_path, "articles.index")
        self.metadata_path = os.path.join(data_path, "articles_metadata.pkl")

        os.makedirs(self.raw_path, exist_ok=True)

        self.embedder = TextEmbedder()
        self.retriever = None
        self.texts: List[str] = []
        self.metadata: List[Dict] = []

        # загружаем индекс и тексты, если они есть
        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.retriever = VectorRetriever.load(self.index_path, self.texts_path)
            self.texts = self.retriever.texts
        else:
            self.retriever = VectorRetriever(dim=dim)

        # загружаем метаданные
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "rb") as f:
                self.metadata = pickle.load(f)

    # ----------------- Работа с исходными файлами -----------------
    def save_file(self, filepath: str) -> str:
        """
        Копирует файл в articles_raw и возвращает путь к нему.
        """
        fname = os.path.basename(filepath)
        dest_path = os.path.join(self.raw_path, fname)
        shutil.copy(filepath, dest_path)
        return dest_path

    # ----------------- Добавление новой статьи -----------------
    def add_article(self, filepath: str, chunks: List[str], title: str = None, authors: List[str] = None):
        """
        Добавляет статью в базу:
        - сохраняет файл
        - добавляет тексты и эмбеддинги
        - обновляет метаданные
        """
        # 1️⃣ Сохраняем исходный файл
        file_path = self.save_file(filepath)

        # 2️⃣ Генерируем эмбеддинги для чанков
        embeddings = self.embedder.encode(chunks)

        # 3️⃣ Добавляем в FAISS и локальные тексты
        self.retriever.add_embeddings(embeddings, chunks)
        self.texts.extend(chunks)

        # 4️⃣ Обновляем метаданные
        base_index = len(self.metadata)
        for i, chunk in enumerate(chunks):
            entry = {
                "chunk_index": base_index + i,
                "file_path": file_path,
                "title": title,
                "authors": authors or []
            }
            self.metadata.append(entry)

        # 5️⃣ Сохраняем на диск
        self.save_all()

    # ----------------- Сохранение на диск -----------------
    def save_all(self):
        """Сохраняет FAISS индекс, тексты и метаданные."""
        self.retriever.save(self.index_path, self.texts_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[+] Метаданные сохранены в {self.metadata_path}")

    # ----------------- Поиск -----------------
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Поиск ближайших по смыслу фрагментов текста.
        Возвращает список словарей с текстом, расстоянием и путем к файлу.
        """
        # 1️⃣ Векторизация запроса
        query_vector = self.embedder.encode([query_text])

        # 2️⃣ Поиск через FAISS
        results = self.retriever.search(query_vector, top_k=top_k)

        # 3️⃣ Привязка к метаданным
        output = []
        for text, dist in results:
            idx = self.texts.index(text)  # находим индекс в self.texts
            meta = self.metadata[idx]     # соответствующие метаданные
            output.append({
                "text": text,
                "distance": dist,
                "file_path": meta.get("file_path"),
                "title": meta.get("title"),
                "authors": meta.get("authors")
            })
        return output
