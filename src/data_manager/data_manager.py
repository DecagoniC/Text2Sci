import os
import shutil
import pickle
from typing import List, Dict, Tuple
import numpy as np

from embedding.embedder import TextEmbedder
from retrieval.retriever import VectorRetriever, Chunk
from extract.text_extractor import DocumentExtractor
from preprocess.chunker import TextPreprocessor

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

        os.makedirs(self.raw_path, exist_ok=True)

        self.embedder = TextEmbedder()
        self.retriever = None
        self.texts: List[Chunk] = []

        test_emb = self.embedder.encode(["тест"])
        real_dim = test_emb.shape[1]

        if os.path.exists(self.index_path) and os.path.exists(self.texts_path):
            self.retriever = VectorRetriever.load(self.index_path, self.texts_path)
            self.texts = self.retriever.collector
        else:
            self.retriever = VectorRetriever(dim=real_dim)


    # ----------------- Работа с исходными файлами -----------------
    def save_file(self, filepath: str) -> str:
        fname = os.path.basename(filepath)
        dest_path = os.path.join(self.raw_path, fname)
        shutil.copy(filepath, dest_path)
        return dest_path

    # ----------------- Добавление новой статьи -----------------
    def add_article(self, filepath: str):
        file_path = self.save_file(filepath)

        doc_ext = DocumentExtractor()
        raw_text = doc_ext.extract(file_path)

        pre_raw = TextPreprocessor(use_lemmatization=False)
        raw_chunks = pre_raw.process(raw_text, links=False, lover=False,cut=False)

        pre_proc = TextPreprocessor(use_lemmatization=True)

        processed_chunks = [pre_proc.clean_text(chunk) for chunk in raw_chunks]
        processed_chunks = [pre_proc.lemmatize_text(chunk) for chunk in processed_chunks]

        embeddings = self.embedder.encode(processed_chunks)

        chunks = [Chunk(text=raw_chunk, file_path=file_path) for raw_chunk in raw_chunks]

        self.retriever.add_embeddings(embeddings, chunks)

        self.save_all()

    # ----------------- Сохранение на диск -----------------
    def save_all(self):
        """Сохраняет FAISS индекс, тексты и метаданные."""
        self.retriever.save(self.index_path, self.texts_path)

    # ----------------- Поиск -----------------
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        preprocessor=TextPreprocessor()
        query_text=preprocessor.process(query_text)
        query_vector = self.embedder.encode([query_text])

        results = self.retriever.search(query_vector, top_k=top_k)

        return results
