from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# ожидаем, что эти классы у тебя уже есть
from preprocess.chunker import TextPreprocessor
from embedding.embedder import TextEmbedder
from retrieval.retriever import VectorRetriever, Chunk


class Seeker:
    """
    Утилитарный слой поиска (seeker).
    Работает как: raw query -> preprocess -> embed -> retriever.search -> normalized results.
    """

    def __init__(self, retriever: Optional[VectorRetriever] = None, embedder: Optional[TextEmbedder] = None, preprocessor: Optional[TextPreprocessor] = None,): 
        self.retriever = retriever or VectorRetriever.load("data\\articles.index","data\\articles_texts.pkl")
        self.embedder = embedder or TextEmbedder()
        self.preprocessor = preprocessor or TextPreprocessor(use_lemmatization=True)

    def _prepare_query(self, query: str) -> str:
        """
        Обрабатывает текст запроса:
        - лемматизация
        - разделение на предложения
        Затем объединяет обратно в одну строку для encode.
        """
        sentences = self.preprocessor.process_querry(query)
        if not sentences:
            return query.strip()
        return " ".join(sentences)

    def get_raw_answer(self, query_text: str, top_k: int = 5) -> Tuple[str, List[str]]:
        """
        Возвращает:
        - объединённый текст найденных чанков с метками
        - список уникальных путей к файлам
        """
        query_vector = self.embedder.encode(self.preprocessor.process_querry(query_text))
        chunks = self.retriever.search(query_vector, top_k=top_k)

        combined_text = []
        file_paths_set = set()

        for chunk in chunks:
            text = chunk["text"]
            file_path = chunk["file_path"]
            combined_text.append(f"[START CHUNK] {text} [END CHUNK]")
            file_paths_set.add(file_path)

        return " ".join(combined_text), list(file_paths_set)
