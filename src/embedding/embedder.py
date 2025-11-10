from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import os

class TextEmbedder:
    def __init__(self, model_name: str = "sberbank-ai/sbert_large_nlu_ru", local_dir: str = "models/sbert_ru_large"):
        # Проверяем, существует ли локальная копия
        if not os.path.exists(local_dir):
            print(f"[+] Модель не найдена в {local_dir}. Скачиваю {model_name} с HuggingFace...")
            model = SentenceTransformer(model_name)
            os.makedirs(os.path.dirname(local_dir), exist_ok=True)
            model.save(local_dir)
            print(f"[+] Модель сохранена в {local_dir}")

        print(f"[+] Загружается модель эмбеддингов из {local_dir}")
        self.model = SentenceTransformer(local_dir)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
