import sys
import os
import re
import difflib

import numpy as np
import pytest

# Добавляем корень проекта
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.embedding.embedder import TextEmbedder
from src.extract.text_extractor import DocumentExtractor


# Глобальный эмбеддер (ленивая загрузка)
embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")

# Экстрактор текстов
extractor = DocumentExtractor()

# Папка с тестовыми файлами
FILES_DIR = os.path.join(os.path.dirname(__file__), "files_for_testing")


def normalize(text: str) -> str:
    """Нормализация текста."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    """Процент схожести строк."""
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


# ------------------------
# 1. Пустой список
# ------------------------
def test_encode_empty_list():
    embeddings = embedder.encode([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, embedder.get_embedding_dim())


# ------------------------
# 2. Один текст
# ------------------------
def test_encode_single_text():
    texts = ["Привет, мир!"]
    embeddings = embedder.encode(texts)
    assert embeddings.shape == (1, embedder.get_embedding_dim())
    assert np.linalg.norm(embeddings[0]) > 0


# ------------------------
# 3. Несколько текстов
# ------------------------
def test_encode_multiple_texts():
    texts = ["Кот", "Собака", "Птица"]
    embeddings = embedder.encode(texts)
    assert embeddings.shape == (3, embedder.get_embedding_dim())
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0)


# ------------------------
# 4. Идентичные тексты
# ------------------------
def test_identical_texts_same_embeddings():
    texts = ["Тест", "Тест"]
    embeddings = embedder.encode(texts)
    assert np.allclose(embeddings[0], embeddings[1])


# ------------------------
# 5. Похожие тексты
# ------------------------
def test_similar_texts_high_cosine():
    texts = [
        "Москва — столица России.",
        "Москва является столицей Российской Федерации.",
        "Париж — столица Франции.",
    ]
    embeddings = embedder.encode(texts)

    cos_sim_01 = np.dot(embeddings[0], embeddings[1])
    cos_sim_02 = np.dot(embeddings[0], embeddings[2])

    print(f"Косинус 0-1: {cos_sim_01:.3f}")
    print(f"Косинус 0-2: {cos_sim_02:.3f}")

    assert cos_sim_01 > 0.80
    assert cos_sim_02 < 0.70


# ------------------------
# 6. Эмбеддинги из файлов
# ------------------------
@pytest.mark.slow
def test_embeddings_from_files():
    file_paths = [
        "1984.txt",
        "1984.pdf",
        "1984.docx",
        "AClockworkOrange.txt",
        "AClockworkOrange.pdf",
        "AClockworkOrange.docx",
    ]

    texts = []
    for fname in file_paths:
        path = os.path.join(FILES_DIR, fname)
        raw = extractor.extract(path)
        texts.append(normalize(raw))

    embeddings = embedder.encode(texts)
    assert embeddings.shape == (6, embedder.get_embedding_dim())

    # 1984 — близки
    sim_1984 = np.dot(embeddings[0], embeddings[1])
    assert sim_1984 > 0.95

    # 1984 vs AClockworkOrange — далеки
    sim_cross = np.dot(embeddings[0], embeddings[3])
    assert sim_cross < 0.60


# ------------------------
# 7. Размерность
# ------------------------
def test_embedding_dimension():
    dim = embedder.get_embedding_dim()
    assert dim > 0
    assert embedder.encode(["test"]).shape[1] == dim


# ------------------------
# 8. Нормализация
# ------------------------
def test_embeddings_are_normalized():
    embeddings = embedder.encode(["нормализация", "векторов"])
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


# ------------------------
# 9. Прогресс-бар
# ------------------------
def test_progress_bar_does_not_crash():
    try:
        embedder.encode(["a"] * 10)
    except Exception as e:
        pytest.fail(f"Прогресс-бар упал: {e}")


# ------------------------
# 10. Повторное использование
# ------------------------
def test_model_reuse():
    e1 = TextEmbedder("all-MiniLM-L6-v2")
    e2 = TextEmbedder("all-MiniLM-L6-v2")
    assert e1.encode(["test"]).shape == (1, 384)
    assert e2.encode(["test"]).shape == (1, 384)
