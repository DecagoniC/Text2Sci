"""Модуль для разбиения текста на чанки и предварительной обработки."""

from typing import List, Dict
import re
import pymorphy2


class TextPreprocessor:
    """Класс для очистки текста, лемматизации и разбиения на чанки."""

    def __init__(self, chunk_size: int = 300, use_lemmatization: bool = True):
        self.morph = pymorphy2.MorphAnalyzer()
        self.chunk_size = chunk_size
        self.use_lemmatization = use_lemmatization
        self._lemma_cache: Dict[str, str] = {}

    def clean_text(self, text: str, lover: bool = True, links: bool = True, cut: bool=True) -> str:
        """Очистка текста: нижний регистр, удаление ссылок и спецсимволов."""
        text = text.replace("ё", "е").replace("Ё", "Е")
        if lover:
            text = text.lower()
        if links:
            text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        if cut:
            text = re.sub(r"[^a-zа-я0-9.,!?;:\-()\s]", "", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """Разделение текста на предложения с учётом чисел и сокращений."""
        text = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.replace("<DOT>", ".").strip() for s in sentences if s.strip()]

    def lemmatize_text(self, text: str) -> str:
        """Лемматизация текста с кешированием."""
        tokens = text.split()
        lemmas: List[str] = []
        for token in tokens:
            lemma = self._lemma_cache.get(token)
            if not lemma:
                parsed = self.morph.parse(token)
                lemma = parsed[0].normal_form if parsed else token
                self._lemma_cache[token] = lemma
            lemmas.append(lemma)
        return " ".join(lemmas)

    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """Объединяет предложения в чанки фиксированной длины по словам."""
        chunks: List[str] = []
        current_chunk: List[str] = []
        word_count: int = 0

        for sent in sentences:
            tokens = sent.split()
            if word_count + len(tokens) > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                word_count = 0
            current_chunk.extend(tokens)
            word_count += len(tokens)

        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    def process_querry(self, text: str, lover: bool = True, links: bool = True, cut: bool=True):
        cleaned = self.clean_text(text, lover, links, cut)
        sentences = self.split_sentences(cleaned)
        if self.use_lemmatization:
            sentences = [self.lemmatize_text(s) for s in sentences]
        return sentences
    
    def process(self, text: str, lover: bool = True, links: bool = True, cut: bool=True) -> List[str]:
        """Полный пайплайн: очистка → (лемматизация) → разбиение на предложения → чанки."""
        cleaned = self.clean_text(text, lover, links, cut)
        sentences = self.split_sentences(cleaned)
        if self.use_lemmatization:
            sentences = [self.lemmatize_text(s) for s in sentences]
        return self.chunk_sentences(sentences)
