import re
from typing import List
import pymorphy2

class TextPreprocessor:
    def __init__(self, chunk_size: int = 300, use_lemmatization: bool = True):
        self.morph = pymorphy2.MorphAnalyzer()
        self.chunk_size = chunk_size
        self.use_lemmatization = use_lemmatization
        self._lemma_cache = {}  # кеш лемм

    def clean_text(self, text: str) -> str:
        """
        Очистка текста:
        - Приведение к нижнему регистру
        - Удаление ссылок
        - Сохранение латиницы и цифр
        - Удаление лишних пробелов и спецсимволов
        """
        text = text.replace('ё', 'е').replace('Ё', 'Е')
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)  # убрать ссылки
        text = re.sub(r'\s+', ' ', text)  # нормализовать пробелы
        # оставить только буквы (рус+лат), цифры, базовую пунктуацию и пробелы
        text = re.sub(r"[^a-zа-я0-9.,!?;:\-()\s]", "", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        """
        Разделение на предложения с учётом дат, чисел и сокращений.
        """
        # защитим даты и десятичные числа (например, 13.02.2006 или 3.14)
        text = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', text)
        # временно заменяем точки внутри чисел на <DOT>

        # теперь разделяем по ., !, ?
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # возвращаем точки обратно
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # фильтруем пустые
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def lemmatize_text(self, text: str) -> str:
        """
        Лемматизация текста с кешированием.
        """
        tokens = text.split()
        lemmas = []
        for token in tokens:
            if token in self._lemma_cache:
                lemmas.append(self._lemma_cache[token])
                continue
            parsed = self.morph.parse(token)
            lemma = parsed[0].normal_form if parsed else token
            self._lemma_cache[token] = lemma
            lemmas.append(lemma)
        return ' '.join(lemmas)

    def chunk_sentences(self, sentences: List[str]) -> List[str]:
        """
        Объединяет предложения в чанки фиксированной длины (по словам).
        """
        chunks = []
        current_chunk = []
        word_count = 0

        for sent in sentences:
            tokens = sent.split()
            if word_count + len(tokens) > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                word_count = 0
            current_chunk.extend(tokens)
            word_count += len(tokens)

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def process(self, text: str) -> List[str]:
        """
        Полный пайплайн: очистка → (лемматизация) → разбиение на предложения → чанки.
        """
        cleaned = self.clean_text(text)
        sentences = self.split_sentences(cleaned)
        if self.use_lemmatization:
            sentences = [self.lemmatize_text(s) for s in sentences]
        chunks = self.chunk_sentences(sentences)
        return chunks
