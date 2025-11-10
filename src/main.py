import os
import numpy as np

from extract.text_extractor import DocumentExtractor
from preprocess.text_preprocessor import TextPreprocessor
from embedding.text_embedder import TextEmbedder
from retrieval.vector_retriever import VectorRetriever


def main():
    # --- Параметры проекта ---
    data_dir = os.path.join(os.path.dirname(__file__), "../data")
    os.makedirs(data_dir, exist_ok=True)

    index_name = "articles"      # имя индексных файлов
    dim = 768                    # размерность эмбеддингов
    docs_path = os.path.join(data_dir, "docs")  # папка с исходными документами

    # --- Инициализация компонентов ---
    extractor = DocumentExtractor()
    preprocessor = TextPreprocessor(chunk_size=300)
    embedder = TextEmbedder()

    print("[1] Проверяем наличие сохранённого индекса...")
    try:
        retriever = VectorRetriever.load(index_name)
        print("[+] Индекс найден и загружен.")
    except FileNotFoundError:
        print("[!] Индекс не найден. Создаётся новый...")
        retriever = VectorRetriever(dim=dim)

        all_chunks = []

        # --- Обработка всех документов из data/docs ---
        for fname in os.listdir(docs_path):
            fpath = os.path.join(docs_path, fname)
            if not os.path.isfile(fpath):
                continue

            print(f"[2] Обработка: {fname}")
            raw_text = extractor.extract(fpath)
            chunks = preprocessor.process(raw_text)
            all_chunks.extend(chunks)

        print(f"[3] Эмбеддирование {len(all_chunks)} чанков...")
        embeddings = embedder.encode(all_chunks)

        retriever.add_embeddings(embeddings, all_chunks)
        retriever.save(index_name)
        print("[+] Индекс успешно создан и сохранён.")

    # --- Поиск по запросу пользователя ---
    while True:
        query = input("\nВведите запрос (или 'exit'): ").strip()
        if query.lower() == "exit":
            break

        query_chunks = preprocessor.process_querry(query)
        query_vector = embedder.encode(query_chunks)
        results = retriever.search(query_vector[0], top_k=5)

        print("\n🔎 Результаты поиска:")
        for i, (text, dist) in enumerate(results, 1):
            print(f"\n[{i}] dist={dist:.4f}\n{text[:500]}...")  # показываем первые 500 символов


if __name__ == "__main__":
    main()
