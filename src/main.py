from extract.text_extractor import DocumentExtractor


def main():
    de = DocumentExtractor()

    path = "test.pdf"  # можно поменять на .txt или .docx

    try:
        text = de.extract(path)
        print("=== Извлечённый текст ===")
        print(text[:1000])  # выводим первые 1000 символов
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
