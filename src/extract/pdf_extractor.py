import os
from typing import Optional
import PyPDF2
from docx import Document  # для работы с .docx файлами


class DocumentExtractor:
    """
    Класс для извлечения текста из PDF, TXT и DOCX файлов.
    """

    def __init__(self):
        pass

    def extract(self, filepath: str) -> Optional[str]:
        """
        Определяет тип файла и вызывает соответствующий метод.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл '{filepath}' не найден.")

        ext = os.path.splitext(filepath)[1].lower()

        if ext == ".pdf":
            return self._extract_from_pdf(filepath)
        elif ext == ".txt":
            return self._extract_from_txt(filepath)
        elif ext == ".docx":
            return self._extract_from_docx(filepath)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {ext}")

    def _extract_from_pdf(self, filepath: str) -> str:
        """
        Извлекает текст из PDF-файла.
        """
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
                else:
                    print(f"[!] Предупреждение: страница {i+1} пуста или не читается.")
        return "\n".join(text)

    def _extract_from_txt(self, filepath: str) -> str:
        """
        Извлекает текст из обычного .txt файла.
        """
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _extract_from_docx(self, filepath: str) -> str:
        """
        Извлекает текст из .docx документа.
        """
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
