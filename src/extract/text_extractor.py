import os
from typing import Optional
import fitz  # PyMuPDF
from docx import Document
from PIL import Image
import numpy as np
import easyocr
import io


class DocumentExtractor:
    def __init__(self, ocr_languages=("en", "ru")):
        self.ocr_reader = easyocr.Reader(list(ocr_languages), gpu=False)

    def extract(self, filepath: str) -> Optional[str]:
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
        text = []
        doc = fitz.open(filepath)
        for i, page in enumerate(doc):
            page_text = page.get_text().strip()
            if page_text:
                text.append(page_text)
            else:
                # OCR для страницы с картинкой
                pix = page.get_pixmap()
                img = np.array(
                    Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                )
                try:
                    ocr_result = self.ocr_reader.readtext(img, detail=0)
                    if ocr_result:
                        text.append(" ".join(ocr_result))
                except Exception as e:
                    print(f"[!] OCR ошибка на странице {i + 1}: {e}")
        return "\n".join(text)

    def _extract_from_txt(self, filepath: str) -> str:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    def _extract_from_docx(self, filepath: str) -> str:
        doc = Document(filepath)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # OCR для картинок внутри DOCX
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_data = rel.target_part.blob
                img = Image.open(io.BytesIO(image_data))
                img_array = np.array(img)
                try:
                    ocr_result = self.ocr_reader.readtext(img_array, detail=0)
                    if ocr_result:
                        paragraphs.append(" ".join(ocr_result))
                except Exception as e:
                    print(f"[!] OCR ошибка на картинке DOCX: {e}")

        return "\n".join(paragraphs)
