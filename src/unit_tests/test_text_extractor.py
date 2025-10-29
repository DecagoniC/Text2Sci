import sys
import os
import re
import difflib
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from extract.text_extractor import DocumentExtractor

extractor = DocumentExtractor()

FILES_DIR = os.path.join(os.path.dirname(__file__), "files_for_testing")


def normalize(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


# ------------------------
# 1. Пустой файл
# ------------------------
@pytest.mark.parametrize("ext", [".txt", ".docx", ".pdf"])
def test_empty_file(ext):
    path = os.path.join(FILES_DIR, f"empty{ext}")
    
    if ext == ".txt":
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
    elif ext == ".docx":
        from docx import Document
        doc = Document()
        doc.save(path)
    elif ext == ".pdf":
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(path)
        c.save()
    
    text = extractor.extract(path)
    assert text == "", f"Файл {ext} должен вернуть пустую строку"


# ------------------------
# 2. Неподдерживаемый формат
# ------------------------
def test_unsupported_format():
    path = os.path.join(FILES_DIR, "file.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write("a,b,c")
    with pytest.raises(ValueError):
        extractor.extract(path)


# ------------------------
# 3. Несуществующий файл
# ------------------------
def test_file_not_found():
    path = os.path.join(FILES_DIR, "nonexistent_file.txt")
    with pytest.raises(FileNotFoundError):
        extractor.extract(path)


# ------------------------
# 4. Битый PDF / DOCX
# ------------------------
@pytest.mark.parametrize("ext", [".pdf", ".docx"])
def test_corrupted_file(ext):
    path = os.path.join(FILES_DIR, f"corrupted{ext}")
    with open(path, "wb") as f:
        f.write(b"\x00\x01\x02\x03\x04")
    with pytest.raises(Exception):
        extractor.extract(path)


# ------------------------
# 5. Точность текста между форматами
# ------------------------
def test_text_equivalence_across_formats():
    txt_path = os.path.join(FILES_DIR, "1984.txt")
    pdf_path = os.path.join(FILES_DIR, "1984.pdf")
    docx_path = os.path.join(FILES_DIR, "1984.docx")

    txt_norm = normalize(extractor.extract(txt_path))
    pdf_norm = normalize(extractor.extract(pdf_path))
    docx_norm = normalize(extractor.extract(docx_path))

    pdf_sim = similarity(txt_norm, pdf_norm)
    docx_sim = similarity(txt_norm, docx_norm)

    print(f"Совпадение PDF vs TXT: {pdf_sim:.2f}%")
    print(f"Совпадение DOCX vs TXT: {docx_sim:.2f}%")

    assert txt_norm == pdf_norm == docx_norm, "Тексты в разных форматах не совпадают!"


# ------------------------
# 6. Разные произведения
# ------------------------
def test_text_difference_between_books():
    # 1984
    txt_1984 = os.path.join(FILES_DIR, "1984.txt")
    pdf_1984 = os.path.join(FILES_DIR, "1984.pdf")
    docx_1984 = os.path.join(FILES_DIR, "1984.docx")

    txt1984_text = normalize(extractor.extract(txt_1984))
    pdf1984_text = normalize(extractor.extract(pdf_1984))
    docx1984_text = normalize(extractor.extract(docx_1984))

    # A Clockwork Orange
    txt_aclock = os.path.join(FILES_DIR, "AClockworkOrange.txt")
    pdf_aclock = os.path.join(FILES_DIR, "AClockworkOrange.pdf")
    docx_aclock = os.path.join(FILES_DIR, "AClockworkOrange.docx")

    txtA_text = normalize(extractor.extract(txt_aclock))
    pdfA_text = normalize(extractor.extract(pdf_aclock))
    docxA_text = normalize(extractor.extract(docx_aclock))

    # Проверяем, что тексты не совпадают между произведениями
    all_1984 = [txt1984_text, pdf1984_text, docx1984_text]
    all_aclock = [txtA_text, pdfA_text, docxA_text]

    for t1 in all_1984:
        for t2 in all_aclock:
            assert t1 != t2, "Тексты разных произведений совпадают!"

# ------------------------
# 7. Текст с картинкой (игнорируем картинку)
# ------------------------
def test_text_with_picture_ignored():
    pdf_text = normalize(extractor.extract(os.path.join(FILES_DIR, "textwithpicture.pdf")))
    docx_text = normalize(extractor.extract(os.path.join(FILES_DIR, "textwithpicture.docx")))
    txt_text = normalize(extractor.extract(os.path.join(FILES_DIR, "textwithpicturetest.txt")))

    pdf_sim = similarity(pdf_text, txt_text)
    docx_sim = similarity(docx_text, txt_text)

    print(f"Совпадение PDF с эталоном: {pdf_sim:.2f}%")
    print(f"Совпадение DOCX с эталоном: {docx_sim:.2f}%")

    assert pdf_sim > 95, f"PDF с картинкой слишком отличается ({pdf_sim:.2f}%)"
    assert docx_sim > 95, f"DOCX с картинкой слишком отличается ({docx_sim:.2f}%)"


# ------------------------
# 8. Только картинка с текстом (OCR)
# ------------------------
@pytest.mark.parametrize("ext", [".pdf", ".docx"])
def test_only_picture_file(ext):
    file_path = os.path.join(FILES_DIR, f"onlypicture{ext}")
    txt_path = os.path.join(FILES_DIR, "onlypicture.txt")

    extracted_text = normalize(extractor.extract(file_path))
    reference_text = normalize(open(txt_path, "r", encoding="utf-8").read())

    sim = similarity(extracted_text, reference_text)
    print(f"Совпадение {ext.upper()} с расшифровкой: {sim:.2f}%")

    assert sim >= 80, f"Текст из {ext} сильно отличается от расшифровки ({sim:.2f}%)"
