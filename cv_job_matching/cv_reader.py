"""Read the text of a CV from PDF, Word, or plain text.

Teammates can call this from the project folder::

    from cv_job_matching import clean_text, read_cv

    text = clean_text(read_cv("samples/sample_cv.pdf"))
"""

from pathlib import Path

import pymupdf
from docx import Document

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")


class CVReadError(Exception):
    """Raised when a CV cannot be read or the file type is not supported."""


def read_cv(file_path):
    """Pick the reader from the file extension and return the CV text."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"CV file not found: {path}")

    extension = path.suffix.lower()
    readers = {
        ".pdf": read_pdf,
        ".docx": read_docx,
        ".txt": read_txt,
    }
    reader = readers.get(extension)
    if reader is None:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise CVReadError(
            f"Unsupported file type ({extension or 'no extension'}). "
            f"Supported types: {supported}."
        )
    return reader(path)


def read_pdf(file_path):
    """Extract text from every page of a PDF CV.

    A scanned PDF is only a picture of the page. This function reads the
    text layer. If that layer is missing, it explains that OCR is required.
    OCR is not part of this module.
    """
    path = Path(file_path)
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise CVReadError(
            f"Could not read '{path.name}'. The PDF may be empty or damaged."
        ) from exc

    try:
        text = ""
        for page in pdf:
            text += page.get_text() + "\n"
    finally:
        pdf.close()

    if not text.strip():
        raise CVReadError(
            f"No text could be extracted from '{path.name}'. "
            "A scanned or image-only PDF needs OCR, which this module does not perform."
        )
    return text


def read_docx(file_path):
    """Extract text from the paragraphs of a Word (.docx) CV."""
    path = Path(file_path)
    try:
        document = Document(path)
    except Exception as exc:
        raise CVReadError(
            f"Could not read '{path.name}'. The Word file may be empty or damaged."
        ) from exc

    text = ""
    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_txt(file_path):
    """Read a plain-text CV. An empty file returns an empty string."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as file:
        return file.read()
