"""Checks that PDF, Word, and text CVs can be read."""

from pathlib import Path

import pytest

from cv_job_matching.cv_reader import (
    CVReadError,
    read_cv,
    read_docx,
    read_pdf,
    read_txt,
)

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_pdf_docx_and_txt_readers_extract_text():
    assert "MAYA HADDAD" in read_pdf(SAMPLES / "sample_cv.pdf")
    assert "MAYA SALEH" in read_docx(SAMPLES / "sample_cv.docx")
    assert "ALEX MORGAN" in read_txt(SAMPLES / "sample_cv.txt")


def test_read_cv_chooses_the_reader_from_the_extension():
    assert "Python" in read_cv(SAMPLES / "sample_cv.pdf")
    assert "Cedarline" in read_cv(SAMPLES / "sample_cv.docx")
    assert "C#" in read_cv(SAMPLES / "sample_cv.txt")


def test_empty_txt_returns_an_empty_string(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    assert read_txt(path) == ""
    assert read_cv(path) == ""


def test_unsupported_file_type_is_rejected(tmp_path):
    path = tmp_path / "cv.jpg"
    path.write_bytes(b"not a cv")
    with pytest.raises(CVReadError):
        read_cv(path)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_cv(tmp_path / "missing.pdf")


def test_damaged_pdf_and_docx_raise_a_clear_error(tmp_path):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"this is not a pdf")
    docx_path = tmp_path / "broken.docx"
    docx_path.write_bytes(b"this is not a word file")

    with pytest.raises(CVReadError):
        read_pdf(pdf_path)
    with pytest.raises(CVReadError):
        read_docx(docx_path)
