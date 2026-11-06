"""Person 1 module: read a CV, clean the text, and load the job dataset."""

from cv_job_matching.cv_reader import (
    CVReadError,
    read_cv,
    read_docx,
    read_pdf,
    read_txt,
)
from cv_job_matching.dataset import load_jobs, prepare_dataset
from cv_job_matching.regex_cleaning import (
    clean_text,
    remove_emails,
    remove_extra_spaces,
    remove_urls,
)

__all__ = [
    "CVReadError",
    "clean_text",
    "load_jobs",
    "prepare_dataset",
    "read_cv",
    "read_docx",
    "read_pdf",
    "read_txt",
    "remove_emails",
    "remove_extra_spaces",
    "remove_urls",
]
