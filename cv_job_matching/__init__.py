"""CV job matching package.

Person 1: read a CV, clean the text, and load the job dataset.
Person 2: tokenize, lowercase, remove stopwords, stem and lemmatize.
"""

from cv_job_matching.cv_reader import (
    CVReadError,
    read_cv,
    read_docx,
    read_pdf,
    read_txt,
)
from cv_job_matching.dataset import load_jobs, prepare_dataset
from cv_job_matching.preprocessing import (
    lemmatize_tokens,
    preprocess,
    preprocess_jobs,
    preprocess_tokens,
    preprocessing_steps,
    remove_stopwords,
    stem_tokens,
    to_lowercase,
    tokenize,
)

from cv_job_matching.skills import (
    analyze_ranked_jobs,
    analyze_skills,
    detect_skills,
    find_missing_skills,
    get_ngrams,
    pos_tag_text,
    top_ngrams,
)

from cv_job_matching.regex_cleaning import (
    clean_text,
    remove_emails,
    remove_extra_spaces,
    remove_urls,
)

__all__ = [
    "CVReadError",
    "clean_text",
    "lemmatize_tokens",
    "load_jobs",
    "prepare_dataset",
    "preprocess",
    "preprocess_jobs",
    "preprocess_tokens",
    "preprocessing_steps",
    "read_cv",
    "read_docx",
    "read_pdf",
    "read_txt",
    "remove_emails",
    "remove_extra_spaces",
    "remove_stopwords",
    "remove_urls",
    "stem_tokens",
    "to_lowercase",
    "tokenize",
     "analyze_ranked_jobs",
    "analyze_skills",
    "detect_skills",
    "find_missing_skills",
    "get_ngrams",
    "pos_tag_text",
    "top_ngrams",
]
