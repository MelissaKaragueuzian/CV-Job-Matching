"""Rank job descriptions against a processed CV with TF-IDF and cosine similarity.

This module deliberately does not read files or perform linguistic preprocessing.
Pass the text produced by the earlier pipeline stages instead.  A job can be a
dictionary (for example, a record from :func:`cv_job_matching.dataset.load_jobs`)
or a row in a pandas DataFrame.  The preferred text field is ``processed_text``;
``job_text`` is used for Person 1's current dataset.

Run the built-in fictional examples from the project root with::

    python -m cv_job_matching.matching
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ``processed_text`` and ``preprocessed_text`` are the integration contract for
# later pipeline stages.  The remaining names make the function useful with the
# current Person 1 dataset and common small test datasets.
TEXT_FIELDS = (
    "processed_text",
    "preprocessed_text",
    "job_text",
    "text",
    "description",
    "Description",
    "Job Description",
)
TITLE_FIELDS = ("title", "job_title", "Job title", "Job Title")
ID_FIELDS = ("id", "job_id", "Job ID", "jobId")


def rank_jobs(cv_text: str, jobs: Any, top_n: int | None = 5) -> list[dict[str, Any]]:
    """Return jobs ordered from the most to least similar to ``cv_text``.

    Parameters
    ----------
    cv_text:
        A string already cleaned/preprocessed by the preceding pipeline stages.
        No stemming, lemmatization, or custom tokenization is repeated here.
    jobs:
        An iterable of mapping-like job records, or a pandas DataFrame.  Each
        record should contain one of :data:`TEXT_FIELDS`; ``job_text`` is the
        field already provided by this repository's data set.  Missing or empty
        job text is valid and receives a score of ``0.0``.
    top_n:
        Maximum number of results.  Use ``None`` to return every job, or ``0``
        to return an empty list.

    Returns
    -------
    list[dict]
        Each dictionary contains ``rank`` (one-based position), ``title``,
        ``similarity_score``, ``job_id`` (the source ID when present, otherwise
        the source position), ``job_index``, and a copy of the original ``job``
        record.  Equal scores retain the jobs' original input order.

    Raises
    ------
    TypeError
        If the CV is not a string, jobs is not an iterable of records, or a
        present text field is not a string.
    ValueError
        If ``top_n`` is negative.
    """
    if not isinstance(cv_text, str):
        raise TypeError("cv_text must be a string of processed CV text.")
    _validate_top_n(top_n)

    records = _normalise_jobs(jobs)
    if not records or top_n == 0:
        return []

    job_texts = [_job_text(record) for record in records]
    scores = _similarity_scores(cv_text, job_texts)

    results = []
    for index, (record, score) in enumerate(zip(records, scores)):
        results.append(
            {
                "title": _first_value(record, TITLE_FIELDS, default="Untitled job"),
                "similarity_score": score,
                "job_id": _first_value(record, ID_FIELDS, default=index),
                "job_index": index,
                "job": dict(record),
            }
        )

    # Python's sort is stable, so equal-score jobs preserve source order.
    results.sort(key=lambda result: result["similarity_score"], reverse=True)
    if top_n is not None:
        results = results[:top_n]

    for position, result in enumerate(results, start=1):
        result["rank"] = position
    return results


def _validate_top_n(top_n: int | None) -> None:
    """Validate the result-limit argument."""
    if top_n is None:
        return
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an integer or None.")
    if top_n < 0:
        raise ValueError("top_n cannot be negative.")


def _normalise_jobs(jobs: Any) -> list[Mapping[str, Any]]:
    """Convert a DataFrame or iterable of mappings to records without mutation."""
    if jobs is None or isinstance(jobs, (str, bytes, Mapping)):
        raise TypeError("jobs must be an iterable of job-record dictionaries or a DataFrame.")

    # Avoid importing pandas: the current DataFrame returned by load_jobs offers
    # this public method, while users of simple lists do not need pandas here.
    if hasattr(jobs, "to_dict") and hasattr(jobs, "columns"):
        jobs = jobs.to_dict(orient="records")

    if not isinstance(jobs, Iterable):
        raise TypeError("jobs must be an iterable of job-record dictionaries or a DataFrame.")

    records = list(jobs)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"Job at index {index} must be a mapping/dictionary.")
    return records


def _job_text(record: Mapping[str, Any]) -> str:
    """Get the first supported text field, treating absent text as empty."""
    for field in TEXT_FIELDS:
        if field in record:
            value = record[field]
            if value is None:
                return ""
            if not isinstance(value, str):
                raise TypeError(f"Job field '{field}' must be a string or None.")
            return value
    return ""


def _first_value(
    record: Mapping[str, Any], fields: tuple[str, ...], default: Any
) -> Any:
    """Return a metadata field when it is present and non-empty."""
    for field in fields:
        value = record.get(field)
        if value is not None and value != "":
            return value
    return default


def _similarity_scores(cv_text: str, job_texts: list[str]) -> list[float]:
    """Fit one shared TF-IDF space and return CV-to-job cosine similarities."""
    documents = [cv_text, *job_texts]
    try:
        matrix = TfidfVectorizer().fit_transform(documents)
    except ValueError as exc:
        # Inputs such as empty strings or only one-character/default-stopword
        # tokens have no usable vocabulary.  They are valid zero-overlap cases.
        if "empty vocabulary" not in str(exc).lower():
            raise
        return [0.0] * len(job_texts)

    raw_scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    # Numerical floating-point noise should never make an interpretable cosine
    # score fall outside the conventional [0, 1] range for TF-IDF vectors.
    return [max(0.0, min(1.0, float(score))) for score in raw_scores]


def _example_jobs() -> list[dict[str, str]]:
    """Provide small fictional jobs used only by the executable examples."""
    return [
        {
            "id": "python-1",
            "title": "Python Backend Developer",
            "job_text": "Python backend developer Flask Django REST API PostgreSQL Docker cloud services",
        },
        {
            "id": "frontend-1",
            "title": "Frontend Developer",
            "job_text": "Frontend developer JavaScript TypeScript React HTML CSS responsive user interfaces",
        },
        {
            "id": "ml-1",
            "title": "Machine Learning Engineer",
            "job_text": "Machine learning engineer Python scikit learn TensorFlow data models model evaluation",
        },
        {
            "id": "design-1",
            "title": "Graphic Designer",
            "job_text": "Graphic designer Adobe Illustrator branding typography visual design",
        },
    ]


def run_examples() -> None:
    """Print four fictional ranking examples for manual demonstration."""
    examples = {
        "Python developer CV": "Python developer with Flask Django REST APIs PostgreSQL and Docker experience.",
        "Frontend developer CV": "Frontend developer building React TypeScript JavaScript HTML CSS interfaces.",
        "Machine learning CV": "Machine learning engineer using Python scikit learn TensorFlow data models.",
        "Weak-match CV": "Marine archaeology researcher studying ancient pottery and underwater excavation.",
    }
    jobs = _example_jobs()
    for label, cv_text in examples.items():
        print(f"\n{label}")
        for result in rank_jobs(cv_text, jobs, top_n=3):
            print(
                f"  {result['rank']}. {result['title']} "
                f"({result['similarity_score']:.4f})"
            )


if __name__ == "__main__":
    run_examples()
