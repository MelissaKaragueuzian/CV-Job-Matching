"""Tests for Person 4's TF-IDF and cosine-similarity ranking module."""

import pytest

from cv_job_matching.matching import rank_jobs


@pytest.fixture
def jobs():
    return [
        {
            "id": "python-1",
            "title": "Python Backend Developer",
            "job_text": "Python backend developer Flask Django REST API PostgreSQL Docker",
        },
        {
            "id": "frontend-1",
            "title": "Frontend Developer",
            "job_text": "Frontend developer JavaScript TypeScript React HTML CSS user interfaces",
        },
        {
            "id": "ml-1",
            "title": "Machine Learning Engineer",
            "job_text": "Machine learning engineer Python scikit learn TensorFlow data models",
        },
    ]


def test_python_cv_strongly_matches_python_job(jobs):
    results = rank_jobs("Python developer using Flask Django REST APIs and PostgreSQL", jobs)
    assert results[0]["title"] == "Python Backend Developer"
    assert results[0]["similarity_score"] > 0


def test_cv_can_have_multiple_relevant_matches(jobs):
    results = rank_jobs("Python machine learning engineer using scikit learn models", jobs)
    assert results[0]["title"] == "Machine Learning Engineer"
    assert results[1]["title"] == "Python Backend Developer"
    assert results[1]["similarity_score"] > 0


def test_unrelated_cv_has_zero_similarity_and_stable_ties(jobs):
    results = rank_jobs("marine archaeology pottery underwater excavation", jobs)
    assert [result["similarity_score"] for result in results] == [0.0, 0.0, 0.0]
    assert [result["job_id"] for result in results] == ["python-1", "frontend-1", "ml-1"]


def test_empty_cv_returns_zero_scores(jobs):
    results = rank_jobs("", jobs)
    assert len(results) == 3
    assert all(result["similarity_score"] == 0.0 for result in results)


def test_empty_job_description_is_returned_with_zero_score():
    results = rank_jobs("Python developer", [{"id": 1, "title": "Empty", "job_text": ""}])
    assert results[0]["title"] == "Empty"
    assert results[0]["similarity_score"] == 0.0


def test_empty_jobs_collection_returns_empty_list():
    assert rank_jobs("Python developer", []) == []


@pytest.mark.parametrize(
    ("cv_text", "jobs", "error"),
    [
        (None, [], TypeError),
        ("Python", "not job records", TypeError),
        ("Python", [{"job_text": 42}], TypeError),
        ("Python", ["not a job record"], TypeError),
    ],
)
def test_invalid_input_is_reported(cv_text, jobs, error):
    with pytest.raises(error):
        rank_jobs(cv_text, jobs)


def test_missing_job_text_is_handled_as_empty():
    results = rank_jobs("Python developer", [{"id": "missing", "title": "No Description"}])
    assert results[0]["similarity_score"] == 0.0


def test_top_n_limits_results_and_none_returns_all(jobs):
    assert len(rank_jobs("Python developer", jobs, top_n=2)) == 2
    assert len(rank_jobs("Python developer", jobs, top_n=None)) == 3
    assert rank_jobs("Python developer", jobs, top_n=0) == []


def test_scores_are_descending_valid_and_include_original_information(jobs):
    results = rank_jobs("Python developer Flask machine learning", jobs)
    scores = [result["similarity_score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert [result["rank"] for result in results] == [1, 2, 3]
    assert results[0]["job"] == jobs[results[0]["job_index"]]


def test_dataframe_input_uses_person_one_job_text_column():
    pandas = pytest.importorskip("pandas")
    jobs = pandas.DataFrame(
        [{"Job title": "Data Analyst", "job_text": "SQL Python data analysis dashboards"}]
    )
    results = rank_jobs("Python SQL data analyst", jobs)
    assert results[0]["title"] == "Data Analyst"
    assert results[0]["similarity_score"] > 0
