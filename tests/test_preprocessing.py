"""Checks for Person 2's preprocessing module."""

import pytest

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


def test_tokenize_drops_punctuation():
    assert tokenize("Python, SQL; and Excel.") == ["Python", "SQL", "and", "Excel"]


def test_tokenize_keeps_technical_names_whole():
    tokens = tokenize("C++, C#, .NET, ASP.NET and Node.js")
    assert tokens == ["C++", "C#", ".NET", "ASP.NET", "and", "Node.js"]


def test_tokenize_splits_hyphenated_words():
    assert tokenize("full-stack") == ["full", "stack"]


def test_to_lowercase():
    assert to_lowercase(["Python", "SQL", "C#"]) == ["python", "sql", "c#"]


def test_remove_stopwords_and_numbers():
    tokens = ["the", "python", "and", "sql", "2025", "3", "with", "c++"]
    assert remove_stopwords(tokens) == ["python", "sql", "c++"]


def test_remove_extra_stopwords():
    assert remove_stopwords(["python", "candidate"], extra_stopwords=["Candidate"]) == ["python"]


def test_stemming():
    assert stem_tokens(["developing", "managed", "skills"]) == ["develop", "manag", "skill"]


def test_lemmatization_nouns_and_verbs():
    assert lemmatize_tokens(["databases", "managed", "skills", "built"]) == [
        "database", "manage", "skill", "build"
    ]


def test_technical_names_are_not_stemmed_or_lemmatized():
    words = ["pandas", "kubernetes", "node.js", "c++", "asp.net"]
    assert stem_tokens(words) == words
    assert lemmatize_tokens(words) == words


def test_lemma_override_for_apis():
    assert lemmatize_tokens(["apis"]) == ["api"]


def test_preprocess_returns_string():
    result = preprocess("Managed SQL databases and developed C# applications.")
    assert result == "manage sql database develop c# application"


def test_preprocess_with_stemming():
    assert preprocess("Developing applications", method="stem") == "develop applic"


def test_preprocess_without_normalization():
    assert preprocess_tokens("Developing applications", method="none") == ["developing", "applications"]


def test_empty_and_stopword_only_text():
    assert preprocess("") == ""
    assert preprocess("   ") == ""
    assert preprocess("the and of with") == ""


def test_invalid_input():
    with pytest.raises(TypeError):
        preprocess(None)
    with pytest.raises(ValueError):
        preprocess("Python", method="magic")


def test_preprocessing_steps_shows_each_stage():
    steps = preprocessing_steps("The Analysts managed Databases")
    assert list(steps) == [
        "original", "tokens", "lowercase", "no_stopwords", "stemmed", "lemmatized"
    ]
    assert steps["tokens"] == ["The", "Analysts", "managed", "Databases"]
    assert steps["no_stopwords"] == ["analysts", "managed", "databases"]
    assert steps["stemmed"] == ["analyst", "manag", "databas"]
    assert steps["lemmatized"] == ["analyst", "manage", "database"]


def test_preprocess_jobs_adds_column_without_changing_input():
    pd = pytest.importorskip("pandas")
    jobs = pd.DataFrame({"job_text": ["Building Python APIs", None, ""]})
    result = preprocess_jobs(jobs)
    assert list(result["processed_text"]) == ["build python api", "", ""]
    assert "processed_text" not in jobs.columns


def test_preprocess_jobs_missing_column():
    pd = pytest.importorskip("pandas")
    with pytest.raises(KeyError):
        preprocess_jobs(pd.DataFrame({"other": ["x"]}))
