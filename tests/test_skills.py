"""Checks for Person 3's linguistic and skill-analysis module."""

import pytest

from cv_job_matching.matching import rank_jobs
from cv_job_matching.skills import (
    analyze_ranked_jobs,
    analyze_skills,
    detect_skills,
    extract_keywords_by_pos,
    find_missing_skills,
    find_missing_terms,
    generate_ngrams,
    get_ngrams,
    group_by_category,
    pos_summary,
    pos_tag_text,
    top_ngrams,
)

CV = (
    "Master student in AI. Skills: Python, SQL, pandas, scikit-learn, "
    "machine learning, Git and GitHub. Good communication and teamwork."
)
JOB = (
    "Machine Learning Engineer. Strong Python and SQL skills. Experience with "
    "TensorFlow, Docker and AWS. Knowledge of Java is a plus."
)


# POS tagging

def test_pos_tag_text_returns_token_tag_pairs():
    tags = dict(pos_tag_text("The engineer builds complex models"))
    assert tags["engineer"].startswith("NN")
    assert tags["builds"].startswith("VB")
    assert tags["complex"] == "JJ"


def test_pos_tag_keeps_technical_names_whole():
    tokens = [token for token, _ in pos_tag_text("We use C++ and .NET")]
    assert "C++" in tokens
    assert ".NET" in tokens


def test_keywords_by_pos_keep_nouns_only_and_lemmatize():
    keywords = extract_keywords_by_pos("The engineers manage databases quickly")
    assert "engineer" in keywords
    assert "database" in keywords
    assert "manage" not in keywords
    assert "the" not in keywords


def test_pos_summary_counts_groups():
    summary = pos_summary("Engineers build complex models")
    assert summary["NOUN"] >= 2
    assert summary["VERB"] >= 1


# n-grams 

def test_generate_ngrams():
    tokens = ["machine", "learning", "model"]
    assert generate_ngrams(tokens, 1) == ["machine", "learning", "model"]
    assert generate_ngrams(tokens, 2) == ["machine learning", "learning model"]
    assert generate_ngrams(tokens, 3) == ["machine learning model"]
    assert generate_ngrams(tokens, 4) == []


def test_generate_ngrams_rejects_zero():
    with pytest.raises(ValueError):
        generate_ngrams(["a"], 0)


def test_get_ngrams_drops_stopword_edges():
    assert get_ngrams("The machine learning models", 2) == [
        "machine learning", "learning models"
    ]


def test_get_ngrams_keeps_inner_stopwords():
    assert "python and sql" in get_ngrams("Python and SQL skills", 3)


def test_get_ngrams_do_not_cross_sentences():
    assert get_ngrams("Python. SQL", 2) == []
    assert get_ngrams("Python, SQL", 2) == []


def test_get_ngrams_lemmatize_option():
    assert get_ngrams("managed databases", 2, lemmatize=True) == ["manage database"]


def test_top_ngrams_counts_repeats():
    text = "project management and project management tools"
    assert top_ngrams(text, 2, 1) == [("project management", 2)]


# skill detection

def test_detect_skills_basic():
    assert detect_skills("Built REST APIs in Python and C++.") == [
        "c++", "python", "rest api"
    ]


def test_java_does_not_match_javascript():
    assert detect_skills("I know JavaScript") == ["javascript"]


def test_symbol_skills_and_dotnet():
    skills = detect_skills("C#, C++, ASP.NET and Node.js experience")
    for expected in ("c#", "c++", ".net", "node.js"):
        assert expected in skills


def test_case_does_not_matter_for_normal_skills():
    assert detect_skills("PYTHON and sql") == ["python", "sql"]


def test_ambiguous_words_need_capital_letter():
    assert "excel" not in detect_skills("We want people who excel in teams")
    assert "excel" in detect_skills("Advanced Excel and SAP")
    assert "sap" in detect_skills("Advanced Excel and SAP")


def test_detect_skills_empty_text():
    assert detect_skills("") == []


def test_detect_skills_rejects_non_string():
    with pytest.raises(TypeError):
        detect_skills(None)


def test_group_by_category():
    grouped = group_by_category(["python", "sql", "communication"])
    assert grouped["Programming"] == ["python", "sql"]
    assert grouped["Soft skills"] == ["communication"]


# missing skills

def test_find_missing_skills():
    missing = find_missing_skills(CV, JOB)
    assert missing == ["aws", "docker", "java", "tensorflow"]


def test_find_missing_terms_ignores_words_in_cv():
    cv = "I weld steel with care."
    job = "Welding of steel and aluminium. Aluminium fabrication required."
    missing = find_missing_terms(cv, job)
    assert "aluminium" in missing
    assert "steel" not in missing


# full analysis

def test_analyze_skills_structure_and_values():
    result = analyze_skills(CV, JOB)
    assert set(result) == {
        "cv_skills", "job_skills", "matched_skills", "missing_skills",
        "skill_match_pct", "missing_terms", "job_bigrams",
    }
    assert result["matched_skills"] == ["machine learning", "python", "sql"]
    assert result["skill_match_pct"] == round(100 * 3 / 7, 1)


def test_analyze_skills_job_without_skills_gives_zero_percent():
    result = analyze_skills(CV, "Looking for a friendly person.")
    assert result["skill_match_pct"] == 0.0
    assert result["missing_skills"] == []


def test_analyze_skills_empty_cv():
    result = analyze_skills("", JOB)
    assert result["matched_skills"] == []
    assert result["skill_match_pct"] == 0.0


def test_analyze_skills_rejects_non_string():
    with pytest.raises(TypeError):
        analyze_skills(CV, None)


def test_analyze_ranked_jobs_adds_skills_to_matching_output():
    jobs = [
        {"id": 1, "title": "ML Engineer", "job_text": JOB},
        {"id": 2, "title": "Designer", "job_text": "Figma and Photoshop skills."},
    ]
    ranked = rank_jobs(CV, jobs, top_n=2)
    analyzed = analyze_ranked_jobs(CV, ranked)
    assert len(analyzed) == 2
    for original, item in zip(ranked, analyzed):
        assert item["rank"] == original["rank"]
        assert "skills" in item
        assert "skills" not in original  # input not changed
    assert analyzed[0]["title"] == "ML Engineer"
    assert "docker" in analyzed[0]["skills"]["missing_skills"]
