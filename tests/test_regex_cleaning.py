"""Checks that emails, links, and extra spaces are cleaned."""

from cv_job_matching.regex_cleaning import (
    clean_text,
    remove_emails,
    remove_extra_spaces,
    remove_urls,
)


def test_remove_emails():
    sample = "My name is Melissa. Email: melissa@gmail.com. Skills: Python, SQL."
    result = remove_emails(sample)
    assert "melissa@gmail.com" not in result
    assert "Python, SQL." in result


def test_remove_urls():
    sample = "I know Python. Portfolio: https://example.com/projects Skills: SQL."
    result = remove_urls(sample)
    assert "https://example.com/projects" not in result
    assert "Python" in result
    assert "SQL." in result


def test_remove_extra_spaces():
    assert remove_extra_spaces("Python     SQL      Machine Learning") == (
        "Python SQL Machine Learning"
    )
    assert remove_extra_spaces("  Python\t\tDeveloper \n\n SQL  ") == "Python Developer SQL"


def test_clean_text_removes_email_url_and_extra_whitespace():
    sample = """
My name is Melissa.
Email: melissa@gmail.com
Portfolio: https://example.com

Skills: Python,   SQL,    Machine Learning
Experience: 2 years
"""
    assert clean_text(sample) == (
        "My name is Melissa. Email: Portfolio: "
        "Skills: Python, SQL, Machine Learning Experience: 2 years"
    )


def test_clean_text_keeps_technical_skill_names():
    sample = (
        "Skills: C++ C# .NET ASP.NET Node.js. "
        "Email dev@example.com and see https://portfolio.dev"
    )
    cleaned = clean_text(sample)
    for skill in ("C++", "C#", ".NET", "ASP.NET", "Node.js"):
        assert skill in cleaned
    assert "@" not in cleaned
    assert "http" not in cleaned.lower()
