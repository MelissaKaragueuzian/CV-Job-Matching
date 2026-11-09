"""Integration tests for app.py: the whole pipeline runs on the fictional samples."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_with_sample(sample="sample_cv.pdf"):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio(key="source").set_value("Use a sample CV").run()
    at.selectbox(key="sample").set_value(sample).run()
    return at


def similarity_scores(at):
    """The 'Similarity' metric of each job card, as numbers."""
    return [float(m.value.rstrip("%")) for m in at.metric if m.label == "Similarity"]


def test_landing_page_without_cv():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    assert similarity_scores(at) == []
    assert any("How it works" in s.value for s in at.subheader)


@pytest.mark.parametrize("sample", ["sample_cv.docx", "sample_cv.pdf", "sample_cv.txt"])
def test_sample_cv_shows_ranked_jobs(sample):
    at = run_with_sample(sample)
    assert not at.exception
    scores = similarity_scores(at)
    assert len(scores) == 5  # default slider value
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > 0


def job_titles(at):
    """Job card headings, such as '#### 6. Data Analyst'."""
    return [m.value for m in at.markdown if m.value.startswith("#### ")]


def test_more_than_five_jobs_are_paged():
    at = run_with_sample()
    at.slider(key="top_n").set_value(12).run()
    assert not at.exception
    assert len(similarity_scores(at)) == 5            # page 1 shows 5 cards
    assert job_titles(at)[0].startswith("#### 1.")

    at.button_group(key="page_3").set_value(3).run()  # 12 jobs -> 3 pages
    assert not at.exception
    assert len(similarity_scores(at)) == 2            # jobs 11 and 12
    assert job_titles(at)[0].startswith("#### 11.")


def test_five_jobs_have_no_pager():
    at = run_with_sample()
    assert not [b for b in at.button_group if b.key and b.key.startswith("page_")]


def open_full_cv(sample):
    at = run_with_sample(sample)
    assert len([m for m in at.metric if m.label == "Words"]) == 1   # summary view
    at.button_group(key="cv_view").set_value("Full CV").run()
    assert not at.exception
    assert not [m for m in at.metric if m.label == "Words"]
    return at


def test_full_cv_pdf_is_shown_as_page_images():
    at = open_full_cv("sample_cv.pdf")
    images = at.get("image")
    assert images, "expected the PDF pages as images"


def test_full_cv_docx_is_shown_as_the_document():
    # With Word or LibreOffice installed the DOCX becomes PDF page images;
    # without them the simplified HTML version is shown instead.
    at = open_full_cv("sample_cv.docx")
    page = " ".join(h.proto.body for h in at.get("html"))
    assert at.get("image") or "<strong>MAYA SALEH</strong>" in page


def test_full_cv_txt_is_shown_on_a_page():
    at = open_full_cv("sample_cv.txt")
    page = " ".join(h.proto.body for h in at.get("html"))
    assert "pre-wrap" in page and "PROFILE" in page.upper()


def test_stemming_option_runs():
    at = run_with_sample()
    at.radio(key="method").set_value("Stemming").run()
    assert not at.exception
    assert len(similarity_scores(at)) == 5


def test_job_type_filter_limits_jobs():
    at = run_with_sample()
    at.multiselect(key="job_types").set_value(["Cybersecurity"]).run()
    assert not at.exception
    jobs_compared = next(m for m in at.metric if m.label == "Jobs compared")
    assert int(jobs_compared.value) == 25  # Cybersecurity jobs in the dataset


def test_cv_from_dataset():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.radio(key="source").set_value("CV from dataset").run()
    at.selectbox(key="cv_category").set_value("CHEF").run()
    assert not at.exception
    assert len(similarity_scores(at)) == 5
    assert any("Chef" in m.value for m in at.markdown if m.value.startswith("Category:"))


def test_high_minimum_similarity_shows_warning():
    at = run_with_sample()
    at.slider(key="min_score").set_value(50).run()
    assert not at.exception
    assert similarity_scores(at) == []
    assert any("No job passes the current filters" in w.value for w in at.warning)
