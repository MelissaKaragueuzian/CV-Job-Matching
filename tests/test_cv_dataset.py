"""Tests for the Hugging Face CV sample (data/cvs_sample.json)."""

import pytest

from cv_job_matching.cv_dataset import EMAIL_PATTERN, PHONE_PATTERN, cv_label, load_cvs


@pytest.fixture(scope="module")
def cvs():
    return load_cvs()


def test_columns_and_size(cvs):
    assert list(cvs.columns) == ["cv_id", "cv_text", "category"]
    assert cvs["category"].nunique() == 24
    assert (cvs.groupby("category").size() == 10).all()


def test_no_empty_or_duplicate_cvs(cvs):
    assert (cvs["cv_text"].str.strip() != "").all()
    assert not cvs["cv_text"].duplicated().any()
    assert not cvs["cv_id"].duplicated().any()


def test_no_contact_details(cvs):
    assert not cvs["cv_text"].str.contains(EMAIL_PATTERN, regex=True).any()
    assert not cvs["cv_text"].str.contains(PHONE_PATTERN, regex=True).any()


def test_label_has_id_and_title(cvs):
    row = cvs.iloc[0]
    label = cv_label(row)
    assert label.startswith(str(row["cv_id"]))
    assert len(label.split(" · ")[1].split()) <= 6
