"""Load example CVs from the Hugging Face resume dataset (added by Person 5).

Source: https://huggingface.co/datasets/opensporks/resumes
(a CC0 copy of the Kaggle "Resume Dataset", snehaanbhawal/resume-dataset).
It holds 2,484 real resumes from livecareer.com in 24 categories (HR,
ENGINEERING, CHEF, ...). Names were removed by the dataset authors and
locations replaced with "City, State".

The app does not need all 15 MB of text, so ``prepare_cv_dataset`` keeps a
small sample per category in ``data/cvs_sample.json``:

1. Read the ID, resume text and category columns (the HTML copy is skipped).
2. Drop empty and duplicate resumes.
3. Drop any resume that still contains an email address or a phone number,
   so no personal contact details reach the repository or the demo.
4. Keep ``per_category`` resumes from each category (fixed random seed, so
   everyone in the team gets the same file).
"""

from pathlib import Path

import pandas as pd

CV_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "cvs_sample.json"
HF_RESUMES = "hf://datasets/opensporks/resumes/Resume/Resume.csv"

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}"


def prepare_cv_dataset(per_category=10, seed=42, save=True):
    """Download the resumes, clean them and keep a sample per category."""
    df = pd.read_csv(HF_RESUMES, usecols=["ID", "Resume_str", "Category"])
    df = df.rename(columns={"ID": "cv_id", "Resume_str": "cv_text", "Category": "category"})

    df["cv_text"] = df["cv_text"].fillna("").astype(str).str.strip()
    df = df[df["cv_text"] != ""]
    df = df.drop_duplicates(subset="cv_text")

    has_contact = (
        df["cv_text"].str.contains(EMAIL_PATTERN, regex=True)
        | df["cv_text"].str.contains(PHONE_PATTERN, regex=True)
    )
    df = df[~has_contact]

    # Shuffle once with a fixed seed, then keep the first rows of each category.
    sample = (
        df.sample(frac=1, random_state=seed)
        .groupby("category")
        .head(per_category)
        .sort_values(["category", "cv_id"])
        .reset_index(drop=True)
    )

    if save:
        CV_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample.to_json(CV_DATA_PATH, orient="records", force_ascii=False, indent=2)
    return sample


def load_cvs(path=None):
    """Return the sampled CVs as a DataFrame: ``cv_id``, ``cv_text``, ``category``."""
    json_path = Path(path) if path else CV_DATA_PATH
    return pd.read_json(json_path)


def cv_label(row):
    """Short label for a CV in a dropdown: its ID and the first words (the job title)."""
    words = row["cv_text"].split()[:6]
    return f"{row['cv_id']} · {' '.join(words).title()}"


if __name__ == "__main__":
    cvs = prepare_cv_dataset()
    print(f"Saved {len(cvs)} CVs in {cvs['category'].nunique()} categories to {CV_DATA_PATH}")
