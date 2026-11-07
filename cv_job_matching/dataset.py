"""Load the job-description JSON prepared from Hugging Face.

The file ``data/jobs_clean.json`` comes from the original split of
https://huggingface.co/datasets/its-zion-18/Jobs-tabular-dataset
Duplicate rows were removed and a ``job_text`` field was added.
"""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "jobs_clean.json"
HF_ORIGINAL = (
    "hf://datasets/its-zion-18/Jobs-tabular-dataset/"
    "data/original-00000-of-00001.parquet"
)


def load_jobs(path=None):
    """Return the cleaned job table as a DataFrame.

    Pass ``path`` to read a different JSON file. The default is
    ``data/jobs_clean.json``.
    """
    json_path = Path(path) if path else DATA_PATH
    return pd.read_json(json_path)


def prepare_dataset(save=True):
    """Download the original Hugging Face split and clean it.

    Steps match ``notebooks/prepare_dataset.ipynb``:

    1. Load the original (scraped) split. The augmented split is not used.
    2. Drop duplicate rows.
    3. Drop rows with an empty title or description.
    4. Build ``job_text`` from the title, description, qualifications, and duties.

    The original wording of each job is kept. Nothing is invented.
    """
    df = pd.read_parquet(HF_ORIGINAL)
    df = df.drop_duplicates()

    for column in ("Job title", "Description", "qualifications", "Duties"):
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[(df["Job title"] != "") & (df["Description"] != "")]
    df = df.copy()
    df["job_text"] = (
        "Job title: " + df["Job title"] + "\n"
        + "Description: " + df["Description"] + "\n"
        + "qualifications: " + df["qualifications"] + "\n"
        + "Duties: " + df["Duties"]
    )

    if save:
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(DATA_PATH, orient="records", force_ascii=False, indent=2)
    return df
