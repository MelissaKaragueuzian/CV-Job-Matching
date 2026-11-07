# CV Job Matching

A project for matching CVs with suitable job opportunities.

Each team member works on their own branch. This branch, `melissa-karakouzian`, contains Person 1's module only: a job-description dataset, a CV reader, and regex text cleaning. It does not match CVs to jobs.

## Person 1 — CV Reader and Dataset

```
CV_Job_Matching/
├── notebooks/
│   ├── prepare_dataset.ipynb
│   ├── cv_reader.ipynb
│   └── regex_cleaning.ipynb
├── cv_job_matching/
│   ├── dataset.py
│   ├── cv_reader.py
│   └── regex_cleaning.py
├── data/
│   └── jobs_clean.json
├── samples/
│   ├── sample_cv.pdf
│   ├── sample_cv.docx
│   └── sample_cv.txt
└── tests/
    ├── test_cv_reader.py
    └── test_regex_cleaning.py
```

The notebooks are the step-by-step version. The `cv_job_matching` package is the same work in files teammates can import.

### Setup

From the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.

### Job descriptions

`data/jobs_clean.json` is the original split of [its-zion-18/Jobs-tabular-dataset](https://huggingface.co/datasets/its-zion-18/Jobs-tabular-dataset). That split has 391 scraped postings. Two duplicate rows were removed, so the file has 389 jobs. The augmented (generated) split is not included. Each job is one JSON object.

`notebooks/prepare_dataset.ipynb` shows the steps: load the file, check empty values, drop duplicates, and combine the title, description, qualifications, and duties into `job_text`. The original wording is kept.

```python
from cv_job_matching import load_jobs

jobs = load_jobs()
print(jobs[["Job title", "job_text"]].head())
```

`prepare_dataset()` downloads the Hugging Face file again and rewrites the JSON file. Day-to-day code only needs `load_jobs()`.

### Reading a CV

Supported files are `.pdf`, `.docx`, and `.txt`. The samples in `samples/` are fictional and safe to commit.

```python
from cv_job_matching import CVReadError, clean_text, read_cv

try:
    text = clean_text(read_cv("samples/sample_cv.pdf"))
except FileNotFoundError:
    print("File was not found.")
except CVReadError as exc:
    print(exc)
```

`read_cv` looks at the extension and calls `read_pdf`, `read_docx`, or `read_txt`.

`clean_text` then:

- removes email addresses
- removes `http://` and `https://` links
- replaces repeated spaces, tabs, and line breaks with one space
- strips spaces at the ends

Names, skills, education, and experience stay. `C++`, `C#`, and `.NET` stay as well.

```text
My name is Melissa. Email: melissa@gmail.com Portfolio: https://example.com  Skills: Python,   SQL
```

becomes:

```text
My name is Melissa. Email: Portfolio: Skills: Python, SQL
```

An empty `.txt` file returns an empty string. A damaged PDF or Word file raises `CVReadError`. Any other extension raises `CVReadError`.

A scanned PDF has no text layer. `read_pdf` raises `CVReadError` and says that OCR is required. This module does not perform OCR.

Run the notebooks from the project folder. They look for `samples/` and `data/` in the current folder, or in the parent folder when the kernel starts inside `notebooks/`.

### Tests

```powershell
python -m pytest
```

### Short explanation for the presentation

Person 1's module is the front door of the project. It prepares a JSON file of real public job postings from the Hugging Face dataset its-zion-18/Jobs-tabular-dataset, drops duplicate rows, and stores one `job_text` field per job. It also reads a CV from PDF, Word, or plain text. A regular-expression cleaner removes email addresses and web links and turns extra spaces into one space, while skill names such as C++, C#, and .NET stay intact. If a PDF is only a scanned image, the reader stops and explains that OCR is required. Other teammates can call `read_cv`, `clean_text`, and `load_jobs`.
