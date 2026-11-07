# CV Job Matching

A project for matching CVs with suitable job opportunities.

Each team member works on their own branch. Person 1's section below covers the job-description dataset, the CV reader, and regex text cleaning. Person 2's section covers text preprocessing.

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

## Person 2 — Text Preprocessing

```
cv_job_matching/preprocessing.py
notebooks/preprocessing.ipynb
tests/test_preprocessing.py
```

### Setup

`nltk` is now in `requirements.txt`. The first call downloads NLTK's stopword list and WordNet automatically. To do it by hand:

```python
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
```

### Using it

```python
from cv_job_matching import clean_text, load_jobs, preprocess, preprocess_jobs, read_cv

cv = preprocess(clean_text(read_cv("samples/sample_cv.pdf")))   # one string
jobs = preprocess_jobs(load_jobs())                              # adds processed_text
```

`rank_jobs` in `matching.py` already reads `processed_text` first, so Person 4 can pass these straight in.

| Function | Input | Output |
|---|---|---|
| `preprocess(text, method="lemmatize")` | cleaned text | one space-separated string |
| `preprocess_tokens(text, method="lemmatize")` | cleaned text | list of tokens |
| `preprocess_jobs(df, text_column="job_text")` | job DataFrame | copy with a `processed_text` column |
| `preprocessing_steps(text)` | text | dict with the text after each step (for slides and the Streamlit demo) |

`method` is `"lemmatize"` (default), `"stem"`, or `"none"`. The single steps (`tokenize`, `to_lowercase`, `remove_stopwords`, `stem_tokens`, `lemmatize_tokens`) can also be called on their own.

### The steps

1. **Tokenization** — an NLTK `RegexpTokenizer` splits the text into words and drops punctuation. `C++`, `C#`, `.NET`, `ASP.NET`, and `Node.js` stay whole.
2. **Lowercasing** — `Python` and `python` become the same token.
3. **Stopword removal** — NLTK's 198 English stopwords (*the, and, with, for*…) and number-only tokens such as `2025` are removed.
4. **Stemming** (Porter) or **lemmatization** (WordNet). Each word is tried as a noun, then as a verb. Technical names (anything containing a symbol, plus `pandas`, `kubernetes`, `aws`…) are not changed.

```text
before:     Built web applications using C#, ASP.NET Core and .NET for 3 clients in 2025.
stopwords:  built web applications using c# asp.net core .net clients
stemmed:    built web applic use c# asp.net core .net client
lemmatized: build web application use c# asp.net core .net client
```

On the 389 job descriptions, the average text goes from 578 words to 387 tokens. All jobs take about 2–3 seconds.

### Limits

- `IT` becomes `it`, a stopword, and is removed.
- Without a part-of-speech tagger, lemmatization sometimes picks the verb form (`meeting` → `meet`).
- Hyphenated names are split (`scikit-learn` → `scikit`, `learn`).
- Stems such as `manag` and `databas` are hard to read; this is why lemmatization is the default.

### Short explanation for the presentation

Person 2's module turns cleaned text into tokens that a computer can compare. It splits the text into words, lowercases them, removes stopwords and numbers, and reduces each word to a base form, so "managed", "managing", and "manages" all count as "manage". Stemming cuts endings by rule and is fast but gives roots like "manag"; lemmatization uses the WordNet dictionary and gives real words, so it is the default. Technical skills like C++, C#, and .NET are kept intact. The output, `processed_text`, is what the TF-IDF matching step uses.
