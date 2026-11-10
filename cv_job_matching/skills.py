"""Person 3 module: linguistic analysis and skill detection.

Input text comes from the earlier stages:

* Person 1: ``clean_text(read_cv(path))``  -> cleaned CV text
* Person 1: ``load_jobs()["job_text"]``     -> job descriptions
* Person 2: ``tokenize`` / ``preprocess_tokens`` are reused here

What this module does
---------------------
1. POS tagging       ``pos_tag_text``, ``extract_keywords_by_pos``
2. Word n-grams      ``get_ngrams``, ``top_ngrams``
3. Skill detection   ``detect_skills`` (dictionary + regex)
4. Missing skills    ``find_missing_skills`` and ``find_missing_terms``
5. One call for the app: ``analyze_skills(cv_text, job_text)``
   and ``analyze_ranked_jobs(cv_text, ranked_jobs)`` for Person 4's output.

Skill detection must run on CLEANED text (Person 1), not on stemmed or
lemmatized text: stemming turns "analysis" into "analysi" and the dictionary
would no longer match. TF-IDF (Person 4) uses the preprocessed text instead.

Run the fictional examples from the project folder with::

    python -m cv_job_matching.skills
"""

import re
from collections import Counter

import nltk

from cv_job_matching.preprocessing import (
    english_stopwords,
    ensure_nltk_data,
    lemmatize_tokens,
    preprocess_tokens,
    remove_stopwords,
    to_lowercase,
    tokenize,
)

# NLTK data (the POS tagger needs its own model)

_TAGGER_RESOURCES = (
    ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
    ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
)


def ensure_tagger_data():
    """Download the POS tagger model only if it is missing.

    Newer NLTK versions use ``averaged_perceptron_tagger_eng``, older ones use
    ``averaged_perceptron_tagger``. Both are tried.
    """
    ensure_nltk_data()
    for resource, path in _TAGGER_RESOURCES:
        try:
            nltk.data.find(path)
            return
        except LookupError:
            nltk.download(resource, quiet=True)


# 1) POS tagging

NOUN_AND_ADJECTIVE_TAGS = ("NN", "NNS", "NNP", "NNPS", "JJ")

# Nouns/adjectives that appear in almost every job ad and say nothing about
# a skill. They are ignored when listing missing terms.
GENERIC_TERMS = {
    "ability", "applicant", "candidate", "company", "day", "employee",
    "environment", "experience", "good", "great", "job", "knowledge", "level",
    "location", "member", "month", "new", "number", "opportunity", "people",
    "per", "person", "position", "preferred", "related", "required", "role",
    "strong", "team", "time", "us", "work", "year", "excellent", "various",
    "other", "ensure", "include", "perform", "provide", "support", "use",
    "responsibility", "responsible", "within", "also", "must", "least",
    "plus", "minimum", "equivalent", "range", "need", "ideal", "key",
    # labels that Person 1's job_text template adds to every job
    "title", "description", "qualification", "duty",
}


def pos_tag_text(text):
    """Return ``[(token, POS tag), ...]`` for a text.

    Uses Person 2's ``tokenize`` (so ``C++`` and ``.NET`` stay whole) and the
    NLTK Penn Treebank tagger. Tags: NN noun, VB verb, JJ adjective, ...
    """
    ensure_tagger_data()
    return nltk.pos_tag(tokenize(text))


def extract_keywords_by_pos(text, keep=NOUN_AND_ADJECTIVE_TAGS):
    """Keep only nouns and adjectives, lowercased and lemmatized.

    Most skills are nouns (``python``, ``budget``) or adjectives in a noun
    phrase (``mechanical engineering``), so verbs and function words are
    dropped. Stopwords, numbers and ``GENERIC_TERMS`` are removed.
    """
    tagged = pos_tag_text(text)
    words = [token.lower() for token, tag in tagged if tag in keep]
    words = remove_stopwords(words)
    words = lemmatize_tokens(words)
    return [word for word in words if word not in GENERIC_TERMS and len(word) > 1]


def pos_summary(text):
    """Count how many words of each part of speech a text has.

    Returns a dict such as ``{"NOUN": 12, "VERB": 5, "ADJECTIVE": 4, ...}``.
    Useful for the slide.
    """
    groups = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJECTIVE", "RB": "ADVERB"}
    counts = Counter()
    for _, tag in pos_tag_text(text):
        counts[groups.get(tag[:2], "OTHER")] += 1
    return dict(counts)


# 2) Word n-grams

def generate_ngrams(tokens, n=2):
    """Slide a window of ``n`` tokens over a list.

    >>> generate_ngrams(["machine", "learning", "model"], 2)
    ['machine learning', 'learning model']
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# Sentence and clause breaks. A period only counts when whitespace or the end
# of the text follows, so ".NET", "Node.js" and "C++." are not cut.
_SEGMENT_SPLIT = re.compile(r"[;:!?,()\[\]|\n]+|\.(?:\s|$)")


def get_ngrams(text, n=2, lemmatize=False):
    """Word n-grams of a text, built inside sentences and clauses.

    Tokens come from Person 2's ``tokenize`` and are lowercased. An n-gram is
    dropped when it starts or ends with a stopword (``of the``, ``and python``)
    or with a number, because it carries no meaning. Stopwords inside a
    phrase are kept (``python and sql``). Phrases never cross a period,
    comma or line break.

    Set ``lemmatize=True`` to reduce the words to their lemma first.
    """
    stop = english_stopwords()
    grams = []
    for segment in _SEGMENT_SPLIT.split(text):
        original = to_lowercase(tokenize(segment))
        words = lemmatize_tokens(original) if lemmatize else original
        for i in range(len(original) - n + 1):
            first, last = original[i], original[i + n - 1]
            if first in stop or last in stop:
                continue
            if not (any(c.isalpha() for c in first) and any(c.isalpha() for c in last)):
                continue
            grams.append(" ".join(words[i:i + n]))
    return grams


def top_ngrams(text, n=2, top_k=10, lemmatize=False):
    """Most frequent n-grams as ``[(ngram, count), ...]``."""
    return Counter(get_ngrams(text, n, lemmatize)).most_common(top_k)


# 3) Skill detection: dictionary + regex

# category -> canonical skill -> aliases found in real CVs and job ads.
# The dataset covers many fields (engineering, sales, marketing, UX,
# cybersecurity, education...), so the dictionary is not only about software.
SKILLS = {
    "Programming": {
        "python": ["python"],
        "java": ["java"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript"],
        "c++": ["c++"],
        "c#": ["c#"],
        ".net": [".net", "asp.net", "dotnet"],
        "r": ["r programming", "rstudio"],
        "matlab": ["matlab"],
        "sql": ["sql", "mysql", "postgresql", "sqlite", "t-sql"],
        "html/css": ["html", "css", "html5", "css3"],
        "php": ["php"],
        "swift": ["Swift"],
        "bash": ["bash", "shell scripting"],
    },
    "Data & AI": {
        "machine learning": ["machine learning"],
        "deep learning": ["deep learning"],
        "nlp": ["nlp", "natural language processing"],
        "computer vision": ["computer vision"],
        "data analysis": ["data analysis", "data analytics"],
        "statistics": ["statistics", "statistical analysis"],
        "data visualization": ["data visualization", "data visualisation"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch"],
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "tableau": ["tableau"],
        "power bi": ["power bi", "powerbi"],
        "big data": ["big data", "hadoop", "Spark"],
    },
    "Web & Software": {
        "react": ["react", "reactjs", "react.js"],
        "angular": ["angular"],
        "node.js": ["node.js", "nodejs"],
        "django": ["django"],
        "flask": ["flask"],
        "rest api": ["rest api", "rest apis", "restful api", "restful apis"],
        "git": ["git", "github", "gitlab"],
        "agile": ["agile", "scrum", "kanban"],
        "ci/cd": ["ci/cd", "continuous integration"],
        "testing": ["unit testing", "test automation", "qa testing"],
    },
    "Cloud & DevOps": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp", "google cloud"],
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "linux": ["linux", "unix"],
        "networking": ["networking", "tcp/ip"],
    },
    "Cybersecurity": {
        "cybersecurity": ["cybersecurity", "cyber security", "information security"],
        "penetration testing": ["penetration testing", "pen testing"],
        "siem": ["siem", "splunk"],
        "risk assessment": ["risk assessment", "risk management"],
        "incident response": ["incident response"],
    },
    "Engineering & Design": {
        "autocad": ["autocad"],
        "solidworks": ["solidworks"],
        "cad": ["cad"],
        "revit": ["revit"],
        "ux design": ["ux design", "user experience", "ux research"],
        "figma": ["figma"],
        "adobe creative suite": ["photoshop", "illustrator", "indesign", "adobe creative"],
        "prototyping": ["prototyping", "wireframing", "wireframes"],
        "six sigma": ["six sigma", "lean manufacturing"],
    },
    "Business Tools": {
        "excel": ["Excel", "Microsoft Excel"],
        "microsoft office": ["microsoft office", "ms office", "powerpoint", "microsoft word"],
        "sap": ["SAP"],
        "salesforce": ["salesforce", "crm"],
        "quickbooks": ["quickbooks"],
        "google analytics": ["google analytics"],
    },
    "Business & Finance": {
        "accounting": ["accounting", "bookkeeping"],
        "budgeting": ["budgeting", "budget management", "financial planning"],
        "financial analysis": ["financial analysis", "financial modeling"],
        "sales": ["sales", "business development", "lead generation"],
        "marketing": ["marketing", "seo", "social media", "content marketing"],
        "customer service": ["customer service", "customer support"],
        "public relations": ["public relations", "media relations", "press releases"],
        "consulting": ["consulting", "stakeholder management"],
        "product management": ["product management", "product roadmap"],
        "project management": ["project management", "pmp"],
    },
    "Education & Writing": {
        "teaching": ["teaching", "curriculum development", "lesson planning"],
        "research": ["research"],
        "writing": ["writing", "editing", "copywriting", "journalism"],
        "grant writing": ["grant writing", "fundraising"],
    },
    "Soft skills": {
        "communication": ["communication", "communication skills"],
        "teamwork": ["teamwork", "team player", "collaboration"],
        "leadership": ["leadership", "team leadership"],
        "problem solving": ["problem solving", "problem-solving"],
        "critical thinking": ["critical thinking", "analytical skills"],
        "time management": ["time management", "organizational skills"],
        "presentation": ["presentation skills", "public speaking"],
    },
}

# Flat view: canonical skill -> (category, aliases)
SKILL_INDEX = {
    skill: (category, aliases)
    for category, skills in SKILLS.items()
    for skill, aliases in skills.items()
}

# Aliases that are ordinary English words in lowercase ("excel in a team",
# "sap" of a tree, "spark" interest, "swift" reply). They only count when
# written exactly with a capital letter, as a product name would be.
# Every alias that starts with an uppercase letter or is all caps in SKILLS
# above is matched case-sensitively; every other alias ignores case.


def _alias_regex(alias):
    """Regex for one alias.

    The look-arounds stop partial matches: ``java`` does not match inside
    ``javascript`` and ``c`` symbols (``c++``, ``c#``) are matched whole.
    """
    body = re.escape(alias)
    flags = 0 if alias != alias.lower() else re.IGNORECASE
    # a leading dot (".net") needs no word boundary before it
    start = r"(?<![\w+#])" if alias[0].isalnum() else r"(?<![\w+#.])"
    return re.compile(start + body + r"(?![\w+#])", flags)


_PATTERNS = {
    skill: [_alias_regex(alias) for alias in aliases]
    for skill, (_, aliases) in SKILL_INDEX.items()
}


def detect_skills(text):
    """Return the sorted list of dictionary skills found in ``text``.

    >>> detect_skills("Built REST APIs in Python and C++.")
    ['c++', 'python', 'rest api']
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    found = {
        skill
        for skill, patterns in _PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    }
    return sorted(found)


def group_by_category(skills):
    """Group canonical skill names by their category, for display."""
    grouped = {}
    for skill in skills:
        category = SKILL_INDEX[skill][0]
        grouped.setdefault(category, []).append(skill)
    return {category: sorted(items) for category, items in sorted(grouped.items())}


# 4) Missing skills and missing terms

def find_missing_skills(cv_text, job_text):
    """Dictionary skills the job asks for that the CV does not show."""
    return sorted(set(detect_skills(job_text)) - set(detect_skills(cv_text)))


def find_missing_terms(cv_text, job_text, top_k=10):
    """Important job words (nouns/adjectives) that never appear in the CV.

    This catches requirements that are not in the dictionary, for example
    ``welding`` or ``curriculum``. Words are compared after lemmatization, so
    ``databases`` in the CV covers ``database`` in the job. Returns the
    ``top_k`` most frequent ones in the job text.
    """
    cv_words = set(preprocess_tokens(cv_text))
    job_counts = Counter(extract_keywords_by_pos(job_text))
    missing = [(word, count) for word, count in job_counts.most_common()
               if word not in cv_words]
    return [word for word, _ in missing[:top_k]]


# 5) One call for the app

def analyze_skills(cv_text, job_text):
    """Compare one CV with one job description.

    Both inputs are cleaned text (Person 1). Returns a dict:

    ``cv_skills``        skills found in the CV
    ``job_skills``       skills found in the job description
    ``matched_skills``   in both
    ``missing_skills``   in the job but not in the CV
    ``skill_match_pct``  matched / job skills * 100 (0.0 if the job lists none)
    ``missing_terms``    other important job words not found in the CV
    ``job_bigrams``      most frequent two-word phrases of the job
    """
    for name, value in (("cv_text", cv_text), ("job_text", job_text)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string.")

    cv_skills = set(detect_skills(cv_text))
    job_skills = set(detect_skills(job_text))
    matched = cv_skills & job_skills
    missing = job_skills - cv_skills
    percent = round(100 * len(matched) / len(job_skills), 1) if job_skills else 0.0

    return {
        "cv_skills": sorted(cv_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "skill_match_pct": percent,
        "missing_terms": find_missing_terms(cv_text, job_text),
        "job_bigrams": [phrase for phrase, _ in top_ngrams(job_text, 2, 5)],
    }


def analyze_ranked_jobs(cv_text, ranked_jobs, job_text_field="job_text"):
    """Add a skill analysis to every result of ``matching.rank_jobs``.

    ``ranked_jobs`` is the list returned by Person 4's ``rank_jobs``. Each
    result is copied and gets a new key ``skills`` holding the
    ``analyze_skills`` dict, so Person 5 can show the top matches and their
    missing skills together. The job text is read from ``result["job"]``.
    """
    analyzed = []
    for result in ranked_jobs:
        job = result.get("job", {})
        job_text = job.get(job_text_field) or ""
        item = dict(result)
        item["skills"] = analyze_skills(cv_text, job_text)
        analyzed.append(item)
    return analyzed


# Fictional examples

EXAMPLE_CV = (
    "Jane Doe. Master student in AI. Skills: Python, SQL, pandas, numpy, "
    "scikit-learn, machine learning, NLP with NLTK, Git and GitHub. "
    "Good communication and teamwork. Built a dashboard in Streamlit."
)

EXAMPLE_JOB = (
    "We are looking for a Machine Learning Engineer with strong Python and "
    "SQL skills. Experience with TensorFlow or PyTorch, Docker and AWS is "
    "required. You will work on natural language processing models, data "
    "visualization and REST APIs. Knowledge of Java is a plus."
)


def run_examples():
    """Print POS tags, n-grams and skills for a fictional CV and job."""
    print("POS tags:", pos_tag_text(EXAMPLE_JOB)[:10])
    print("POS counts:", pos_summary(EXAMPLE_JOB))
    print("Top bigrams:", top_ngrams(EXAMPLE_JOB, 2, 5))
    print("Top trigrams:", top_ngrams(EXAMPLE_JOB, 3, 3))
    print()
    for key, value in analyze_skills(EXAMPLE_CV, EXAMPLE_JOB).items():
        print(f"{key:16}: {value}")


if __name__ == "__main__":
    run_examples()
