"""Person 2 module: turn cleaned text into normalized tokens.

The input is text that Person 1's ``clean_text`` has already tidied (emails,
links and extra spaces removed). The steps are:

1. Tokenization      split the text into words
2. Lowercasing       "Python" and "python" become the same token
3. Stopword removal  drop common words ("the", "and", "with") and numbers
4. Stemming          cut words to a crude root   ("managed" -> "manag")
   or Lemmatization  map words to a dictionary form ("managed" -> "manage")

Lemmatization is the default because its output is still readable words,
which helps the skills module and the demo. Stemming is available with
``method="stem"``.

Technical names are protected: ``C++``, ``C#``, ``.NET``, ``ASP.NET`` and
``Node.js`` stay whole, and words such as ``pandas`` or ``kubernetes`` are not
stemmed or lemmatized.

Run the before/after examples from the project folder with::

    python -m cv_job_matching.preprocessing
"""

from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# One token is either:
#   - a word or number, optionally followed by ++ / # (C++, C#)
#     or by .something (ASP.NET, Node.js, scikit.learn)
#   - a word that starts with a dot (.NET)
TOKEN_PATTERN = r"[A-Za-z0-9]+(?:[+#]+|(?:\.[A-Za-z0-9]+)+)?|\.[A-Za-z]+"

# Words that look like plurals or verbs but are product names.
PROTECTED_WORDS = {
    "aws",
    "ios",
    "jenkins",
    "kubernetes",
    "pandas",
    "redis",
    "sas",
    "analytics",
}

# WordNet gets a few tech plurals wrong ("apis" is a bee genus in WordNet).
LEMMA_OVERRIDES = {
    "apis": "api",
    "sdks": "sdk",
    "llms": "llm",
}

METHODS = ("lemmatize", "stem", "none")

_tokenizer = RegexpTokenizer(TOKEN_PATTERN)
_stemmer = PorterStemmer()


# ---------------------------------------------------------------------------
# NLTK data
# ---------------------------------------------------------------------------

def ensure_nltk_data():
    """Download the NLTK word lists this module needs, only if missing."""
    for resource, path in (("stopwords", "corpora/stopwords"),
                           ("wordnet", "corpora/wordnet")):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


@lru_cache(maxsize=1)
def english_stopwords():
    """Return NLTK's English stopword list as a set."""
    ensure_nltk_data()
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer():
    ensure_nltk_data()
    return WordNetLemmatizer()


# ---------------------------------------------------------------------------
# The four steps
# ---------------------------------------------------------------------------

def tokenize(text):
    """Split text into tokens, keeping C++, C#, .NET and Node.js whole.

    Punctuation is dropped. Hyphenated words are split ("full-stack" ->
    "full", "stack").
    """
    _check_text(text)
    return _tokenizer.tokenize(text)


def to_lowercase(tokens):
    """Lowercase every token."""
    return [token.lower() for token in tokens]


def remove_stopwords(tokens, extra_stopwords=None):
    """Drop English stopwords and tokens that contain no letters.

    Number-only tokens such as "2025" or "3" carry no skill meaning, so they
    are removed here too. Pass ``extra_stopwords`` to drop more words.
    Tokens are expected to be lowercase already.
    """
    stop = english_stopwords()
    if extra_stopwords:
        stop = stop | {word.lower() for word in extra_stopwords}
    return [
        token
        for token in tokens
        if token not in stop and any(char.isalpha() for char in token)
    ]


def stem_tokens(tokens):
    """Reduce each token to its Porter stem ("developing" -> "develop")."""
    return [token if _is_protected(token) else _stemmer.stem(token)
            for token in tokens]


def lemmatize_tokens(tokens):
    """Reduce each token to its WordNet lemma ("databases" -> "database").

    WordNet needs to know if a word is a noun or a verb. Instead of a full
    part-of-speech tagger, each word is tried as a noun first and then as a
    verb, so "skills" -> "skill" and "managed" -> "manage".
    """
    return [_lemmatize_word(token) for token in tokens]


@lru_cache(maxsize=50_000)
def _lemmatize_word(token):
    if token in LEMMA_OVERRIDES:
        return LEMMA_OVERRIDES[token]
    if _is_protected(token):
        return token
    lemmatizer = _lemmatizer()
    noun = lemmatizer.lemmatize(token, pos="n")
    return lemmatizer.lemmatize(noun, pos="v")


def _is_protected(token):
    """Technical names are left alone: anything with symbols, or listed."""
    return token in PROTECTED_WORDS or not token.isalpha()


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def preprocess_tokens(text, method="lemmatize", extra_stopwords=None):
    """Run every step and return the final list of tokens.

    ``method`` is "lemmatize" (default), "stem", or "none".
    """
    _check_method(method)
    tokens = tokenize(text)
    tokens = to_lowercase(tokens)
    tokens = remove_stopwords(tokens, extra_stopwords)
    if method == "lemmatize":
        tokens = lemmatize_tokens(tokens)
    elif method == "stem":
        tokens = stem_tokens(tokens)
    return tokens


def preprocess(text, method="lemmatize", extra_stopwords=None):
    """Run every step and return one space-separated string.

    This is the function teammates call. The string goes straight into
    TF-IDF (Person 4) as ``processed_text``.

    >>> preprocess("Managed SQL databases and developed C# applications.")
    'manage sql database develop c# application'
    """
    return " ".join(preprocess_tokens(text, method, extra_stopwords))


def preprocessing_steps(text, extra_stopwords=None):
    """Return the text after each step, for slides and the Streamlit demo.

    The keys are, in order: ``original``, ``tokens``, ``lowercase``,
    ``no_stopwords``, ``stemmed``, ``lemmatized``.
    """
    tokens = tokenize(text)
    lowercase = to_lowercase(tokens)
    filtered = remove_stopwords(lowercase, extra_stopwords)
    return {
        "original": text,
        "tokens": tokens,
        "lowercase": lowercase,
        "no_stopwords": filtered,
        "stemmed": stem_tokens(filtered),
        "lemmatized": lemmatize_tokens(filtered),
    }


def preprocess_jobs(jobs, text_column="job_text", method="lemmatize",
                    extra_stopwords=None):
    """Add a ``processed_text`` column to a job DataFrame.

    Returns a copy; the original table is not changed. Empty or missing job
    text becomes an empty string. ``rank_jobs`` in ``matching.py`` reads the
    ``processed_text`` column first when it exists.
    """
    if text_column not in jobs.columns:
        raise KeyError(f"Column '{text_column}' was not found in the jobs table.")
    _check_method(method)
    result = jobs.copy()
    result["processed_text"] = [
        preprocess(value if isinstance(value, str) else "", method, extra_stopwords)
        for value in result[text_column]
    ]
    return result


# ---------------------------------------------------------------------------
# Input checks
# ---------------------------------------------------------------------------

def _check_text(text):
    if not isinstance(text, str):
        raise TypeError("text must be a string.")


def _check_method(method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}.")


# ---------------------------------------------------------------------------
# Before/after examples (fictional)
# ---------------------------------------------------------------------------

EXAMPLES = [
    "Developed REST APIs with Python and Flask; managed PostgreSQL databases.",
    "Built web applications using C#, ASP.NET Core and .NET for 3 clients in 2025.",
    "The candidate is responsible for analyzing data and creating dashboards.",
]


def run_examples():
    """Print each step for a few fictional CV sentences."""
    for sentence in EXAMPLES:
        steps = preprocessing_steps(sentence)
        print("\n" + "=" * 70)
        for name, value in steps.items():
            shown = value if isinstance(value, str) else " | ".join(value)
            print(f"{name:>13}: {shown}")


if __name__ == "__main__":
    run_examples()
