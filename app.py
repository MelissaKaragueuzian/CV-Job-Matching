"""Person 5 - Interface & Integration: Streamlit app that runs the whole pipeline.

    1. Extract CV   Person 1   read_cv + clean_text
    2. Preprocess   Person 2   preprocess / preprocess_jobs
    3. Find skills  Person 3   detect_skills + analyze_ranked_jobs
    4. Rank jobs    Person 4   rank_jobs (TF-IDF + cosine similarity)
    5. Show results Person 5   this file

Run from the project folder:

    streamlit run app.py
"""

from __future__ import annotations

import html
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import altair as alt
import mammoth
import pandas as pd
import pymupdf
import streamlit as st
from huggingface_hub import hf_hub_download

from cv_job_matching import CVReadError, clean_text, load_jobs, read_cv
from cv_job_matching.cv_dataset import cv_label, load_cvs
from cv_job_matching.matching import rank_jobs
from cv_job_matching.preprocessing import preprocess, preprocess_jobs, preprocessing_steps
from cv_job_matching.skills import (
    analyze_ranked_jobs,
    detect_skills,
    group_by_category,
    pos_summary,
    top_ngrams,
)

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = PROJECT_DIR / "samples"
MAX_UPLOAD_MB = 5
PAGE_SIZE = 5  # job cards per page
HF_CV_REPO = "opensporks/resumes"
METHOD_LABELS = {"Lemmatization": "lemmatize", "Stemming": "stem"}
CHART_BLUE = "#1f5f99"  # same as primaryColor in .streamlit/config.toml

PIPELINE = [
    (":material/description:", "Extract", "Read PDF / DOCX / TXT, remove emails, links, extra spaces"),
    (":material/text_fields:", "Preprocess", "Tokens, lowercase, stopwords, stemming / lemmatization"),
    (":material/psychology:", "Find skills", "POS tags, n-grams, skill dictionary, missing skills"),
    (":material/leaderboard:", "Rank jobs", "TF-IDF vectors and cosine similarity"),
    (":material/dashboard:", "Show results", "This Streamlit interface"),
]


# Helpers
@st.cache_data(show_spinner="Preprocessing the job descriptions...")
def load_processed_jobs(method: str) -> pd.DataFrame:
    """Load the jobs and add Person 2's processed_text column.

    Streamlit re-runs this whole file on every click. The cache keeps the
    result, so the jobs are preprocessed once per method, not every time.
    """
    return preprocess_jobs(load_jobs(), method=method)


@st.cache_data
def load_cv_dataset() -> pd.DataFrame:
    """240 anonymised CVs (10 per category) from the Hugging Face resume dataset."""
    return load_cvs()


@st.cache_data(show_spinner="Fetching the original CV...")
def dataset_cv_pdf(category: str, cv_id: str) -> bytes | None:
    """Download the original PDF of a dataset CV (kept in the Hugging Face cache).

    Returns None when offline, so the app falls back to showing the text.
    """
    try:
        path = hf_hub_download(HF_CV_REPO, f"data/data/{category}/{cv_id}.pdf", repo_type="dataset")
        return Path(path).read_bytes()
    except Exception:
        return None


def _docx_to_pdf_with_word(docx_path: Path, pdf_path: Path) -> None:
    """Ask Microsoft Word (Windows) to save the DOCX as PDF.

    A separate, invisible Word instance is used so the user's open Word
    windows are not touched. Macros and alerts are switched off.
    """
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()  # Streamlit runs the script in a worker thread
    word = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        word.AutomationSecurity = 3  # msoAutomationSecurityForceDisable: no macros
        doc = word.Documents.Open(str(docx_path), ReadOnly=True, AddToRecentFiles=False,
                                  ConfirmConversions=False)
        doc.SaveAs(str(pdf_path), FileFormat=17)  # 17 = wdFormatPDF
        doc.Close(False)
    finally:
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


def _docx_to_pdf_with_libreoffice(docx_path: Path, out_dir: Path) -> None:
    """Convert with LibreOffice, if installed (works on Windows, macOS and Linux)."""
    soffice = shutil.which("soffice") or next(
        (p for p in (r"C:\Program Files\LibreOffice\program\soffice.exe",
                     "/Applications/LibreOffice.app/Contents/MacOS/soffice") if Path(p).exists()),
        None,
    )
    if soffice is None:
        raise FileNotFoundError("LibreOffice is not installed.")
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir),
                    str(docx_path)], check=True, timeout=60, capture_output=True)


def docx_to_pdf(docx_bytes: bytes) -> bytes | None:
    """Cached DOCX -> PDF. A failed conversion is not cached, so it is retried."""
    try:
        return _docx_to_pdf_cached(docx_bytes)
    except RuntimeError:
        return None


@st.cache_data(show_spinner="Opening the Word document...")
def _docx_to_pdf_cached(docx_bytes: bytes) -> bytes:
    """Turn a DOCX into a PDF so it can be shown exactly as it looks in Word.

    Tries Word first (Windows), then LibreOffice. Returns None if neither is
    available; the app then shows a simplified HTML version instead.
    The temporary files are deleted straight away.
    """
    with tempfile.TemporaryDirectory() as folder:
        docx_path = Path(folder) / "cv.docx"
        pdf_path = Path(folder) / "cv.pdf"
        docx_path.write_bytes(docx_bytes)
        converters = [lambda: _docx_to_pdf_with_libreoffice(docx_path, Path(folder))]
        if sys.platform == "win32":
            converters.insert(0, lambda: _docx_to_pdf_with_word(docx_path, pdf_path))
        for convert in converters:
            try:
                convert()
                if pdf_path.exists():
                    return pdf_path.read_bytes()
            except Exception:
                continue
    raise RuntimeError("No converter could open the DOCX.")


@st.cache_data
def pdf_page_images(pdf_bytes: bytes, max_pages: int = 6) -> list[bytes]:
    """Render each PDF page to a PNG image with PyMuPDF, so it looks like the document."""
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
        return [page.get_pixmap(dpi=110).tobytes("png") for page in list(pdf)[:max_pages]]


def paper(inner_html: str) -> str:
    """Wrap HTML in a white A4-like sheet."""
    return (
        "<div style='background:#ffffff;color:#1b2635;padding:28px 32px;"
        "border:1px solid #d3dbe5;box-shadow:0 1px 4px rgba(19,38,59,.12);"
        "font-family:Calibri,Arial,sans-serif;font-size:13px;line-height:1.45'>"
        f"{inner_html}</div>"
    )


def show_document(cv_file, text: str) -> None:
    """Show the CV as a document in a small scrolling window.

    PDF  -> page images.  DOCX -> converted to HTML with mammoth (headings,
    bold, lists and tables kept).  TXT, or no file -> the text on a white page.
    """
    kind, data = cv_file if cv_file else (None, None)
    if kind == "dataset":
        pdf = dataset_cv_pdf(*data)
        kind, data = (".pdf", pdf) if pdf is not None else (None, None)
        if pdf is None:
            cv_file = None
    elif kind == ".docx":
        pdf = docx_to_pdf(data)
        if pdf is not None:
            kind, data = ".pdf", pdf
        else:
            st.caption("Word or LibreOffice is needed to show this file exactly; "
                       "showing a simplified version.")
    # The key gives the box a CSS class, used in page_style() to move the
    # image full-screen button onto the page.
    with st.container(height=560, key="cv_viewer"):
        if kind == ".pdf":
            pages = pdf_page_images(data)
            for number, image in enumerate(pages, start=1):
                st.image(image, caption=f"Page {number}", width="stretch")
        elif kind == ".docx":
            body = mammoth.convert_to_html(io.BytesIO(data)).value
            st.html(paper(body))
        else:
            if cv_file is None:
                st.caption("The original file is not available offline; showing its text.")
            st.html(paper(f"<div style='white-space:pre-wrap'>{html.escape(text.strip())}</div>"))


def read_uploaded_cv(uploaded_file) -> str:
    """Save the upload to a temporary file, read it, and delete it at once.

    Person 1's read_cv needs a file path, but Streamlit gives bytes. The file
    is removed right after reading, so the app never keeps a copy of a CV.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    handle, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as temp:
            temp.write(uploaded_file.getvalue())
        return read_cv(temp_path)
    finally:
        os.remove(temp_path)


def text_or_blank(value) -> str:
    """Dataset fields can be NaN (a float) when empty; show those as ''."""
    return value.strip() if isinstance(value, str) else ""


def job_details(job: dict) -> str:
    """'Company · Job Type · Pay', skipping empty fields."""
    values = [text_or_blank(job.get(key)) for key in ("Company", "Job Type", "Pay")]
    return " · ".join(v for v in values if v)


def badges(items, color: str, empty: str = "none") -> str:
    """Turn a list of words into coloured Streamlit badges."""
    if not items:
        return f":gray[{empty}]"
    return " ".join(f":{color}-badge[{item}]" for item in items)


def short(items, limit=40) -> str:
    """Show a token list as text, cut after `limit` items."""
    text = " · ".join(items[:limit])
    return text + (" …" if len(items) > limit else "")


# ---------------------------------------------------------------------------
# Sidebar: input, filters, method
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Smart CV & Job Matching", page_icon=":material/work:", layout="wide",
                   initial_sidebar_state="expanded")

# Small CSS changes Streamlit has no setting for:
# - hide the Deploy button and the three-dot menu. Only those two: the whole
#   toolbar also holds the button that re-opens a collapsed sidebar;
# - in the CV viewer, put each page's full-screen button on the page itself.
#   By default it sits just above the image, where the scrolling box cuts it off.
st.html("""
<style>
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"] { display: none !important; }
/* Uploader: once a CV is loaded, the small "+" (add files) button becomes a
   full-width "New CV" button under the file. Choosing a file replaces the CV. */
.st-key-upload [data-testid="stFileUploaderDropzone"] > div {
    flex-direction: column;
    align-items: stretch;
    gap: 0.6rem;
}
.st-key-upload button[aria-label="Add files"] {
    width: 100%;
    padding: 0.5rem 0;
    border-radius: 0.4rem;
    background: #1f5f99;
    color: #ffffff;
}
.st-key-upload button[aria-label="Add files"]:hover { background: #174a78; }
.st-key-upload button[aria-label="Add files"] > div { display: none; }
.st-key-upload button[aria-label="Add files"]::after {
    content: "New CV";
    font-weight: 600;
}
.st-key-cv_viewer [data-testid="stElementToolbar"] {
    top: 0.5rem !important;
    right: 0.5rem !important;
    z-index: 5;
}
</style>
""")

with st.sidebar:
    st.markdown("### :material/work: Smart CV")
    st.caption("NLP group project · CV & job matching")
    st.divider()
    st.header(":material/tune: Filters")
    top_n = st.slider("Jobs to show", 1, 20, 5, key="top_n")
    all_jobs = load_jobs()
    job_types = sorted({text_or_blank(t) for t in all_jobs["Job Type"]} - {"", "unknown"},
                       key=str.lower)
    chosen_types = st.multiselect("Job types", job_types, placeholder="All job types",
                                  key="job_types")
    min_score = st.slider("Minimum similarity (%)", 0, 50, 0, step=5, key="min_score")

    st.header(":material/science: Method")
    method_label = st.radio("Word normalisation", list(METHOD_LABELS), horizontal=True, key="method")
    method = METHOD_LABELS[method_label]

    st.divider()
    st.caption(":material/lock: Uploaded CVs are read once and deleted. Use a fictional CV for the demo.")

# ---------------------------------------------------------------------------
# Layout: results in the middle, the current CV on the right
# ---------------------------------------------------------------------------
main, cv_panel = st.columns([2.3, 1], gap="large")

with cv_panel:
    # One box for everything about the CV. The details are added to it further
    # down, once the pipeline has run.
    cv_box = st.container(border=True)
    with cv_box:
        st.subheader(":material/badge: Current CV")
        source = st.radio("CV source", ["Upload a CV", "Use a sample CV", "CV from dataset"],
                          key="source")
        raw_text = None
        cv_category = None   # only set for CVs from the Hugging Face dataset
        cv_file = None       # (extension, bytes) of the original document, for the Full CV view
        if source == "Upload a CV":
            uploaded = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"],
                                        key="upload")
            if uploaded is not None:
                if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
                    st.error(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                else:
                    try:
                        raw_text = read_uploaded_cv(uploaded)
                        # Kept in memory only for this page view, never written to disk.
                        cv_file = (Path(uploaded.name).suffix.lower(), uploaded.getvalue())
                    except (CVReadError, UnicodeDecodeError) as exc:
                        st.error(f"Could not read this CV: {exc}")
        elif source == "Use a sample CV":
            samples = sorted(p.name for p in SAMPLES_DIR.glob("sample_cv.*"))
            choice = st.selectbox("Sample CV (fictional)", samples, key="sample")
            raw_text = read_cv(SAMPLES_DIR / choice)
            cv_file = (Path(choice).suffix.lower(), (SAMPLES_DIR / choice).read_bytes())
        else:
            cvs = load_cv_dataset()
            categories = sorted(cvs["category"].unique())
            cv_category = st.selectbox("Category", categories, key="cv_category",
                                       format_func=lambda c: c.replace("-", " ").title())
            in_category = cvs[cvs["category"] == cv_category]
            labels = {cv_label(row): row["cv_text"] for _, row in in_category.iterrows()}
            chosen = st.selectbox("CV", list(labels), key="dataset_cv")
            raw_text = labels[chosen]
            cv_id = chosen.split(" · ")[0]
            # The original PDF is only downloaded if "Full CV" is opened.
            cv_file = ("dataset", (cv_category, cv_id))
            st.caption("Anonymised resumes from [opensporks/resumes]"
                       "(https://huggingface.co/datasets/opensporks/resumes), CC0.")

        # Two views of the chosen CV, shown in the box below: a short summary,
        # or the document itself.
        view = "Summary"
        if raw_text is not None:
            st.divider()
            view = st.segmented_control(
                "View", ["Summary", "Full CV"], default="Summary", key="cv_view",
                label_visibility="collapsed", width="stretch",
            ) or "Summary"  # clicking the selected option again clears it

with main:
    # Header
    st.title("Smart CV & Job Matching")
    st.caption("Find the job descriptions that best match a CV, and the skills it is missing.")

    if raw_text is None:
        # Landing state: explain the pipeline until a CV is given.
        st.info("Choose a CV in the **Current CV** panel on the right to start.",
                icon=":material/arrow_forward:")
        st.subheader("How it works")
        with st.container(border=True):
            for number, (icon, name, what) in enumerate(PIPELINE, start=1):
                st.markdown(f"{icon} **{number}. {name}**  \n:gray[{what}]")
        st.stop()

    # Run the pipeline
    cleaned = clean_text(raw_text)                         # 1. Person 1
    if not cleaned:
        st.warning("The CV is empty, so there is nothing to match.", icon=":material/warning:")
        st.stop()

    processed_cv = preprocess(cleaned, method=method)      # 2. Person 2
    jobs = load_processed_jobs(method)                     # 2. Person 2, on every job
    if chosen_types:
        jobs = jobs[jobs["Job Type"].isin(chosen_types)]

    ranked = rank_jobs(processed_cv, jobs, top_n=None)     # 4. Person 4 (all jobs)
    ranked = [r for r in ranked if r["similarity_score"] * 100 >= min_score][:top_n]
    # 3. Person 3 compares skills on the CLEANED text, not the preprocessed one:
    #    stemming would turn "analysis" into "analysi" and break the dictionary.
    results = analyze_ranked_jobs(cleaned, ranked)
    cv_skills = detect_skills(cleaned)

    # Summary row
    best = results[0]["similarity_score"] if results else 0.0
    avg_skill = (sum(r["skills"]["skill_match_pct"] for r in results) / len(results)) if results else 0.0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Jobs compared", len(jobs), border=True)
    k2.metric("Best match", f"{best:.0%}", border=True)
    k3.metric("Skills in CV", len(cv_skills), border=True)
    k4.metric("Skill match", f"{avg_skill:.0f}%", border=True,
              help="Average share of each shown job's skills that the CV has.")

    # Right panel: what the app read from the CV, added under the view switch
    with cv_box:
        if cv_category:
            # The dataset says which field this CV belongs to, so the audience
            # can judge whether the top jobs make sense for it.
            st.markdown(f"Category: :blue-badge[{cv_category.replace('-', ' ').title()}]")

        if view == "Summary":
            c1, c2 = st.columns(2)
            c1.metric("Words", len(cleaned.split()))
            c2.metric("Skills", len(cv_skills))
            st.markdown("**Skills found**")
            if cv_skills:
                for category, skills in group_by_category(cv_skills).items():
                    st.caption(category)
                    st.markdown(badges(skills, "blue"))
            else:
                st.caption("No skills from the dictionary were found.")
        else:
            show_document(cv_file, raw_text)

    tab_results, tab_compare, tab_pipeline = st.tabs(
        [":material/work: Top matches", ":material/bar_chart: Compare", ":material/account_tree: Pipeline"]
    )

    # Tab 1: job cards
    with tab_results:
        if not results:
            st.warning("No job passes the current filters. Lower the minimum similarity "
                       "or choose more job types.", icon=":material/filter_alt_off:")
        elif best == 0:
            st.warning("No job shares any words with this CV, so the ranking is not meaningful.",
                       icon=":material/warning:")

        # Show the jobs 5 per page. The page count depends on the filters, so it
        # is part of the widget key: a new count starts again on page 1.
        page_count = math.ceil(len(results) / PAGE_SIZE)
        page = 1
        if page_count > 1:
            page = st.segmented_control(
                "Page", list(range(1, page_count + 1)), default=1, key=f"page_{page_count}",
                format_func=lambda p: f"Page {p}",
            ) or 1
        first = (page - 1) * PAGE_SIZE
        page_results = results[first:first + PAGE_SIZE]
        if page_count > 1:
            st.caption(f"Jobs {first + 1}–{first + len(page_results)} of {len(results)}")

        for result in page_results:
            job, skills, score = result["job"], result["skills"], result["similarity_score"]
            with st.container(border=True):
                left, mid, right = st.columns([3, 1, 1], vertical_alignment="center")
                with left:
                    st.markdown(f"#### {result['rank']}. {result['title']}")
                    st.caption(job_details(job))
                mid.metric("Similarity", f"{score:.0%}")
                right.metric("Skill match", f"{skills['skill_match_pct']:.0f}%")
                st.progress(min(score, 1.0))

                st.markdown(f"**You have:** {badges(skills['matched_skills'], 'green')}")
                st.markdown(f"**To add:** {badges(skills['missing_skills'], 'red')}")
                st.markdown(f"**Other job keywords not in your CV:** {badges(skills['missing_terms'], 'gray')}")
                with st.expander("Full job description", icon=":material/article:"):
                    st.text(job["job_text"])

    # Tab 2: chart + table
    with tab_compare:
        if not results:
            st.caption("Nothing to compare with the current filters.")
        else:
            table = pd.DataFrame(
                {
                    "Rank": r["rank"],
                    "Job": r["title"],
                    "Company": text_or_blank(r["job"].get("Company")) or "—",
                    "Type": text_or_blank(r["job"].get("Job Type")),
                    "Similarity": round(r["similarity_score"] * 100, 1),
                    "Skill match": r["skills"]["skill_match_pct"],
                }
                for r in results
            )
            st.markdown("**Similarity of the top jobs** (cosine similarity of TF-IDF vectors, %)")
            chart = (
                alt.Chart(table)
                .mark_bar(color=CHART_BLUE, cornerRadiusEnd=4, height={"band": 0.7})
                .encode(
                    x=alt.X("Similarity:Q", title="Similarity (%)",
                            axis=alt.Axis(grid=True, gridOpacity=0.3)),
                    y=alt.Y("Job:N", sort=alt.SortField("Rank"), title=None,
                            axis=alt.Axis(labelLimit=260)),
                    tooltip=["Rank", "Job", "Company", "Type",
                             alt.Tooltip("Similarity:Q", format=".1f", title="Similarity (%)"),
                             alt.Tooltip("Skill match:Q", format=".0f", title="Skill match (%)")],
                )
                .properties(height=max(160, 36 * len(table)))
            )
            st.altair_chart(chart, width="stretch")

            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Similarity": st.column_config.ProgressColumn("Similarity", format="%.1f%%",
                                                                  min_value=0, max_value=100),
                    "Skill match": st.column_config.ProgressColumn("Skill match", format="%.0f%%",
                                                                   min_value=0, max_value=100),
                },
            )

    # Tab 3: what each step does
    with tab_pipeline:
        st.caption("What each teammate's step does to this CV. Long lists are cut.")
        steps = preprocessing_steps(cleaned)

        with st.expander("1 · Extract & clean — Person 1", icon=":material/description:", expanded=True):
            left, right = st.columns(2)
            left.markdown("**Text from the file**")
            left.text(raw_text[:1200])
            right.markdown("**After regex cleaning**")
            right.text(cleaned[:1200])

        with st.expander("2 · Preprocess — Person 2", icon=":material/text_fields:"):
            for label, key in (("Tokens", "tokens"), ("Lowercase", "lowercase"),
                               ("Without stopwords", "no_stopwords"), ("Stemmed", "stemmed"),
                               ("Lemmatized", "lemmatized")):
                st.markdown(f"**{label}** ({len(steps[key])})")
                st.caption(short(steps[key]))
            st.markdown(f"**Sent to TF-IDF** ({method_label.lower()})")
            st.code(processed_cv[:600], language=None, wrap_lines=True)

        with st.expander("3 · Linguistic analysis — Person 3", icon=":material/psychology:"):
            pos = pos_summary(cleaned)
            for column, (tag, count) in zip(st.columns(len(pos)), sorted(pos.items(), key=lambda x: -x[1])):
                column.metric(tag.title(), count)
            bigrams = [f"{gram} ({count})" for gram, count in top_ngrams(cleaned, 2, 8)]
            st.markdown("**Most frequent bigrams**")
            st.markdown(badges(bigrams, "gray"))

        with st.expander("4 · Rank — Person 4", icon=":material/leaderboard:"):
            st.markdown(
                f"The CV and **{len(jobs)}** job descriptions are turned into TF-IDF vectors in one "
                f"shared vocabulary. Each job gets the cosine similarity between its vector and the "
                f"CV's. **{len(results)}** jobs are shown after the filters."
            )
