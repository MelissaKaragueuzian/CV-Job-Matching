"""Remove emails and links from CV text, then tidy the spaces."""

import re


def remove_emails(text):
    """Replace email addresses with a space.

    The pattern is: username @ domain . extension
    """
    return re.sub(
        r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b",
        " ",
        text,
    )


def remove_urls(text):
    """Replace http:// and https:// links with a space.

    Skill names such as C++, C#, and .NET do not match this pattern.
    """
    return re.sub(
        r"https?://\S+",
        " ",
        text,
    )


def remove_extra_spaces(text):
    """Turn spaces, tabs, and line breaks into a single space."""
    cleaned_text = re.sub(r"\s+", " ", text)
    return cleaned_text.strip()


def clean_text(text):
    """Run the three cleaning steps and return one tidy string."""
    text = remove_emails(text)
    text = remove_urls(text)
    text = remove_extra_spaces(text)
    return text
