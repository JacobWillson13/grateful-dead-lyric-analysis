"""Tokenization and lexical metrics. Only numbers derived from these leave the local cache."""
from collections import Counter
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TOKEN_PATTERN = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)


def title_key(title):
    text = unicodedata.normalize("NFKD", str(title)).casefold()
    return "".join(ch for ch in text if ch.isalnum())


def tokenize(text):
    """Unicode alphabetic tokens; internal apostrophes stay inside contractions."""
    return [word.replace("’", "'").casefold() for word in TOKEN_PATTERN.findall(text)]


def mattr(tokens, window=50):
    """Mean unique-token proportion across all contiguous fixed-size windows.

    Return NaN for short documents rather than silently changing the window.
    """
    if window < 1:
        raise ValueError("window must be positive")
    if len(tokens) < window:
        return float("nan")
    counts = Counter(tokens[:window])
    total = len(counts) / window
    for i in range(window, len(tokens)):
        outgoing = tokens[i-window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
        counts[tokens[i]] += 1
        total += len(counts) / window
    return total / (len(tokens)-window+1)


def clean_source(text, title, filename_title=None):
    """Remove an exactly normalized leading title, preserving all other lines.

    This conservative rule avoids silently deleting lyric lines or editorial text.
    Potential editorial annotations are flagged for review, not guessed away.
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    candidates = {title_key(title), title_key(filename_title or title)}
    removed = bool(lines and title_key(lines[0].strip()) in candidates)
    if removed:
        lines.pop(0)
    annotation = re.compile(r"^(?:copyright\b|words\s+(?:by|and)\b|music\s+by\b|lyrics\s+by\b|notes?\s*:|\[|\()", re.I)
    flags = sum(bool(annotation.search(line.strip())) for line in lines if line.strip())
    return "\n".join(lines).strip(), removed, flags


def text_metrics(text):
    tokens = tokenize(text)
    n = len(tokens)
    unique = len(set(tokens))
    # Alphabetic character length excludes apostrophe punctuation.
    lengths = [sum(c.isalpha() for c in token) for token in tokens]
    lines = [" ".join(tokenize(line)) for line in text.splitlines()]
    lines = [line for line in lines if line]
    return {"word_count": n, "unique_word_count": unique,
        "avg_word_length": float(np.mean(lengths)) if n else float("nan"),
        "type_token_ratio": unique/n if n else float("nan"),
        "mattr_50": mattr(tokens, 50), "mattr_25": mattr(tokens, 25),
        "lyric_line_count": len(lines),
        "repeated_line_share": (len(lines)-len(set(lines)))/len(lines) if lines else float("nan")}


def load_metrics(path=None):
    frame = pd.read_csv(path or ROOT / "data/processed/song_metrics.csv")
    required = {"song_id", "title", "source_url", "word_count", "unique_word_count", "avg_word_length", "type_token_ratio", "mattr_50", "repeated_line_share", "annotation_flag_count", "duplicate_text"}
    if required - set(frame):
        raise ValueError("Derived metrics are missing required columns")
    if frame.song_id.duplicated().any():
        raise ValueError("Duplicate source entry IDs")
    if frame.empty or frame[["word_count", "unique_word_count"]].isna().any().any() or (frame.word_count < 0).any() or (frame.word_count < frame.unique_word_count).any():
        raise ValueError("Invalid word counts")
    return frame
