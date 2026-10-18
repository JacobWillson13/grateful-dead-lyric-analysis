"""NLP on the per-song word counts: play-count matching, distinctive words, and topics.

Everything here runs on `data/processed/term_counts.csv.gz` (song, word, count).
No lyric text or word order is needed.
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfTransformer

from lyric_features import ROOT, title_key

# Setlist entries with no sung lyrics.
INSTRUMENTALS = {"Drums", "Space", "Jam", "Feedback", "Slipknot!"}


def load_terms(path=None):
    terms = pd.read_csv(path or ROOT / "data/processed/term_counts.csv.gz", dtype={"song_id": str, "term": str})
    if set(terms) != {"song_id", "term", "count"} or terms.isna().any().any():
        raise ValueError("Invalid term-count schema")
    if terms.duplicated(["song_id", "term"]).any() or (terms["count"] <= 0).any():
        raise ValueError("Duplicate or non-positive term counts")
    return terms


def load_stopwords():
    return set((ROOT / "data/visual_stopwords.txt").read_text().split())


def content_word(term, stopwords):
    """Keep words of 3+ letters that are not function words."""
    return sum(c.isalpha() for c in term) >= 3 and term not in stopwords


def match_play_counts(plays, metrics, aliases, top_n=50):
    """Attach setlist.fm play counts to lyric entries.

    Matching is exact after normalization, or through an explicit alias row; never fuzzy.
    Instrumentals are dropped before ranking. If two setlist rows land on the same
    lyric entry, the one with more plays is kept.
    """
    alias = {title_key(r.setlist_title): r.lyrics_title for r in aliases.itertuples()}
    by_key = {}
    for row in metrics.itertuples():
        by_key.setdefault(title_key(row.title), []).append((row.title, row.song_id))

    def lookup(title):
        hits = by_key.get(title_key(title), [])
        if len(hits) > 1:
            hits = [h for h in hits if h[0] == title]
            if len(hits) != 1:
                raise ValueError(f"Ambiguous lyric title for {title!r}")
        return hits[0][1] if hits else ""

    songs = plays[~plays.setlist_title.isin(INSTRUMENTALS)].copy()
    songs = songs.sort_values(["times_played", "setlist_title"], ascending=[False, True])
    songs["play_rank"] = songs.times_played.rank(method="min", ascending=False).astype(int)
    target = songs.setlist_title.map(lambda t: alias.get(title_key(t), t))
    songs["song_id"] = target.map(lookup)
    songs["matched"] = songs.song_id.ne("")
    songs["top_50"] = songs.play_rank.le(top_n)
    dupes = songs.matched & songs.song_id.duplicated(keep="first")
    songs.loc[dupes, ["song_id", "matched"]] = ["", False]
    return songs.reset_index(drop=True)


def song_term_matrix(terms, song_ids, vocab, binary=False):
    sub = terms[terms.song_id.isin(song_ids) & terms.term.isin(vocab)]
    rows = pd.Index(song_ids).get_indexer(sub.song_id)
    cols = pd.Index(vocab).get_indexer(sub.term)
    values = np.ones(len(sub)) if binary else sub["count"].to_numpy(float)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(song_ids), len(vocab)))


def distinctive_words(terms, group_ids, other_ids, stopwords, min_songs=5, prior_strength=200):
    """Weighted log-odds with an informative Dirichlet prior (Monroe et al., 2008).

    Each song counts once per word it uses, so a chorus sung twelve times does not
    outweigh a word used across many songs. Positive z = more typical of `group_ids`.
    """
    group_ids, other_ids = list(group_ids), list(other_ids)
    if set(group_ids) & set(other_ids):
        raise ValueError("Groups overlap")
    present = terms[terms.song_id.isin(group_ids + other_ids)]
    df = present.groupby("term").song_id.nunique()
    vocab = sorted(t for t, n in df.items() if n >= min_songs and content_word(t, stopwords))
    y_i = np.asarray(song_term_matrix(terms, group_ids, vocab, binary=True).sum(axis=0)).ravel()
    y_j = np.asarray(song_term_matrix(terms, other_ids, vocab, binary=True).sum(axis=0)).ravel()
    total = y_i + y_j
    alpha = prior_strength * total / total.sum()
    a0 = alpha.sum()
    n_i, n_j = y_i.sum(), y_j.sum()
    delta = (np.log((y_i + alpha) / (n_i + a0 - y_i - alpha))
             - np.log((y_j + alpha) / (n_j + a0 - y_j - alpha)))
    z = delta / np.sqrt(1 / (y_i + alpha) + 1 / (y_j + alpha))
    return pd.DataFrame({"term": vocab, "z": z, "log_odds": delta,
        "share_group": y_i / len(group_ids), "share_other": y_j / len(other_ids)}
        ).sort_values(["z", "term"], ascending=[False, True]).reset_index(drop=True)


def fit_topics(terms, song_ids, stopwords, k=8, min_songs=5, max_share=0.4, top_words=8, seed=0):
    """NMF on sublinear TF-IDF with NNDSVD init. The init uses a randomized SVD, so the seed is fixed."""
    song_ids = sorted(song_ids)
    df = terms[terms.song_id.isin(song_ids)].groupby("term").song_id.nunique()
    vocab = sorted(t for t, n in df.items()
                   if n >= min_songs and n / len(song_ids) <= max_share and content_word(t, stopwords))
    counts = song_term_matrix(terms, song_ids, vocab)
    tfidf = TfidfTransformer(sublinear_tf=True).fit_transform(counts)
    model = NMF(n_components=k, init="nndsvd", max_iter=2000, tol=1e-6, random_state=seed)
    weights = model.fit_transform(tfidf)
    words = pd.DataFrame([{"topic": t, "rank": r + 1, "term": vocab[i], "weight": model.components_[t, i]}
        for t in range(k) for r, i in enumerate(np.argsort(-model.components_[t])[:top_words])])
    shares = weights / np.where(weights.sum(axis=1, keepdims=True) == 0, 1, weights.sum(axis=1, keepdims=True))
    songs = pd.DataFrame(shares, columns=[f"topic_{t}" for t in range(k)])
    songs.insert(0, "song_id", song_ids)
    songs["dominant_topic"] = np.where(weights.sum(axis=1) == 0, -1, weights.argmax(axis=1))
    return words, songs
