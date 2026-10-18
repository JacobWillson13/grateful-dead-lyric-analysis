"""Build every result table and figure offline from the committed numerical data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator, PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats

from lyric_features import load_metrics
from nlp import distinctive_words, fit_topics, load_stopwords, load_terms, match_play_counts

BLUE, GRAY = "#2a78d6", "#a3a29c"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
TOPICS = 7
TOPIC_ANCHORS = {  # Labels read off each topic's top words; a label applies only if its anchor word is there.
    "train": "Trains and the road", "papa": "Mama, papa, and home", "honey": "Love songs",
    "dancing": "Party and dance", "rain": "Rain, wind, and song", "feel": "Time and reflection",
    "man": "Characters and tall tales",
}
COMPARE = [("word_count", "Word count", None), ("repeated_line_share", "Repeated lines", "pct"),
           ("type_token_ratio", "Type-token ratio", None), ("mattr_50", "MATTR-50", None)]


def style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
        "axes.axisbelow": True, "axes.titlesize": 10.5, "axes.titleweight": "bold"})


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0
    for i, idx in enumerate(order):
        running = max(running, min(1, (len(pvalues) - i) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def fig_length(eligible, rho):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained", sharey=True)
    for ax, metric, label in [(axes[0], "type_token_ratio", "Type-token ratio, whole song"),
                              (axes[1], "mattr_50", "MATTR-50, sliding 50-word window")]:
        ax.scatter(eligible.word_count, eligible[metric], s=16, alpha=.6, color=BLUE, edgecolors="none")
        ax.set(xscale="log", ylim=(0, 1.02), xlabel="Word count (log scale)",
            title=f"{label}   ρ = {rho[metric]:+.2f}")
        ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(ScalarFormatter())
    axes[0].set_ylabel("Share of unique words")
    fig.suptitle("Whole-song TTR punishes long songs. A fixed window removes most of that.",
        fontsize=12.5, fontweight="bold", x=.01, ha="left")
    fig.savefig(FIGURES / "1_length_vs_diversity.png", dpi=150)
    plt.close(fig)


def fig_top50(eligible, table):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), layout="constrained")
    rng = np.random.default_rng(503)
    for ax, (metric, label, fmt) in zip(axes, COMPARE):
        for y, is_top, color in [(1, True, BLUE), (0, False, GRAY)]:
            values = eligible.loc[eligible.top_50.eq(is_top), metric]
            ax.scatter(values, y + rng.uniform(-.17, .17, len(values)), s=12, alpha=.55,
                color=color, edgecolors="none")
            ax.plot([values.median()] * 2, [y - .3, y + .3], color=INK, linewidth=2, solid_capstyle="round")
        row = table.set_index("metric").loc[metric]
        verdict = f"p = {row.p_holm:.3f}" if row.p_holm >= .001 else "p < 0.001"
        ax.set_title(f"{label}\nr = {row.rank_biserial:+.2f}, {verdict}", loc="left")
        ax.set(yticks=[0, 1], ylim=(-.55, 1.55))
        ax.grid(axis="y", visible=False)
        if metric == "word_count":
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 400]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(ScalarFormatter())
        if fmt == "pct":
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticklabels(["Rest of\narchive", "50 most\nplayed"] if ax is axes[0] else ["", ""])
    n_top, n_rest = eligible.top_50.sum(), (~eligible.top_50).sum()
    fig.suptitle("The most-played songs are longer and repeat more lines. Their vocabulary is just as varied.",
        fontsize=12.5, fontweight="bold", x=.01, ha="left")
    fig.text(.01, -.04, f"{n_top} most-played songs vs {n_rest} other archive entries. Black ticks are medians. "
        "r is rank-biserial correlation; p is Mann-Whitney U, Holm-adjusted across the four tests.",
        color=INK_2, fontsize=8.5)
    fig.savefig(FIGURES / "2_most_played_vs_rest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_words(words, n=12):
    picked = pd.concat([words.head(n), words.tail(n)]).sort_values("z")
    fig, ax = plt.subplots(figsize=(8, 6.6), layout="constrained")
    colors = np.where(picked.z > 0, BLUE, GRAY)
    ax.barh(picked.term, picked.z, color=colors, height=.68)
    for x in (-1.96, 1.96):
        ax.axvline(x, color=INK_2, linewidth=.8)
    ax.axvline(0, color=INK, linewidth=.8)
    ax.set(xlabel="Weighted log-odds z-score (each song counts once per word)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    edge = ax.get_xaxis_transform()
    ax.text(.08, 1.01, "More typical of the 50 most played →", transform=edge, ha="left", va="bottom",
        color=BLUE, fontsize=9, fontweight="bold")
    ax.text(-.08, 1.01, "← More typical of the rest", transform=edge, ha="right", va="bottom",
        color=INK_2, fontsize=9, fontweight="bold")
    fig.suptitle("Most-played songs lean toward stories: wine, Texas, bottles, the band.\n"
        "The rest lean toward reflection: see, time, think, feel.", fontsize=12, fontweight="bold",
        x=.01, ha="left")
    fig.text(.01, -.03, "Lines at ±1.96. With ~760 words tested, no single word is significant after correction; "
        "read this as a lean, not a rule.", color=INK_2, fontsize=8.5)
    fig.savefig(FIGURES / "3_distinctive_words.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_topics(prevalence):
    data = prevalence.sort_values("share_rest")
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    y = np.arange(len(data))
    ax.hlines(y, data[["share_top_50", "share_rest"]].min(axis=1), data[["share_top_50", "share_rest"]].max(axis=1),
        color=GRID, linewidth=3)
    ax.scatter(data.share_rest, y, s=70, color=GRAY, zorder=3, edgecolors=SURFACE, linewidths=2, label="Rest of archive")
    ax.scatter(data.share_top_50, y, s=70, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=2, label="50 most played")
    ax.set_yticks(y, [f"{r.label}\n{r.top_words}" for r in data.itertuples()], fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(MultipleLocator(.05))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlim(0, max(data.share_top_50.max(), data.share_rest.max()) + .03)
    for r, yy in zip(data.itertuples(), y):
        if r.p_holm < .05:
            ax.text(max(r.share_top_50, r.share_rest) + .006, yy, "p < 0.05" if r.p_holm >= .001 else "p < 0.001",
                va="center", fontsize=8, color=INK_2)
    ax.set_xlabel("Average share of each song's topic mix")
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle("Seven topics found by NMF. The setlist staples skew toward trains and the road, away from reflection.",
        fontsize=12, fontweight="bold", x=.01, ha="left")
    fig.text(.01, -.03, "Labels are read off each topic's top words. p is Mann-Whitney U on per-song topic share, "
        "Holm-adjusted across the seven topics; unlabeled gaps are not significant.", color=INK_2, fontsize=8.5)
    fig.savefig(FIGURES / "4_topics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    style()
    metrics = load_metrics()
    terms, stopwords = load_terms(), load_stopwords()
    plays = match_play_counts(pd.read_csv(ROOT / "data/setlistfm_play_counts.csv"), metrics,
        pd.read_csv(ROOT / "data/title_aliases.csv"))
    titles = metrics.set_index("song_id").title
    plays.assign(lyrics_title=plays.song_id.map(titles)).to_csv(RESULTS / "play_count_matches.csv", index=False)
    top = set(plays.loc[plays.top_50 & plays.matched, "song_id"])
    clean = metrics[metrics.annotation_flag_count.eq(0) & ~metrics.duplicate_text]
    eligible = clean[clean.word_count >= 50].assign(top_50=lambda d: d.song_id.isin(top))

    # 1. Length and lexical diversity.
    rho = {m: eligible.word_count.corr(eligible[m], method="spearman") for m in ["type_token_ratio", "mattr_25", "mattr_50"]}
    pd.DataFrame({"metric": rho.keys(), "spearman_with_word_count": rho.values(), "songs": len(eligible)}
        ).to_csv(RESULTS / "length_vs_diversity.csv", index=False, float_format="%.6f")
    fig_length(eligible, rho)

    # 2. Most played vs the rest.
    rows = []
    for metric, _, _ in COMPARE:
        a, b = eligible.loc[eligible.top_50, metric], eligible.loc[~eligible.top_50, metric]
        test = stats.mannwhitneyu(a, b)
        rows.append({"metric": metric, "median_top_50": a.median(), "median_rest": b.median(),
            "rank_biserial": 2 * test.statistic / (len(a) * len(b)) - 1, "p": test.pvalue,
            "n_top_50": len(a), "n_rest": len(b)})
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table.p.to_numpy())
    table.to_csv(RESULTS / "most_played_vs_rest.csv", index=False, float_format="%.6f")
    fig_top50(eligible, table)

    # 3. Distinctive words.
    group = sorted(top & set(clean.song_id))
    words = distinctive_words(terms, group, sorted(set(clean.song_id) - top), stopwords)
    words.to_csv(RESULTS / "distinctive_words.csv", index=False, float_format="%.6f")
    fig_words(words)

    # 4. Topics, plus a seed-stability check.
    topic_words, song_topics = fit_topics(terms, clean.song_id, stopwords, k=TOPICS)
    topic_words.to_csv(RESULTS / "topics.csv", index=False, float_format="%.6f")
    summary = topic_words.groupby("topic").term.apply(list)
    reruns = [fit_topics(terms, clean.song_id, stopwords, k=TOPICS, seed=seed)[0].groupby("topic").term.apply(set)
              for seed in range(1, 10)]
    stability = [np.mean([max(len(set(w) & other) / len(w) for other in run) for run in reruns]) for w in summary]
    labels = {}
    for t, w in summary.items():
        hits = [label for anchor, label in TOPIC_ANCHORS.items() if anchor in w]
        labels[t] = hits[0] if len(hits) == 1 else f"Topic {t}"
    song_topics["top_50"] = song_topics.song_id.isin(top)
    shares = song_topics.groupby("top_50")[[f"topic_{t}" for t in summary.index]].mean()
    pvalues = [stats.mannwhitneyu(song_topics.loc[song_topics.top_50, f"topic_{t}"],
        song_topics.loc[~song_topics.top_50, f"topic_{t}"]).pvalue for t in summary.index]
    prevalence = pd.DataFrame({"topic": summary.index, "label": [labels[t] for t in summary.index],
        "top_words": [", ".join(w[:5]) for w in summary], "seed_stability": stability,
        "share_top_50": shares.loc[True].to_numpy(), "share_rest": shares.loc[False].to_numpy(),
        "p": pvalues, "p_holm": holm(np.array(pvalues))})
    prevalence.to_csv(RESULTS / "topic_prevalence.csv", index=False, float_format="%.6f")
    fig_topics(prevalence)

    print(f"{len(eligible)} songs analyzed ({eligible.top_50.sum()} most played); "
          f"TTR rho {rho['type_token_ratio']:+.3f}, MATTR-50 rho {rho['mattr_50']:+.3f}")


if __name__ == "__main__":
    main()
