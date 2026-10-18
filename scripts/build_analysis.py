"""Build every result table and figure offline from the committed numerical data."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator, PercentFormatter, ScalarFormatter
import numpy as np
import pandas as pd
from scipy import stats

from lyric_features import load_metrics
from nlp import content_word, distinctive_words, fit_topics, load_stopwords, load_terms, match_play_counts
from wordcloud import WordCloud

BLUE, GRAY = "#2a78d6", "#a3a29c"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
TOPICS = 7
TOPIC_ANCHORS = {  # Labels read off each topic's top words; a label applies only if its anchor word is there.
    "train": "Trains and the road", "papa": "Mama, papa, and home", "honey": "Love songs",
    "dancing": "Party and dance", "rain": "Rain, wind, and song", "feel": "Time and reflection",
    "man": "Characters and tall tales",
}
STYLE_LABELS = ["Goin' Down The Road Feelin' Bad", "Sugaree", "Not Fade Away", "Crazy Fingers",
    "Visions of Johanna", "Truckin'", "Eyes Of The World", "Casey Jones", "Dark Star"]
CLOUD_FILLER = set("""got know now one well come comes see way take say make let tell look going gonna never still
much put two really even every things thing another keep around get gets right said came done made""".split())
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
    fig.savefig(FIGURES / "length_vs_diversity.png", dpi=150)
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
    fig.savefig(FIGURES / "most_played_vs_rest.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(FIGURES / "distinctive_words.png", dpi=150, bbox_inches="tight")
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
    fig.savefig(FIGURES / "topics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_word_axis(axis, ticks=(50, 100, 200, 400)):
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(ScalarFormatter())


def fig_wordcloud(terms, clean_ids, stopwords, words):
    """Size = number of songs using the word. Blue = leans toward the 50 most played (z >= 1)."""
    sub = terms[terms.song_id.isin(clean_ids)]
    coverage = sub.groupby("term").song_id.nunique()
    coverage = coverage[[content_word(t, stopwords) and t not in CLOUD_FILLER for t in coverage.index]]
    coverage = coverage.sort_values(ascending=False, kind="stable").head(160)
    lean = words.set_index("term").z
    font = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold"))

    def color(word, **_):
        return BLUE if lean.get(word, 0) >= 1 else "#3d3c39"

    cloud = WordCloud(width=1800, height=880, background_color=SURFACE, font_path=font, max_words=160,
        prefer_horizontal=.92, relative_scaling=.3, max_font_size=170, min_font_size=12, margin=4,
        random_state=503, color_func=color).generate_from_frequencies(coverage.to_dict())
    fig = plt.figure(figsize=(12, 6.8))
    ax = fig.add_axes([.01, .01, .98, .885])
    ax.imshow(cloud.to_array(), interpolation="bilinear")
    ax.set_axis_off()
    fig.text(.01, .985, "The vocabulary of 305 Grateful Dead lyrics", fontsize=14, fontweight="bold", va="top")
    fig.text(.01, .94, "Size is the number of songs that use the word. Blue words lean toward the 50 most-played songs. "
        "Function words and common fillers are hidden.", color=INK_2, fontsize=9.5, va="top")
    fig.savefig(FIGURES / "wordcloud.png", dpi=150)
    plt.close(fig)
    return coverage.rename("songs").reset_index().assign(z=lambda d: d.term.map(lean))


def fig_setlist(plays, titles):
    top = plays[plays.top_50].sort_values(["times_played", "setlist_title"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(8, 12.5), layout="constrained")
    y = np.arange(len(top))
    ax.hlines(y, 0, top.times_played, color=BLUE, linewidth=2)
    ax.scatter(top.times_played, y, s=36, color=BLUE, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.set_yticks(y, top.setlist_title, fontsize=8.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, top.times_played.max() * 1.1)
    ax.set_ylim(-.8, len(top) - .2)
    ax.set_xlabel("Times played live")
    for r, yy in zip(top.itertuples(), y):
        if yy >= len(top) - 3 or yy == 0:
            ax.text(r.times_played + 10, yy, f"{r.times_played}", va="center", fontsize=8.5, color=INK_2)
    ax.set_title("setlist.fm, 2,337 setlists. Drums, Space, and other instrumentals excluded.", loc="left",
        fontsize=9, fontweight="normal", color=INK_2, pad=10)
    fig.suptitle("The 50 songs the Grateful Dead played most", fontsize=13, fontweight="bold", x=.01, ha="left")
    fig.savefig(FIGURES / "top50_setlist.png", dpi=150)
    plt.close(fig)


def fig_plays_vs_diversity(joined, rho, p):
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for is_top, color, label in [(False, GRAY, "Played 58 to 274 times"), (True, BLUE, "50 most played")]:
        d = joined[joined.top_50.eq(is_top)]
        ax.scatter(d.times_played, d.mattr_50, s=34, color=color, alpha=.85, edgecolors=SURFACE,
            linewidths=1, label=label, zorder=3)
    for name, dx, dy in [("Me and My Uncle", -6, 8), ("Goin' Down The Road Feelin' Bad", 8, -2),
                         ("Crazy Fingers", 8, 2), ("Not Fade Away", 8, -2), ("Sugaree", 8, -2)]:
        row = joined[joined.title.eq(name)]
        if not row.empty:
            r = row.iloc[0]
            ax.annotate(name, (r.times_played, r.mattr_50), xytext=(dx, dy), textcoords="offset points",
                fontsize=8.5, color=INK_2, ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    log_word_axis(ax.xaxis, (60, 100, 200, 400, 600))
    ax.set(xlabel="Times played live (log scale)", ylabel="MATTR-50")
    ax.legend(frameon=False, loc="lower left")
    verdict = f"p = {p:.2f}"
    fig.suptitle(f"Playing a song more didn't mean simpler lyrics: Spearman ρ = {rho:+.2f}, {verdict}",
        fontsize=12, fontweight="bold", x=.01, ha="left")
    fig.text(.01, -.03, f"{len(joined)} songs with lyrics in the archive and at least 58 plays on setlist.fm.",
        color=INK_2, fontsize=8.5)
    fig.savefig(FIGURES / "plays_vs_diversity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_style_map(eligible):
    fig, ax = plt.subplots(figsize=(10, 6.4), layout="constrained")
    for is_top, color, label, size in [(False, GRAY, "Rest of archive", 22), (True, BLUE, "50 most played", 34)]:
        d = eligible[eligible.top_50.eq(is_top)]
        ax.scatter(d.repeated_line_share, d.mattr_50, s=size, color=color, alpha=.8, edgecolors=SURFACE,
            linewidths=1, label=label, zorder=3 if is_top else 2)
    offsets = {"Goin' Down The Road Feelin' Bad": (-8, 6, "right"), "Sugaree": (8, 0, "left"),
        "Not Fade Away": (8, -4, "left"), "Crazy Fingers": (8, 2, "left"), "Visions of Johanna": (8, -2, "left"),
        "Truckin'": (8, 4, "left"), "Eyes Of The World": (8, -2, "left"), "Casey Jones": (8, 3, "left"),
        "Dark Star": (8, 4, "left")}
    for name, (dx, dy, ha) in offsets.items():
        row = eligible[eligible.title.eq(name)]
        if not row.empty:
            r = row.iloc[0]
            ax.annotate(name.replace("Of The", "of the"), (r.repeated_line_share, r.mattr_50), xytext=(dx, dy),
                textcoords="offset points", fontsize=8.5, color=INK, ha=ha, va="center")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel="Share of lines that repeat an earlier line", ylabel="MATTR-50 (vocabulary variety)",
        xlim=(-.03, .8))
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Two ways a lyric can be simple: repeating lines, or reusing words", fontsize=12.5,
        fontweight="bold", x=.01, ha="left")
    fig.text(.01, -.03, f"{len(eligible)} songs with at least 50 words. The traditional blues Goin' Down The Road is the most "
        "repetitive entry; Crazy Fingers is near the top for vocabulary variety.", color=INK_2, fontsize=8.5)
    fig.savefig(FIGURES / "style_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_topic_heatmap(song_topics, prevalence, titles):
    order = prevalence.sort_values("share_top_50", ascending=False)
    cols = [f"topic_{t}" for t in order.topic]
    top = song_topics[song_topics.top_50].copy()
    top["title"] = top.song_id.map(titles)
    top["dominant"] = pd.Categorical(top[cols].idxmax(axis=1), categories=cols, ordered=True)
    top["peak"] = top[cols].max(axis=1)
    top = top.sort_values(["dominant", "peak"], ascending=[True, False])
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(8.6, 13), layout="constrained")
    image = ax.imshow(top[cols].to_numpy(), cmap=cmap, vmin=0, vmax=.8, aspect="auto")
    ax.set_xticks(range(len(cols)), order.label, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(top)), [t.replace("   ", ": ") for t in top.title], fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-.5, len(top)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, shrink=.35, pad=.02, location="top")
    bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.outline.set_visible(False)
    bar.set_label("Share of the song's topic mix", color=INK_2)
    fig.suptitle("What the 50 most-played songs are about", fontsize=13, fontweight="bold", x=.01, ha="left")
    fig.savefig(FIGURES / "top50_topic_mix.png", dpi=150)
    plt.close(fig)
    return top[["song_id", "title", *cols]]


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
    fig_setlist(plays, titles)
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
    fig_wordcloud(terms, clean.song_id, stopwords, words).to_csv(RESULTS / "wordcloud_weights.csv",
        index=False, float_format="%.6f")

    # Plays vs metrics across every matched song, plus the style map.
    joined = eligible.merge(plays.loc[plays.matched, ["song_id", "times_played"]], on="song_id")
    play_rows = [{"metric": m, **dict(zip(["spearman_rho", "p"], stats.spearmanr(joined.times_played, joined[m]))),
        "songs": len(joined)} for m in ["word_count", "repeated_line_share", "type_token_ratio", "mattr_50"]]
    play_table = pd.DataFrame(play_rows)
    play_table.to_csv(RESULTS / "plays_vs_metrics.csv", index=False, float_format="%.6f")
    mattr_row = play_table.set_index("metric").loc["mattr_50"]
    fig_plays_vs_diversity(joined, mattr_row.spearman_rho, mattr_row.p)
    fig_style_map(eligible)

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
    fig_topic_heatmap(song_topics, prevalence, titles).to_csv(RESULTS / "top50_topic_mix.csv",
        index=False, float_format="%.6f")

    print(f"{len(eligible)} songs analyzed ({eligible.top_50.sum()} most played); "
          f"TTR rho {rho['type_token_ratio']:+.3f}, MATTR-50 rho {rho['mattr_50']:+.3f}")


if __name__ == "__main__":
    main()
