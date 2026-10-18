"""Compute distributable metrics from the ignored source cache; never export lyrics."""
import hashlib
import gzip
import json
from collections import Counter
from pathlib import Path
import sys
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from lyric_features import clean_source, text_metrics, tokenize


def main():
    manifest = json.loads((ROOT / "data/source_manifest.json").read_text())
    rows = []
    term_rows = []
    for entry in manifest["sources"]:
        if entry["status"] != "ok":
            continue
        raw = (ROOT / ".cache/lyrics" / (entry["song_id"] + ".txt")).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["raw_sha256"]:
            raise ValueError("Cache differs from manifest; refresh or recover the matching source snapshot")
        try:
            text, encoding = raw.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            text, encoding = raw.decode("cp1252"), "cp1252"
        filename = unquote(Path(urlparse(entry["source_url"]).path).stem).replace("_", " ")
        clean, removed, flags = clean_source(text, entry["title"], filename)
        term_rows.extend({"song_id": entry["song_id"], "term": term, "count": count}
            for term, count in Counter(tokenize(clean)).items())
        rows.append({"song_id": entry["song_id"], "title": entry["title"], "source_url": entry["source_url"],
            "raw_sha256": entry["raw_sha256"], "source_encoding": encoding,
            "title_header_removed": removed, "annotation_flag_count": flags,
            "clean_text_sha256": hashlib.sha256(clean.encode()).hexdigest(), **text_metrics(clean)})
    frame = pd.DataFrame(rows).sort_values("song_id")
    if frame.empty:
        raise ValueError("No successful source documents")
    frame["duplicate_text"] = frame.clean_text_sha256.duplicated(keep=False)
    target = ROOT / "data/processed"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "song_metrics.csv", index=False, float_format="%.8f")
    counts = pd.DataFrame(term_rows).sort_values(["song_id", "term"])
    # A fixed gzip timestamp makes regenerated word-count data byte-identical.
    (target / "term_counts.csv.gz").write_bytes(gzip.compress(counts.to_csv(index=False).encode(), mtime=0))
    print(f"Built metrics for {len(frame)} source entries; no lyric text exported")


if __name__ == "__main__":
    main()
