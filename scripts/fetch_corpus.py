"""Fetch the CMU collection to an ignored local cache; publish metadata, not lyrics."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import time
from urllib.parse import quote, urljoin, urlparse
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
INDEX_URL = "https://www.cs.cmu.edu/~mleone/dead-lyrics.html"
CACHE = ROOT / ".cache" / "lyrics"


class IndexParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.label = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href:
            url = urljoin(INDEX_URL, self.href)
            if url.endswith(".txt") and urlparse(url).hostname == "www.cs.cmu.edu":
                self.links.append({"title": "".join(self.label).strip(), "source_url": url})
            self.href = None


def download(url, allow_html=False):
    encoded = quote(url, safe=":/%?=&',()")
    request = Request(encoded, headers={"User-Agent": "LyricAnalyticsResearch/1.0 (metadata and text metrics)"})
    for attempt in range(3):
        try:
            with urlopen(request, timeout=25) as response:
                raw = response.read(500001)
                if len(raw) > 500000 or (not allow_html and b"<html" in raw[:500].lower()) or not raw.strip():
                    raise ValueError("Unexpected empty, HTML, or oversized response")
                return raw
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch cached source files")
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / ".cache" / "source_index.html"
    if args.refresh or not index_path.exists():
        index_path.write_bytes(download(INDEX_URL, allow_html=True))
    index = IndexParser()
    index.feed(index_path.read_bytes().decode("utf-8", errors="replace"))
    if not index.links or len({x["source_url"] for x in index.links}) != len(index.links):
        raise ValueError("Empty or duplicate source URLs in index")
    started = datetime.now(timezone.utc).isoformat()

    def fetch(item):
        row = dict(item)
        row["song_id"] = hashlib.sha256(item["source_url"].encode()).hexdigest()[:16]
        path = CACHE / (row["song_id"] + ".txt")
        try:
            cached = path.exists() and not args.refresh
            raw = path.read_bytes() if cached else download(item["source_url"])
            if not cached:
                path.write_bytes(raw)
                time.sleep(.25)
            row.update(status="ok", raw_sha256=hashlib.sha256(raw).hexdigest(), raw_bytes=len(raw))
        except (OSError, ValueError) as exc:
            row.update(status="failed", error=type(exc).__name__)
        return row

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = []
        for row in pool.map(fetch, index.links):
            rows.append(row)
            if len(rows) % 25 == 0:
                print(f"Processed {len(rows)}/{len(index.links)} source entries", flush=True)
    manifest = {"index_url": INDEX_URL, "retrieval_run_started_utc": started,
        "retrieval_run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(), "sources": rows}
    target = ROOT / "data" / "source_manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    failed = sum(x["status"] != "ok" for x in rows)
    print(f"Manifest saved: {len(rows)-failed} successful, {failed} failed. Lyrics remain in .cache/.", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
