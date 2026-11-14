"""Backfill New York Times article history into the nytnews collection from the NYT Archive API.

The section RSS feeds (nytimes.com/rss) carry the latest 20-60 articles. The Archive
API returns every article of a month, so one call per month recovers the history:
headline, summary, lead paragraph, date and link.

- Each article is filed under the nytnews source of its section, from the URL path
  (/2024/05/01/world/europe/... is nytnews_europe). A section with no feed (Learning,
  Briefing, Insider) gets an archive-only source, nytnews_archive_learning, titled
  "NYT > Learning (archive)".
- Left out: podcast pages (the nytimes collection has those), corrections, paid
  death notices, puzzle companions (Spelling Bee forum, Connections) and documents
  without a headline.
- An article's id is `<source>:<url>`, the id the RSS ingest gives it (NYT guids are
  the URL), so an article both sources have is stored once; one listed in several
  section feeds is several items, and a feed keeps one result per URL.
- `ingested_at` is the publication date, not now, so a decade of articles does not
  show as new in every feed for a week (see backfill_nyt_archive.py).

Needs NYT_API_KEY, and COSMOS_* / OPENROUTER_* as the service has them. The free tier
allows 5 calls a minute; responses are cached under data/nyt_news_archive (trimmed to
the fields used), so a rerun only fetches what is missing.

    .venv/bin/python scripts/backfill_nyt_news.py [--start 2016-11] [--end 2026-10]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from qdrss.app import build_services
from qdrss.feeds import parse_date
from qdrss.ingest import load_sources
from qdrss.models import Item

CACHE = Path("data/nyt_news_archive")
MANIFEST = Path(__file__).resolve().parent.parent / "sources.yaml"
COLLECTION = "nytnews"
FALLBACK = "nytnews_archive"
FALLBACK_TITLE = "New York Times (archive)"
SPACING = 12.5  # seconds between uncached calls: the free tier allows 5 a minute
FIELDS = ("web_url", "headline", "abstract", "lead_paragraph", "pub_date", "type_of_material")
SKIP_MATERIAL = {"correction", "paid death notice"}
# Podcast pages (the nytimes collection has them), puzzle companions and the daily corrections list.
SKIP_PATHS = ("/podcasts/", "/crosswords/", "/pageoneplus/")

# URL path after the date -> source, longest prefix first.
SECTIONS: dict[str, str] = {
    "world/africa": "nytnews_africa",
    "world/americas": "nytnews_americas",
    "world/canada": "nytnews_americas",
    "world/asia": "nytnews_asiapacific",
    "world/australia": "nytnews_asiapacific",
    "world/europe": "nytnews_europe",
    "world/middleeast": "nytnews_middleeast",
    "world": "nytnews_world",
    "us/politics": "nytnews_politics",
    "us/elections": "nytnews_politics",
    "us": "nytnews_us",
    "nyregion": "nytnews_nyregion",
    "upshot": "nytnews_upshot",
    "education": "nytnews_education",
    "obituaries": "nytnews_obituaries",
    "weather": "nytnews_weather",
    "business/dealbook": "nytnews_dealbook",
    "business/economy": "nytnews_economy",
    "business/media": "nytnews_mediaandadvertising",
    "business/smallbusiness": "nytnews_smallbusiness",
    "business/energy-environment": "nytnews_energyenvironment",
    "business": "nytnews_business",
    "your-money": "nytnews_yourmoney",
    "jobs": "nytnews_jobs",
    "realestate": "nytnews_realestate",
    "automobiles": "nytnews_automobiles",
    "technology/personaltech": "nytnews_personaltech",
    "technology": "nytnews_technology",
    "science/space": "nytnews_space",
    "science": "nytnews_science",
    "climate": "nytnews_climate",
    "health": "nytnews_health",
    "well": "nytnews_well",
    "sports/baseball": "nytnews_baseball",
    "sports/ncaabasketball": "nytnews_collegebasketball",
    "sports/ncaafootball": "nytnews_collegefootball",
    "sports/golf": "nytnews_golf",
    "sports/hockey": "nytnews_hockey",
    "sports/basketball": "nytnews_probasketball",
    "sports/football": "nytnews_profootball",
    "sports/soccer": "nytnews_soccer",
    "sports/tennis": "nytnews_tennis",
    "arts/design": "nytnews_artanddesign",
    "arts/dance": "nytnews_dance",
    "arts/music": "nytnews_music",
    "arts/television": "nytnews_television",
    "arts": "nytnews_arts",
    "theater": "nytnews_theater",
    "movies": "nytnews_movies",
    "style/weddings": "nytnews_weddings",
    "fashion/weddings": "nytnews_weddings",
    "style": "nytnews_fashionandstyle",
    "fashion": "nytnews_fashionandstyle",
    "dining": "nytnews_diningandwine",
    "t-magazine": "nytnews_tmagazine",
    "travel": "nytnews_travel",
    "opinion": "nytnews_opinion",
    "books": "nytnews_books",
    "magazine": "nytnews_magazine",
}
# Articles are /YYYY/MM/DD/<section>/[<subsection>/]<slug>; live-blog posts and interactives put
# live/ or interactive/ first, and a live post adds the blog's own slug after the section.
_PATH = re.compile(r"nytimes\.com/(?:live/|interactive/)?\d{4}/\d{2}/\d{2}/(.+)/[^/]+$")


def source_for(url: str) -> str:
    m = _PATH.search(url)
    if not m:
        return FALLBACK
    path = m.group(1).lower()
    for prefix in sorted(SECTIONS, key=len, reverse=True):
        if path == prefix or path.startswith(prefix + "/"):
            return SECTIONS[prefix]
    # A section with no feed of its own (learning, briefing, insider) is an archive-only source.
    section = re.sub(r"[^a-z0-9]+", "_", path.split("/")[0]).strip("_")
    return f"{FALLBACK}_{section}" if section else FALLBACK


def wanted(doc: dict) -> bool:
    url = (doc.get("web_url") or "").lower()
    if not url or any(part in url for part in SKIP_PATHS):
        return False
    if (doc.get("type_of_material") or "").lower() in SKIP_MATERIAL:
        return False
    return bool(((doc.get("headline") or {}).get("main") or "").strip())


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    year, month = map(int, start.split("-"))
    last = tuple(map(int, end.split("-")))
    out = []
    while (year, month) <= last:
        out.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def fetch_month(year: int, month: int, key: str) -> list[dict]:
    path = CACHE / f"{year}-{month:02d}.json"
    if path.is_file():
        return json.loads(path.read_text())
    url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={key}"
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                docs = json.load(response)["response"]["docs"]
            break
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            time.sleep(60)
    time.sleep(SPACING)
    kept = [{k: doc.get(k) for k in FIELDS} for doc in docs if wanted(doc)]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(kept))
    return kept


def archive_title(source: str) -> str:
    section = source.removeprefix(FALLBACK).strip("_").replace("_", " ").title()
    return f"NYT > {section} (archive)" if section else FALLBACK_TITLE


def to_item(doc: dict, titles: dict[str, str]) -> Item | None:
    published = parse_date(doc.get("pub_date") or "")
    if published is None:
        return None
    url = doc["web_url"]
    source = source_for(url)
    content = "\n".join(p for p in (doc.get("abstract"), doc.get("lead_paragraph")) if p)
    return Item(
        id=f"{source}:{url}",
        source=source,
        source_title=titles.get(source) or archive_title(source),
        url=url,
        title=doc["headline"]["main"].strip(),
        content=content,
        published_at=published,
        published_raw=doc.get("pub_date") or "",
        ingested_at=published,
        collection=COLLECTION,
    )


def source_titles() -> dict[str, str]:
    """Each nytnews source's display title, from its comment in the manifest ("NYT > World > Africa")."""
    names = {s.name for s in load_sources(MANIFEST) if s.collection == COLLECTION}
    titles = {}
    lines = MANIFEST.read_text().splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"- name: (\S+)$", line)
        if m and m.group(1) in names and i + 2 < len(lines):
            titles[m.group(1)] = lines[i + 2].strip().lstrip("# ").split(";")[0]
    return titles


async def main() -> None:
    parser = argparse.ArgumentParser()
    now = datetime.now(UTC)
    # The Archive API refuses the current month (403) until it is over.
    last = f"{now.year - (now.month == 1)}-{(now.month - 2) % 12 + 1:02d}"
    parser.add_argument("--start", default=f"{now.year - 10}-{now.month:02d}")
    parser.add_argument("--end", default=last)
    parser.add_argument("--no-embed", action="store_true", help="store only; the next ingest embeds")
    parser.add_argument("--fetch-only", action="store_true",
                        help="fill the cache from the Archive API and stop; no store needed")
    args = parser.parse_args()
    load_dotenv()
    key = os.environ["NYT_API_KEY"]
    logging.basicConfig(level=logging.WARNING)
    if args.fetch_only:
        for year, month in month_range(args.start, args.end):
            print(f"{year}-{month:02d}: {len(fetch_month(year, month, key)):5} articles cached", flush=True)
        return

    titles = source_titles()
    missing = set(SECTIONS.values()) - set(titles)
    if missing:
        raise SystemExit(f"section sources not in the manifest: {sorted(missing)}")

    services = build_services()
    await services.setup()
    try:
        added = 0
        by_source: dict[str, int] = {}
        for year, month in month_range(args.start, args.end):
            items = [item for doc in fetch_month(year, month, key) if (item := to_item(doc, titles))]
            for item in items:
                by_source[item.source] = by_source.get(item.source, 0) + 1
            stored = await services.store.insert_missing(items) if items else 0
            added += stored
            print(f"{year}-{month:02d}: {len(items):5} articles, {stored:5} new  (stored so far {added})",
                  flush=True)
        print(f"stored {added} new archive articles")
        for source, count in sorted(by_source.items(), key=lambda x: -x[1])[:30]:
            print(f"  {source:36} {count}")
        if not args.no_embed:
            print("embedded:", await services.embed_missing(), flush=True)
    finally:
        await services.store.close()


if __name__ == "__main__":
    asyncio.run(main())
