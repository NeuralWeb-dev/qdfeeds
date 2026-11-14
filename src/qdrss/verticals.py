"""The NYT search page's tabs: each vertical is a set of collections, a set of sources, or an
NLWeb backend.

A qdrss vertical searches only its scope, which is the strongest bias there is: a Travel
search returns Travel articles. A scope is either whole collections (All, Audio), which
also reaches the archive-only sources scripts/backfill_nyt_news.py creates for sections
with no feed (nytnews_archive_<section>), or sources named by selectors resolved against
the manifest: a glob over source names (`nytnews_athletic_*`) or an exact name. The few
archive-only names a selector uses are in ARCHIVE_SOURCES. A source scope also names its
sources' collections, so a store partitioned by collection queries only those partitions.

Cooking and Wirecutter are NLWeb sites (nlweb-samples on Cloud Run): their recipes and
reviews are not RSS, so the page asks those backends directly through /api/search.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .ingest import Source

ARCHIVE_SOURCES = {"nytnews_archive_sports": "nytnews"}  # NYT sports desks with no feed of their own -> collection

NYT_SPORTS = ["nytnews_baseball", "nytnews_collegebasketball", "nytnews_collegefootball", "nytnews_golf",
              "nytnews_hockey", "nytnews_probasketball", "nytnews_profootball", "nytnews_soccer",
              "nytnews_tennis", "nytnews_archive_sports"]


@dataclass(frozen=True)
class Vertical:
    id: str
    label: str
    selectors: tuple[str, ...] = ()
    collections: tuple[str, ...] = ()  # instead of selectors: every item of these collections
    nlweb_url: str = ""  # an NLWeb /ask endpoint instead of qdrss sources
    nlweb_site: str = ""
    examples: tuple[str, ...] = ()


VERTICALS: tuple[Vertical, ...] = (
    Vertical("all", "All", collections=("nytnews", "nytimes"),
             examples=("How tariffs are changing what Americans pay", "The race to build A.I. data centers")),
    Vertical("us", "U.S.", ("nytnews_us", "nytnews_politics", "nytnews_nyregion", "nytnews_education",
                            "nytnews_upshot", "nytnews_weather", "nytnews_obituaries", "nytnews_science",
                            "nytnews_space", "nytnews_climate", "nytnews_energyenvironment", "nytnews_health"),
             examples=("Redistricting fights before the midterms", "Measles outbreaks and vaccine exemptions")),
    Vertical("world", "World", ("nytnews_world", "nytnews_africa", "nytnews_americas", "nytnews_asiapacific",
                                "nytnews_europe", "nytnews_middleeast"),
             examples=("Ceasefire talks in Ukraine", "Elections in Latin America")),
    Vertical("business", "Business", ("nytnews_business", "nytnews_economy", "nytnews_dealbook",
                                      "nytnews_mediaandadvertising", "nytnews_smallbusiness", "nytnews_yourmoney",
                                      "nytnews_jobs", "nytnews_technology", "nytnews_personaltech",
                                      "nytnews_automobiles"),
             examples=("Mortgage rates and home buyers", "Antitrust cases against big tech")),
    Vertical("arts", "Arts", ("nytnews_arts", "nytnews_artanddesign", "nytnews_dance", "nytnews_music",
                              "nytnews_television", "nytnews_theater", "nytnews_movies", "nytnews_books"),
             examples=("Broadway revivals this season", "Novels about climate change")),
    Vertical("lifestyle", "Lifestyle", ("nytnews_fashionandstyle", "nytnews_style", "nytnews_weddings",
                                        "nytnews_diningandwine", "nytnews_well", "nytnews_realestate",
                                        "nytnews_tmagazine", "nytnews_magazine", "nytnews_travel"),
             examples=("Sleep and exercise for people over 50", "Small apartments in New York")),
    Vertical("travel", "Travel", ("nytnews_travel",),
             examples=("A weekend in Lisbon", "Traveling with kids in national parks")),
    Vertical("opinion", "Opinion", ("nytnews_opinion", "nytnews_sundayreview", "nyt_the_ezra_klein_show",
                                    "nyt_interesting_times", "nyt_the_opinions", "nyt_matter_of_opinion",
                                    "nyt_the_argument", "nyt_sway", "nyt_the_choice"),
             examples=("The case for and against a wealth tax", "What liberals get wrong about immigration")),
    Vertical("audio", "Audio", collections=("nytimes",),
             examples=("The Supreme Court and presidential power", "Taylor Swift")),
    Vertical("cooking", "Cooking", nlweb_url="https://nlweb-recipes-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="nytimes",
             examples=("Quick weeknight pasta", "Vegetarian Thanksgiving mains")),
    Vertical("wirecutter", "Wirecutter", nlweb_url="https://nlweb-reviews-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="wirecutter",
             examples=("A quiet dishwasher", "Running shoes for flat feet")),
    Vertical("athletic", "The Athletic", ("nytnews_athletic_*", "nyt_athletic_*", "athletic_*", *NYT_SPORTS),
             examples=("Quarterback injuries this season", "The Premier League title race")),
)
BY_ID = {v.id: v for v in VERTICALS}


@dataclass(frozen=True)
class Scope:
    sources: frozenset[str] | None = None
    collections: frozenset[str] | None = None


def resolve(vertical: Vertical, sources: list[Source], strict: bool = False) -> Scope:
    """What a vertical searches. With strict, a name the manifest lacks is an error (a typo would
    silently empty a tab); the shipped manifest is checked that way in the tests, and the app
    resolves leniently so a test or partial manifest still starts."""
    if vertical.collections:
        known = {s.collection for s in sources}
        if strict and set(vertical.collections) - known:
            raise ValueError(f"vertical {vertical.id}: no collection {sorted(set(vertical.collections) - known)}")
        return Scope(collections=frozenset(vertical.collections))
    home = {s.name: s.collection for s in sources} | ARCHIVE_SOURCES
    names = list(home)
    out: set[str] = set()
    for selector in vertical.selectors:
        if any(ch in selector for ch in "*?["):
            out |= set(fnmatch.filter(names, selector))
        elif selector in names:
            out.add(selector)
        elif strict:
            raise ValueError(f"vertical {vertical.id}: no source {selector!r}")
    # The sources' collections too: a store partitioned by collection (Cosmos) then asks only those
    # partitions instead of fanning out over all of them, which is 15x slower.
    return Scope(sources=frozenset(out), collections=frozenset(home[name] for name in out))
