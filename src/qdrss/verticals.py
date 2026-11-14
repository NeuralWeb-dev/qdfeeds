"""The NYT search page's tabs: each vertical is a set of sources, or an NLWeb backend.

A qdrss vertical searches only its sources, which is the strongest bias there is: a
Travel search returns Travel articles. Sources are named by selectors resolved against
the manifest: `collection:<name>` (every source of a collection), a glob over source
names (`nytnews_athletic_*`), or an exact name. Names that only the archive backfill
creates (scripts/backfill_nyt_news.py files a section with no feed under
nytnews_archive_<section>) are listed in ARCHIVE_ONLY so selectors can reach them.

Cooking and Wirecutter are NLWeb sites (nlweb-samples on Cloud Run): their recipes and
reviews are not RSS, so the page asks those backends directly through /api/search.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .ingest import Source

ARCHIVE_ONLY = ("nytnews_archive", "nytnews_archive_learning", "nytnews_archive_briefing",
                "nytnews_archive_insider", "nytnews_archive_espanol", "nytnews_archive_sports")

NYT_SPORTS = ["nytnews_baseball", "nytnews_collegebasketball", "nytnews_collegefootball", "nytnews_golf",
              "nytnews_hockey", "nytnews_probasketball", "nytnews_profootball", "nytnews_soccer",
              "nytnews_tennis", "nytnews_archive_sports"]


@dataclass(frozen=True)
class Vertical:
    id: str
    label: str
    selectors: tuple[str, ...] = ()
    nlweb_url: str = ""  # an NLWeb /ask endpoint instead of qdrss sources
    nlweb_site: str = ""
    examples: tuple[str, ...] = ()


VERTICALS: tuple[Vertical, ...] = (
    Vertical("all", "All", ("collection:nytnews", "collection:nytimes", *ARCHIVE_ONLY),
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
    Vertical("audio", "Audio", ("collection:nytimes",),
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


def resolve(vertical: Vertical, sources: list[Source], strict: bool = False) -> frozenset[str]:
    """The source names a vertical searches. With strict, an exact name the manifest lacks is an
    error (a typo would silently empty a tab); the shipped manifest is checked that way in the tests,
    and the app resolves leniently so a test or partial manifest still starts."""
    names = [s.name for s in sources] + list(ARCHIVE_ONLY)
    out: set[str] = set()
    for selector in vertical.selectors:
        if selector.startswith("collection:"):
            out |= {s.name for s in sources if s.collection == selector.removeprefix("collection:")}
        elif any(ch in selector for ch in "*?["):
            out |= set(fnmatch.filter(names, selector))
        elif selector in names:
            out.add(selector)
        elif strict:
            raise ValueError(f"vertical {vertical.id}: no source {selector!r}")
    return frozenset(out)
