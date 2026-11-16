"""The NYT search page's tabs: each vertical is a set of collections, a set of sources, or an
NLWeb backend.

A qdrss vertical searches only its scope, which is the strongest bias there is: a Travel
search returns Travel articles. A scope is either whole collections (All, Podcasts), which
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
             examples=(
                 "What was Bessent's role in Jensen Huang's case?",
                 "Why did Jensen Huang resign from Nvidia?",
                 "Are tariffs the reason my groceries cost more?",
                 "Will A.I. data centers raise my electric bill?",
                 "What do the midterm results mean for 2028?",
                 "Why is home insurance so expensive in Florida now?",
                 "Is it a good time to buy a house?",
                 "Latest on the Nvidia investigation",
                 "New developments in A.I. regulation",
                 "Updates on the Ukraine peace talks",
             )),
    Vertical("us", "U.S.", ("nytnews_us", "nytnews_politics", "nytnews_nyregion", "nytnews_education",
                            "nytnews_upshot", "nytnews_weather", "nytnews_obituaries", "nytnews_science",
                            "nytnews_space", "nytnews_climate", "nytnews_energyenvironment", "nytnews_health"),
             examples=(
                 "What did redistricting do to the House map in Texas and California?",
                 "Is measles back in my state, and should adults get a booster?",
                 "What happens to people picked up in immigration raids?",
                 "Do school phone bans actually work?",
                 "Why are so many rural hospitals closing?",
                 "Are overdose deaths really going down, and why?",
                 "Who is running for governor of New York?",
                 "Latest on immigration enforcement in my city",
                 "New Supreme Court rulings and what they mean",
                 "Climate and extreme weather news in the U.S.",
             )),
    Vertical("world", "World", ("nytnews_world", "nytnews_africa", "nytnews_americas", "nytnews_asiapacific",
                                "nytnews_europe", "nytnews_middleeast"),
             examples=(
                 "What would Ukraine have to give up in a peace deal?",
                 "Who is rebuilding Gaza, and who is paying for it?",
                 "Why are so many young people in China out of work?",
                 "Has Milei actually fixed Argentina's economy?",
                 "Why is the far right gaining in Germany and France?",
                 "What has changed in Syria since Assad fell?",
                 "How likely is China to blockade Taiwan?",
                 "Latest on the war in Ukraine",
                 "News from the Middle East",
                 "Elections around the world this year",
             )),
    Vertical("business", "Business", ("nytnews_business", "nytnews_economy", "nytnews_dealbook",
                                      "nytnews_mediaandadvertising", "nytnews_smallbusiness", "nytnews_yourmoney",
                                      "nytnews_jobs", "nytnews_technology", "nytnews_personaltech",
                                      "nytnews_automobiles"),
             examples=(
                 "What happens to Nvidia now that Jensen Huang is gone?",
                 "Is A.I. taking entry-level jobs from college graduates?",
                 "Why are mortgage rates still high after the Fed cut rates?",
                 "Which companies are winning and losing from tariffs?",
                 "Will Google be broken up after the antitrust ruling?",
                 "Why are electric car makers pulling back?",
                 "Should I worry about a recession?",
                 "Latest A.I. industry news",
                 "Interest rates and the Fed",
                 "Big tech antitrust cases",
             )),
    Vertical("arts", "Arts", ("nytnews_arts", "nytnews_artanddesign", "nytnews_dance", "nytnews_music",
                              "nytnews_television", "nytnews_theater", "nytnews_movies", "nytnews_books"),
             examples=(
                 "What are the best new novels this fall?",
                 "Which Broadway shows are worth the ticket price this season?",
                 "What should I watch this weekend?",
                 "Is A.I. music going to put musicians out of work?",
                 "Which museum shows are worth a trip this year?",
                 "What were the best movies of the year so far?",
                 "Why are so many books being banned from school libraries?",
                 "New book reviews",
                 "New movie and TV reviews",
                 "Theater reviews in New York",
             )),
    Vertical("lifestyle", "Lifestyle", ("nytnews_fashionandstyle", "nytnews_style", "nytnews_weddings",
                                        "nytnews_diningandwine", "nytnews_well", "nytnews_realestate",
                                        "nytnews_tmagazine", "nytnews_magazine", "nytnews_travel"),
             examples=(
                 "What exercise is best for staying sharp after 60?",
                 "How do I make a tiny apartment feel bigger?",
                 "Why are so many people giving up alcohol?",
                 "What actually helps with insomnia?",
                 "How much screen time is too much for teenagers?",
                 "What should I know before buying a home with my partner?",
                 "Wedding stories of couples who met late in life",
                 "New restaurants worth trying in New York",
                 "Health and wellness research news",
                 "Real estate trends in New York",
             )),
    Vertical("travel", "Travel", ("nytnews_travel",),
             examples=(
                 "What should I do with a long weekend in Lisbon?",
                 "Which national parks are less crowded but just as good?",
                 "Is it cheaper to ski in the Alps than in Colorado?",
                 "Can I take a night train across Europe instead of flying?",
                 "Where can I see the northern lights this winter?",
                 "Where should I go in Japan with young kids?",
                 "Which Greek islands are quiet in summer?",
                 "36 Hours city guides",
                 "Travel deals and tips",
                 "Places to go this year",
             )),
    Vertical("opinion", "Opinion", ("nytnews_opinion", "nytnews_sundayreview", "nyt_the_ezra_klein_show",
                                    "nyt_interesting_times", "nyt_the_opinions", "nyt_matter_of_opinion",
                                    "nyt_the_argument", "nyt_sway", "nyt_the_choice"),
             examples=(
                 "Should billionaires pay a wealth tax?",
                 "Should Jensen Huang have been forced out of Nvidia?",
                 "How can Democrats win back working-class voters?",
                 "Is American democracy really in danger?",
                 "Should A.I. companies be responsible for what their chatbots say?",
                 "What would actually make housing affordable?",
                 "Should the Supreme Court have term limits?",
                 "Opinion columns on A.I.",
                 "Opinion pieces on immigration",
                 "Debates about the economy and tariffs",
             )),
    Vertical("audio", "Podcasts", collections=("nytimes",),  # id kept: feed URLs carry vertical=audio
             examples=(
                 "Has the Supreme Court given the president too much power?",
                 "Where is A.I. heading, according to the people building it?",
                 "Podcasts about wrongful convictions",
                 "Is a recession coming?",
                 "How should I handle my kids and their phones?",
                 "How did Taylor Swift build her business empire?",
                 "Why are so many adults lonely?",
                 "New episodes about artificial intelligence",
                 "Podcast episodes about the economy",
                 "Episodes about American politics this week",
             )),
    Vertical("cooking", "Cooking", nlweb_url="https://nlweb-recipes-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="nytimes",
             examples=(
                 "What can I make for Thanksgiving that vegetarians and meat eaters will both love?",
                 "What's a quick pasta I can make with what's in my pantry?",
                 "Easy sheet-pan dinners for a weeknight",
                 "A chocolate birthday cake that's not too hard",
                 "Soups I can make on Sunday and freeze for the week",
                 "Make-ahead breakfasts for busy mornings",
                 "Gluten-free desserts that everyone will eat",
                 "What should I bring to a potluck?",
                 "Easy Indian vegetarian dinners",
                 "Simple cocktails for a party",
             )),
    Vertical("wirecutter", "Wirecutter", nlweb_url="https://nlweb-reviews-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="wirecutter",
             examples=(
                 "What's the best robot vacuum for pet hair?",
                 "Is an electric toothbrush worth it, and which one?",
                 "Do air purifiers help with allergies?",
                 "What laptop should I get for college?",
                 "Which noise-canceling headphones are best for flying?",
                 "What chef's knife should a beginner buy?",
                 "What's a good office chair for back pain?",
                 "Which carry-on suitcase is worth the money?",
                 "Which air fryer should I buy?",
                 "What's a good coffee maker for one person?",
             )),
    Vertical("athletic", "The Athletic", ("nytnews_athletic_*", "nyt_athletic_*", "athletic_*", *NYT_SPORTS),
             examples=(
                 "How are the Knicks doing this season?",
                 "Who will win the Premier League this year?",
                 "Which quarterback injuries matter most for the playoffs?",
                 "Who are the top prospects in the next NBA draft?",
                 "How does the College Football Playoff work now?",
                 "Why are so many pitchers getting Tommy John surgery?",
                 "Is the U.S. ready for the World Cup?",
                 "Latest Yankees news",
                 "NFL playoff picture each week",
                 "Transfer rumors in European soccer",
             )),
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
