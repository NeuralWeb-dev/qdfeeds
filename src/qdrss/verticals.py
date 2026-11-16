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
                 "How has the Trump administration changed the rules for selling A.I. chips to China?",
                 "Which industries have tariffs hurt most, and who has been exempted?",
                 "How are A.I. data centers changing electricity prices for ordinary households?",
                 "What did the midterm results mean for Democrats' chances in 2028?",
                 "How has the war in Ukraine changed Europe's defense spending since 2022?",
                 "Why have home insurance costs soared in Florida and California?",
                 "How has the Supreme Court expanded presidential power, case by case?",
                 "What happened to the companies that went all-in on remote work?",
                 "How have weight-loss drugs like Ozempic changed the food industry?",
             )),
    Vertical("us", "U.S.", ("nytnews_us", "nytnews_politics", "nytnews_nyregion", "nytnews_education",
                            "nytnews_upshot", "nytnews_weather", "nytnews_obituaries", "nytnews_science",
                            "nytnews_space", "nytnews_climate", "nytnews_energyenvironment", "nytnews_health"),
             examples=(
                 "How did mid-decade redistricting in Texas and California change the House map?",
                 "Why are measles cases rising in states with more vaccine exemptions?",
                 "How have immigration raids affected farm and construction labor?",
                 "Do school cellphone bans improve grades or just behavior?",
                 "How did cuts to FEMA change hurricane recovery on the Gulf Coast?",
                 "Why are so many rural hospitals closing their maternity wards?",
                 "What is driving the decline in traffic deaths in some cities and not others?",
                 "How has the fentanyl crisis changed since overdose deaths began to fall?",
                 "Which coastal towns are planning managed retreat from rising seas?",
                 "How have cuts to federal science funding affected university research labs?",
             )),
    Vertical("world", "World", ("nytnews_world", "nytnews_africa", "nytnews_americas", "nytnews_asiapacific",
                                "nytnews_europe", "nytnews_middleeast"),
             examples=(
                 "What would a ceasefire in Ukraine mean for the territory Russia holds?",
                 "How has Gaza's reconstruction been funded, and who controls it?",
                 "Why is China's youth unemployment so high despite its tech boom?",
                 "How did Javier Milei's shock therapy change Argentina's economy?",
                 "What is driving the rise of far-right parties in Germany and France?",
                 "How are drought and conflict pushing migration out of the Sahel?",
                 "Why did India and Pakistan come close to war, and how was it defused?",
                 "How is Japan coping with its shrinking and aging population?",
                 "What has changed in Syria since the fall of Assad?",
                 "How are Taiwan's chipmakers preparing for a possible Chinese blockade?",
             )),
    Vertical("business", "Business", ("nytnews_business", "nytnews_economy", "nytnews_dealbook",
                                      "nytnews_mediaandadvertising", "nytnews_smallbusiness", "nytnews_yourmoney",
                                      "nytnews_jobs", "nytnews_technology", "nytnews_personaltech",
                                      "nytnews_automobiles"),
             examples=(
                 "How did Jensen Huang's downfall affect Nvidia's stock and its A.I. customers?",
                 "Which companies are winning and losing from the tariffs on Chinese goods?",
                 "Is A.I. actually replacing entry-level white-collar jobs?",
                 "Why are mortgage rates still high even after the Fed cut rates?",
                 "How are the antitrust cases against Google and Meta likely to end?",
                 "Why are electric vehicle makers cutting back after years of growth?",
                 "How are empty offices and high interest rates squeezing commercial real estate?",
                 "What happens to retirement savings if stablecoins become mainstream?",
                 "How are streaming services trying to stop losing subscribers?",
                 "Why are so many small businesses struggling with credit card fees?",
             )),
    Vertical("arts", "Arts", ("nytnews_arts", "nytnews_artanddesign", "nytnews_dance", "nytnews_music",
                              "nytnews_television", "nytnews_theater", "nytnews_movies", "nytnews_books"),
             examples=(
                 "Which Broadway shows this season are revivals, and how do they differ from the originals?",
                 "What novels best capture life after the pandemic?",
                 "How has A.I. changed the way musicians write and record songs?",
                 "Which museum retrospectives are worth traveling for this year?",
                 "How did prestige TV dramas change after the writers' strike?",
                 "Why are so many hit movies now based on video games?",
                 "Which young classical musicians are critics most excited about?",
                 "How are book bans in school libraries affecting authors and publishers?",
                 "What made this year's best horror films so different from the usual?",
                 "How are museums dealing with demands to return looted art?",
             )),
    Vertical("lifestyle", "Lifestyle", ("nytnews_fashionandstyle", "nytnews_style", "nytnews_weddings",
                                        "nytnews_diningandwine", "nytnews_well", "nytnews_realestate",
                                        "nytnews_tmagazine", "nytnews_magazine", "nytnews_travel"),
             examples=(
                 "What kinds of exercise best protect memory and balance after 60?",
                 "How are New Yorkers making tiny apartments feel bigger?",
                 "Couples who met late in life and married in their 70s",
                 "Why are more people choosing not to drink alcohol, and what are they drinking instead?",
                 "How are people furnishing homes with secondhand and vintage pieces?",
                 "What sleep habits actually help with insomnia, according to research?",
                 "Why are young people leaving big cities for small towns?",
                 "How is the fashion industry dealing with the glut of unsold clothes?",
                 "What should you know before buying a home with a partner you're not married to?",
                 "How are parents managing their teenagers' phone and social media use?",
             )),
    Vertical("travel", "Travel", ("nytnews_travel",),
             examples=(
                 "A long weekend in Lisbon on a modest budget",
                 "Which national parks are least crowded but just as beautiful?",
                 "36 Hours in a European city that isn't overrun by tourists",
                 "Ski resorts in the Alps that are cheaper than the big American ones",
                 "Overnight train trips across Europe that replace a flight",
                 "How cities like Venice and Barcelona are limiting tourists",
                 "Traveling in Japan with young kids",
                 "Where to see the northern lights, and when",
                 "Road trips through the American Southwest off the interstate",
                 "Islands in Greece that are quiet in summer",
             )),
    Vertical("opinion", "Opinion", ("nytnews_opinion", "nytnews_sundayreview", "nyt_the_ezra_klein_show",
                                    "nyt_interesting_times", "nyt_the_opinions", "nyt_matter_of_opinion",
                                    "nyt_the_argument", "nyt_sway", "nyt_the_choice"),
             examples=(
                 "The case for and against a wealth tax on billionaires",
                 "Should tech CEOs like Jensen Huang be held to a higher standard?",
                 "What the Democratic Party needs to do to win back working-class voters",
                 "Is American democracy really in decline, and what would fix it?",
                 "Should A.I. companies be liable for what their chatbots say?",
                 "Why housing is so expensive, and which policies would actually help",
                 "Is it time to rethink how the Supreme Court works?",
                 "What liberals get wrong about immigration",
                 "Should colleges stop taking public positions on political issues?",
                 "Are tariffs a smart way to bring manufacturing back?",
             )),
    Vertical("audio", "Podcasts", collections=("nytimes",),  # id kept: feed URLs carry vertical=audio
             examples=(
                 "Episodes about how the Supreme Court has expanded presidential power",
                 "Conversations with A.I. researchers about where the technology is heading",
                 "Investigative series about wrongful convictions",
                 "Interviews with economists about whether a recession is coming",
                 "Episodes about raising kids in the age of smartphones",
                 "How Taylor Swift built a business empire",
                 "Debates about whether the U.S. should keep arming Ukraine",
                 "Episodes about the people and companies behind the A.I. boom",
                 "Conversations about loneliness and friendship in middle age",
                 "Episodes explaining the midterm election results",
             )),
    Vertical("cooking", "Cooking", nlweb_url="https://nlweb-recipes-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="nytimes",
             examples=(
                 "A vegetarian Thanksgiving main that will impress meat eaters",
                 "Weeknight pasta that takes 20 minutes with pantry ingredients",
                 "Sheet-pan chicken dinners with crispy vegetables",
                 "A showstopper chocolate cake for a birthday party",
                 "Hearty soups that freeze well for winter",
                 "Make-ahead breakfasts for a busy week",
                 "Gluten-free desserts that don't taste gluten-free",
                 "Dishes to bring to a potluck that travel well",
                 "Indian vegetarian dinners for beginners",
                 "Cocktails with fewer than four ingredients",
             )),
    Vertical("wirecutter", "Wirecutter", nlweb_url="https://nlweb-reviews-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="wirecutter",
             examples=(
                 "A robot vacuum that can handle pet hair and rugs",
                 "Running shoes for flat feet and long runs",
                 "An air purifier for allergies in a small bedroom",
                 "A lightweight laptop for a college student on a budget",
                 "Noise-canceling headphones for long flights",
                 "A chef's knife for a beginner cook",
                 "An office chair for back pain",
                 "A carry-on suitcase that survives budget airlines",
                 "A mattress for side sleepers",
                 "A coffee maker for one person",
             )),
    Vertical("athletic", "The Athletic", ("nytnews_athletic_*", "nyt_athletic_*", "athletic_*", *NYT_SPORTS),
             examples=(
                 "Which quarterback injuries have changed the playoff race this season?",
                 "Who is favored in the Premier League title race, and why?",
                 "How are the Knicks doing this season, and is Jalen Brunson playing like an MVP?",
                 "What did the World Series reveal about building a modern pitching staff?",
                 "How will the expanded College Football Playoff change the rankings?",
                 "Top prospects in the next NBA draft and the teams tanking for them",
                 "How has NIL money changed college basketball recruiting?",
                 "Which young tennis players could replace the Big Three?",
                 "How is the U.S. men's soccer team preparing for the World Cup?",
                 "Why are pitchers' elbow injuries rising across baseball?",
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
