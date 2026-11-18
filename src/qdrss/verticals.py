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
                 "How did a Singaporean shell company end up at the center of a scandal that toppled Nvidia's chief executive?",
                 "Are the new tariffs actually bringing factory jobs back to the Midwest, or just raising prices for farmers and small manufacturers?",
                 "Why are utilities in Virginia and Texas asking ordinary customers to pay for the power lines feeding A.I. data centers?",
                 "Which Republicans who voted for the tariffs lost their seats in the midterms, and did farm-state voters turn on them?",
                 "How are retirees in Florida coping with homeowners insurance premiums that tripled after the last hurricanes?",
                 "What happened to the families who were separated at the border in 2018, years later?",
                 "News about Chinese spying on American tech companies and the executives caught up in it",
                 "How A.I. is changing the way doctors diagnose patients in American hospitals",
                 "Stories about cities trying to fix homelessness with housing first, and whether it is working",
                 "How did the midterm results change the odds of Congress passing A.I. regulation, tariff relief or a government funding deal before the end of the year?",
                 "What connects the Nvidia export scandal, the Megaspeed investigation and the White House's China trade talks, and who in the administration knew what and when?",
                 "Why are grocery, electricity and insurance bills still rising even as overall inflation cools, and which households are hit hardest?",
             )),
    Vertical("us", "U.S.", ("nytnews_us", "nytnews_politics", "nytnews_nyregion", "nytnews_education",
                            "nytnews_upshot", "nytnews_weather", "nytnews_obituaries", "nytnews_science",
                            "nytnews_space", "nytnews_climate", "nytnews_energyenvironment", "nytnews_health"),
             examples=(
                 "How did the Texas redistricting fight push California to redraw its own maps, and which House seats changed hands because of it?",
                 "Why are measles outbreaks concentrated in Mennonite and other religious communities with low vaccination rates, and how are health officials responding?",
                 "What has happened to the farmworkers and meatpacking workers in small towns after immigration raids emptied their workplaces?",
                 "Why are rural hospitals closing their labor and delivery units, and how far do pregnant women now drive to give birth?",
                 "Did banning phones in schools change anything measurable, like test scores, bullying or teen anxiety?",
                 "Which states have legalized psychedelics for treating veterans with PTSD, and what have the early results been?",
                 "How are New York City's congestion pricing tolls changing traffic, subway ridership and businesses below 60th Street?",
                 "News about how cuts to the National Weather Service affect tornado and hurricane warnings",
                 "Stories about towns along the Colorado River fighting over shrinking water supplies",
                 "Updates on lawsuits challenging President Trump's use of the National Guard in American cities",
                 "After the midterms, which state ballot measures on abortion, minimum wage and redistricting passed, and how will they change policy next year?",
                 "How are school districts that lost federal education funding deciding what to cut, and which students are losing services first?",
                 "Why are home insurers pulling out of California, Florida and Louisiana, and what happens to homeowners left with only the state insurer of last resort?",
             )),
    Vertical("world", "World", ("nytnews_world", "nytnews_africa", "nytnews_americas", "nytnews_asiapacific",
                                "nytnews_europe", "nytnews_middleeast"),
             examples=(
                 "How do Arab citizens of Israel feel about the war in Gaza, and has it changed their relationship with the Israeli state?",
                 "How are Ukrainian soldiers and civilians near the front lines reacting to American pressure to accept a ceasefire with Russia?",
                 "How did Javier Milei bring down inflation in Argentina, and who has paid the price for his spending cuts?",
                 "Why are young Chinese college graduates lying flat or pretending to go to work instead of competing for jobs?",
                 "Which Syrian refugees in Germany and Turkey are actually returning home after Assad's fall, and what are they finding?",
                 "How are Taiwanese families and businesses preparing for the possibility of a Chinese invasion or blockade?",
                 "Why is the AfD winning over young voters in eastern Germany, and what are mainstream parties doing about it?",
                 "News about how the war in Sudan is affecting civilians and the humanitarian aid that reaches them",
                 "Stories about Russian sabotage and drone incursions in NATO countries in Europe",
                 "Updates on hostage negotiations and ceasefire talks between Israel and Hamas",
                 "How have Trump's tariffs and troop decisions changed how European leaders plan for defense without the United States, and who is spending more?",
                 "What would a Russia-Ukraine ceasefire along current front lines mean for the occupied territories, sanctions and Ukrainian refugees in Europe?",
                 "How is China using its control of rare earth minerals as leverage in trade talks with the U.S. and Europe, and are alternatives emerging?",
             )),
    Vertical("business", "Business", ("nytnews_business", "nytnews_economy", "nytnews_dealbook",
                                      "nytnews_mediaandadvertising", "nytnews_smallbusiness", "nytnews_yourmoney",
                                      "nytnews_jobs", "nytnews_technology", "nytnews_personaltech",
                                      "nytnews_automobiles"),
             examples=(
                 "What role did Howard Lutnick and the Commerce Department play in letting Nvidia chips reach China through Megaspeed?",
                 "Which kinds of entry-level white-collar jobs are companies actually replacing with A.I., and what are new graduates doing instead?",
                 "Why are pet owners paying so much more at the vet, and who owns the clinics now?",
                 "Why are car dealers stuck with unsold electric vehicles after the federal tax credit expired?",
                 "If the Fed keeps cutting rates, why are mortgage rates for first-time home buyers still above six percent?",
                 "How did the Google search antitrust ruling change the deal that makes Google the default search engine on iPhones?",
                 "Which American companies moved production out of China because of tariffs, and where did they go?",
                 "News about how Wall Street and the Fed are reacting to A.I. bubble fears",
                 "Stories about workers organizing unions at Starbucks, Amazon and car plants",
                 "Updates on the fight over who will lead Nvidia after Jensen Huang",
                 "How are tariffs on steel, aluminum and Chinese parts showing up in the prices of cars, appliances and holiday gifts this year?",
                 "How are A.I. data center deals between OpenAI, Nvidia, Oracle and Microsoft financed, and why are investors worried about circular spending?",
                 "Why are layoffs rising at big tech and consulting firms even as profits stay high, and which jobs are being cut?",
             )),
    Vertical("arts", "Arts", ("nytnews_arts", "nytnews_artanddesign", "nytnews_dance", "nytnews_music",
                              "nytnews_television", "nytnews_theater", "nytnews_movies", "nytnews_books"),
             examples=(
                 "Which Broadway revivals this season reimagine the original in a way critics loved, and which ones fell flat?",
                 "What are the best novels of the year written by first-time authors, according to the Times critics?",
                 "How are record labels and musicians fighting over A.I. songs that imitate real singers' voices?",
                 "Which museums have returned looted artifacts to Nigeria, Greece or Cambodia, and what prompted them to do it?",
                 "Why have so many prestige TV shows been canceled after one season, and what does that mean for writers?",
                 "How are independent bookstores surviving against Amazon, and why are some of them actually growing?",
                 "What did critics think of the new Taylor Swift album compared with her earlier records?",
                 "Reviews of new novels about immigration and family secrets",
                 "News about cuts to public funding for the arts and orchestras",
                 "Off-Broadway plays that critics are raving about",
                 "Which movies are leading the Oscar race this year, what did Times critics think of them, and how did they do at the box office?",
                 "How are Hollywood studios using A.I. in visual effects, dubbing and scripts after the strikes, and what protections did actors and writers actually win?",
                 "Which books and authors were on the Times best-books lists this year, and what themes connect the novels critics praised most?",
             )),
    Vertical("lifestyle", "Lifestyle", ("nytnews_fashionandstyle", "nytnews_style", "nytnews_weddings",
                                        "nytnews_diningandwine", "nytnews_well", "nytnews_realestate",
                                        "nytnews_tmagazine", "nytnews_magazine", "nytnews_travel"),
             examples=(
                 "What does the research say about strength training versus walking for keeping your memory sharp after 65?",
                 "How are couples in their 70s and 80s finding love again after losing a spouse, and how do their adult children react?",
                 "Why are so many young adults giving up alcohol, and what are bars doing to keep them coming in?",
                 "How are families with small children living in one-bedroom apartments in Manhattan and Brooklyn?",
                 "What should unmarried couples know before buying a home together, in case they break up?",
                 "Which restaurants in New York are worth the wait for a reservation, and how do people actually get a table?",
                 "How do GLP-1 drugs like Ozempic change what people eat and how restaurants plan their menus?",
                 "New restaurant reviews in New York",
                 "Stories about people who changed careers in their 50s",
                 "Advice columns about family and relationships",
                 "What do doctors and recent studies say about taking GLP-1 drugs long term, including muscle loss, cost and what happens when people stop?",
                 "How are adult children managing the cost and logistics of caring for aging parents while raising their own kids?",
                 "Is it smarter to rent or buy in high-cost cities right now, given mortgage rates, rising insurance and property taxes?",
             )),
    Vertical("travel", "Travel", ("nytnews_travel",),
             examples=(
                 "What can you do in Lisbon for three days without spending much money or getting stuck in tourist crowds?",
                 "Which national parks are as spectacular as Yosemite or Zion but far less crowded in summer?",
                 "Is a European ski trip to Austria or Italy cheaper than skiing in Colorado once you count lift tickets?",
                 "Which overnight trains in Europe are worth taking instead of flying, and how comfortable are the sleeper cars?",
                 "Where in Japan should a family with young children go beyond Tokyo and Kyoto?",
                 "How are Venice, Barcelona and Amsterdam limiting tourists, and how does that change a visit?",
                 "Where and when is the best chance of seeing the northern lights without spending a fortune?",
                 "36 Hours guides to cities in Europe",
                 "Travel stories about trips through the American South",
                 "Places to go for food lovers",
                 "How are tariffs, a weaker dollar and new entry fees like Europe's ETIAS changing the cost of an international trip next year?",
                 "Where can I go for a winter beach vacation that is warm in January, safe, not overrun and reachable with one connection from the East Coast?",
                 "What should travelers know about new U.S. border rules, social media checks and visa changes before visiting or returning to the country?",
             )),
    Vertical("opinion", "Opinion", ("nytnews_opinion", "nytnews_sundayreview", "nyt_the_ezra_klein_show",
                                    "nyt_interesting_times", "nyt_the_opinions", "nyt_matter_of_opinion",
                                    "nyt_the_argument", "nyt_sway", "nyt_the_choice"),
             examples=(
                 "Was Nvidia right to push out Jensen Huang, and what does his downfall say about powerful tech founders?",
                 "Would a wealth tax on billionaires actually raise much money, or would the rich just move their fortunes abroad?",
                 "Why have working-class voters, including many Black and Latino men, drifted toward Republicans, and can Democrats win them back?",
                 "Should A.I. companies be legally responsible when their chatbots give teenagers harmful advice?",
                 "Would term limits or more justices fix the Supreme Court, or make the problem worse?",
                 "What would it take to build enough housing to bring rents down in expensive cities?",
                 "Is it wrong to want fewer immigrants, and how should liberals talk about immigration?",
                 "Columns by David Brooks about American culture and character",
                 "Opinion pieces about the future of the Democratic Party",
                 "Debates about free speech on college campuses",
                 "What are columnists arguing Democrats should learn from the midterm results, and do they agree on whether to move left or to the center?",
                 "Is the A.I. boom a bubble, and what do opinion writers think happens to jobs, power prices and the stock market if it bursts?",
                 "Should the U.S. keep restricting chip sales to China after the Nvidia scandal, or do export controls backfire on American companies?",
             )),
    Vertical("audio", "Podcasts", collections=("nytimes",),  # id kept: feed URLs carry vertical=audio
             examples=(
                 "Episodes where people building A.I. explain what they think it will do to jobs and to their own children's future",
                 "Podcasts that investigate wrongful convictions and the people who spent decades in prison for crimes they did not commit",
                 "Conversations about why so many middle-aged men say they have no close friends, and what to do about it",
                 "Episodes explaining how the Supreme Court's immunity ruling changed what a president can legally do",
                 "Interviews with economists about whether tariffs and A.I. spending are hiding a coming recession",
                 "How should parents talk to teenagers about sex, porn and dating apps?",
                 "Episodes about Taylor Swift's fans, her business and what the Eras Tour did to the economy",
                 "New episodes of The Daily about politics",
                 "Ezra Klein interviews about A.I.",
                 "Hard Fork episodes about tech companies",
                 "Podcast episodes that explain how the Nvidia scandal unfolded, who the key players were and what it means for the A.I. race with China",
                 "Conversations about what the midterm election results reveal about American voters heading into the 2028 presidential race",
             )),
    Vertical("cooking", "Cooking", nlweb_url="https://nlweb-recipes-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="nytimes",
             examples=(
                 "A vegetarian Thanksgiving main that will impress my meat-eating relatives and can be made the day before",
                 "A 20-minute weeknight pasta using only pantry staples like canned tomatoes, anchovies and garlic",
                 "Sheet-pan dinners with chicken thighs and vegetables that get crispy without much work",
                 "A rich chocolate layer cake for a child's birthday that a beginner baker can manage",
                 "Soups and stews I can make on Sunday and freeze in portions for weeknight dinners",
                 "Make-ahead breakfasts with lots of protein for busy weekday mornings",
                 "Gluten-free desserts that taste good enough to serve to guests who eat gluten",
                 "A dish to bring to a potluck that travels well and tastes good at room temperature",
                 "Easy Indian vegetarian dinners with lentils and chickpeas",
                 "Simple cocktails for a holiday party with fewer than four ingredients",
                 "A Thanksgiving menu for 12 that includes a vegan and a gluten-free guest, with dishes I can make two days ahead and only one oven",
                 "A dairy-free, nut-free dinner a picky eight-year-old and adults will both enjoy, ready in under 40 minutes on a weeknight",
                 "What can I make with leftover turkey, cranberry sauce and mashed potatoes that doesn't taste like Thanksgiving again?",
             )),
    Vertical("wirecutter", "Wirecutter", nlweb_url="https://nlweb-reviews-3zepccvmbq-uc.a.run.app/ask",
             nlweb_site="wirecutter",
             examples=(
                 "A robot vacuum that handles dog hair and thick rugs without getting stuck on cords",
                 "Is an electric toothbrush worth it, and which one?",
                 "An air purifier for allergies in a small bedroom that is quiet enough to sleep with",
                 "A lightweight laptop for a college student that lasts all day on a battery and costs under $1,000",
                 "Noise-canceling headphones that are comfortable on a long flight and work for phone calls",
                 "A good chef's knife for someone just learning to cook",
                 "An office chair for back pain if I work from home all day",
                 "A carry-on suitcase that fits budget airline size limits and survives rough handling",
                 "Which air fryer should I buy?",
                 "A coffee maker for one person who drinks a cup or two a day",
                 "Holiday gifts under $50 for a teenager who has everything, that Wirecutter staff actually recommend",
                 "A quiet dishwasher and a reliable washing machine for a small apartment, and are they worth buying before tariff price increases?",
             )),
    Vertical("athletic", "The Athletic", ("nytnews_athletic_*", "nyt_athletic_*", "athletic_*", *NYT_SPORTS),
             examples=(
                 "Is Jalen Brunson playing at an MVP level this season, and can the Knicks finally get past the Celtics?",
                 "Which teams can realistically challenge for the Premier League title, and what are their biggest weaknesses?",
                 "How has the rise in pitchers' elbow injuries changed the way teams develop young arms?",
                 "How has NIL money changed which high school stars choose which colleges, and are boosters running recruiting now?",
                 "Which quarterback injuries this season have changed the NFL playoff picture the most?",
                 "Is the U.S. men's soccer team good enough to go deep in a World Cup on home soil?",
                 "Which young tennis players have taken over from Federer, Nadal and Djokovic, and who is next?",
                 "Latest Yankees and Mets offseason moves",
                 "Weekly NFL power rankings and playoff picture",
                 "Transfer rumors involving Premier League clubs",
                 "How did the trade deadline and injuries reshape the NFL playoff race, and which contenders have the easiest remaining schedules?",
                 "Which NBA teams look like real title contenders a month into the season, and how much of it is luck versus sustainable play?",
                 "How will the expanded College Football Playoff field look if the season ended today, and which conference champions get byes?",
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
