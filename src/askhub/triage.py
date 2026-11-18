"""Query triage: how hard is this request, and what does it require?

One cheap call, run beside retrieval, sorts a request into a tier and lists the
constraints it states. The tier picks how much work the rest of the query gets:

  simple  The request only names a dish or kind of dish ("chocolate chip
          cookies", "a classic lasagna"). Any good recipe for it will do, so the
          ranker's choices are shown with the description the page's own markup
          carries, and a cheap model writes the summary. No expansion, no
          describer.
  hard    Everything else gets the full treatment: query expansion, the
          describer (which checks each item against the request and drops what
          fails), substitutions for dietary restrictions.

Constraints carry a kind so that harder tiers can be told apart later (an
allergen rule turns on hidden ingredients; an ingredient preference does not).
The dietary kinds are what restrictions.py used to detect, and keep its routing:

The 2026-10-04 edge-case comparison (eval/edge_cases) found the cheap rankers
weakest exactly on dietary restrictions: nearly every one put a wheat dish first
for a celiac guest and a puff-pastry Wellington first for a gluten-free
Thanksgiving. Only Gemini 3.8 Flash got these right, and it costs ~8x per query,
so a collection can route restricted requests to it
(Config.restriction_ranking_model).

Only some kinds earn the stronger model. On 100 diet queries (2026-10-04, graded
by Claude Sonnet 5, top-5 precision 20B -> 3.8 Flash): allergen rules gained
most (gluten-free .77 -> .93, dairy-free .79 -> .98, nut-free .73 -> .98, pairs
.76 -> .98), because they turn on hidden ingredients; vegetarian, vegan, kosher,
halal and egg-free were already .81-1.00 on 20B; nutritional rules (diabetic,
low-sodium) got worse, the stronger model returning nothing on 12 queries. So
Config.restriction_route_kinds lists the kinds with a clear measured gain:
gluten, dairy and nuts. Eggs showed none (1.00 on both), so egg-free stays on the
default model.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .providers import LanguageModel

logger = logging.getLogger(__name__)

# Kinds that are hard requirements on what the food may contain: these become the
# request's restrictions, which the ranker and describer enforce.
DIETARY_KINDS = frozenset({
    "gluten", "dairy", "nuts", "eggs", "other_allergen", "ethical", "religious",
    "nutritional",
})

PROMPT = (
    "Triage a recipe search request. List every constraint it states beyond naming "
    "a dish, each with one kind. Dietary restrictions are hard requirements on what "
    "the food may contain because of health, allergy, religion or ethics for someone "
    "who will eat it; their kinds are gluten (gluten-free, celiac, wheat), dairy "
    "(dairy-free, lactose), nuts (peanuts or tree nuts), eggs, other_allergen (soy, "
    "shellfish, sesame and the like), ethical (vegetarian, vegan), religious (kosher, "
    "halal) or nutritional (sugar, carbohydrates, sodium, calories). Other "
    "constraints are not restrictions: ingredient (an ingredient to use, use up or "
    "avoid by taste, such as \"with chickpeas\", \"no pasta\", \"leftover rice\"), "
    "time (quick, weeknight, make-ahead, under 30 minutes), method (equipment, "
    "technique, one pot, no oven), occasion (a holiday, party, crowd, picnic, who it "
    "is for), or other (anything else the recipe must satisfy: not spicy, cheap, "
    "healthy, high-protein, easy, highly rated, keeps well). Then give the tier: "
    "simple when the "
    "request only names a dish or a kind of dish, optionally with a cuisine or "
    "course, and states no constraint (\"chocolate chip cookies\", \"a classic "
    "lasagna\", \"thai green curry\", \"roast chicken\"); hard otherwise, including "
    "questions, vague requests and anything that refers to an earlier turn. Return "
    "JSON {\"tier\": \"simple\" or \"hard\", \"constraints\": [{\"name\": short "
    "phrase, \"kind\": kind}]}, constraints empty when there are none."
)


@dataclass(frozen=True, slots=True)
class Triage:
    tier: str = "hard"
    constraints: tuple[tuple[str, str], ...] = ()

    @property
    def simple(self) -> bool:
        return self.tier == "simple"

    @property
    def restrictions(self) -> tuple[tuple[str, str], ...]:
        """The dietary constraints, as (name, kind)."""
        return tuple(item for item in self.constraints if item[1] in DIETARY_KINDS)

    def wire(self) -> dict:
        return {
            "tier": self.tier,
            "constraints": [{"name": name, "kind": kind} for name, kind in self.constraints],
        }


# A failed or slow check gets the full treatment, which is what every request
# had before triage existed.
HARD = Triage()


def parse(output: object) -> Triage:
    if not isinstance(output, dict):
        return HARD
    constraints = []
    for item in output.get("constraints") or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            kind = str(item.get("kind") or "").strip().lower()
        else:
            name, kind = str(item).strip(), ""
        if name:
            constraints.append((name, kind or "other"))
    # A request that states a constraint is not simple, whatever the tier says.
    tier = str(output.get("tier") or "").strip().lower()
    simple = tier == "simple" and not constraints
    return Triage("simple" if simple else "hard", tuple(constraints[:8]))


async def triage(query: str, llm: LanguageModel, timeout_seconds: float) -> Triage:
    """The request's tier and constraints; HARD if the check fails or is slow.

    Fails to the full treatment, so a broken check costs money, never quality.
    """
    try:
        output = await asyncio.wait_for(
            llm.structured(PROMPT, {"request": query}), timeout_seconds
        )
    except Exception as error:
        logger.warning("triage failed (%s)", type(error).__name__)
        return HARD
    return parse(output)
