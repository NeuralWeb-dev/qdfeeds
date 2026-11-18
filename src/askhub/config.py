from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Keys the ranking projection keeps by default. The projection is recursive, so
# nested keys (`ratingValue`, `addressLocality`, `price`) have to be listed too,
# or a rating or an address reaches the judge as an empty object. Long text the
# judge does not need to decide relevance -- recipe instructions, review bodies
# -- is left out: it is most of the tokens and none of the signal.
DEFAULT_RANKING_FIELDS: tuple[str, ...] = (
    "@type", "name", "description", "url", "headline", "alternativeHeadline",
    "alternateName", "author", "datePublished", "keywords", "genre",
    # recipes
    "recipeCategory", "recipeCuisine", "recipeIngredient", "recipeYield", "totalTime",
    "prepTime", "cookTime", "nutrition", "calories", "suitableForDiet",
    # ratings, everywhere
    "aggregateRating", "ratingValue", "ratingCount", "reviewCount",
    # movies
    "actor", "director", "creator", "duration", "contentRating",
    # product reviews
    "hasPart", "brand", "offers", "price", "priceCurrency",
    # trails and homes
    "address", "streetAddress", "addressLocality", "addressRegion", "postalCode",
    "numberOfRooms", "numberOfBedrooms", "numberOfBathroomsTotal", "floorSize", "lotSize",
    "value", "unitCode", "yearBuilt", "schoolDistrict", "propertyType", "features",
    "amenities", "heating", "cooling", "parking", "hoaFee",
)


# Set by benchmarks/run_retrieval.py: 100 queries whose relevance is stated as
# predicates over the corpus, so the gold sets are free and survive a rebuild.
# 40 are known-item lookups (gold is one named repository), 60 are descriptive;
# the 20 whose gold set exceeds 5% of the corpus are excluded from tuning
# because they cannot separate one ranking from another.
#
#                      known-item nDCG@10    descriptive nDCG@10
#   flat BM25                    0.701                  0.607
#   these weights                0.978                  0.627
#
# Two results here were counter to the design intuition and are worth keeping
# in view. Weighting `description` *up* hurts both categories -- the enriched
# one-sentence description is short and generic, while the specifics a
# descriptive query matches on live in `body` (keywords, subject terms,
# intended uses, evidence quotes), so promoting description suppresses the
# evidence. And `b` for `name` turned out not to matter at all: at this weight
# the name field saturates whatever the length normalisation, so all three
# fields keep the standard 0.75 rather than the damped value first proposed.
# The optimum is a broad plateau; these sit mid-plateau, not on a grid edge.
DEFAULT_BM25_FIELD_WEIGHTS: dict[str, float] = {"name": 32.0, "description": 1.0, "body": 0.5}
DEFAULT_BM25_FIELD_B: dict[str, float] = {"name": 0.75, "description": 0.75, "body": 0.75}
# Weights on the per-field embedding channels, tuned on the same benchmark
# (benchmarks/run_retrieval.py --vector).
#
# Splitting the embedding by field matters as much as splitting BM25 did. Scored
# alone, the three channels are good at completely different things:
#
#                    known-item nDCG@10    descriptive nDCG@10
#   name vector              0.885                  0.380
#   description vector       0.254                  0.612
#   body vector              0.231                  0.698
#
# A single vector over the whole record is dominated by `body`, which is 76% of
# the text -- which is why one flat embedding could not answer a query that
# names a model.
#
# These weights are tuned on the union's coverage, not on the vector channel's
# own ranking: the pipeline takes a head from each channel and hands the union
# to the ranking model, so retrieval's job is to get a relevant record in front
# of the judge, not to order it. On that measure, adding the vector channel
# lifts descriptive coverage from 0.923 to 0.974 and raises the mean number of
# relevant records reaching the ranker from 3.87 to 5.96, with known-item
# already at 1.000 from BM25F alone. Coverage is insensitive to the weights
# themselves -- every grid point scored 0.974 -- so these stay equal rather than
# pretending to a precision the measurement does not support.
DEFAULT_VECTOR_FIELD_WEIGHTS: dict[str, float] = {"name": 1.0, "description": 1.0, "body": 1.0}


@dataclass(frozen=True, slots=True)
class Config:
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "openai/gpt-oss-120b"
    ranking_model: str | None = None
    ranking_reasoning_effort: str | None = None
    llm_reasoning_effort: str | None = "low"
    ranking_provider_sort: str | None = None
    # Requests to these sites that state a dietary restriction are ranked by
    # restriction_ranking_model instead of the default (triage.py).
    restriction_ranking_model: str | None = None
    restriction_sites: tuple[str, ...] = ()
    # Which restriction kinds switch models (triage.py has the measurements).
    restriction_route_kinds: tuple[str, ...] = ("gluten", "dairy", "nuts")
    # Sites whose requests are triaged (triage.py) and, when one only names a dish,
    # answered the cheap way: the ranker's choices with their markup descriptions
    # and a summary from simple_summary_model, without expansion or the describer.
    simple_sites: tuple[str, ...] = ()
    simple_summary_model: str | None = None
    # A chat model asked when a decision model returns no results (pipeline.py); "" turns it off.
    decision_fallback_model: str | None = "openai/gpt-oss-20b"
    # Bounds the triage call, which also finds restrictions for restriction_sites.
    triage_timeout: float = 4.0
    # Sites whose ranker reads dietary restrictions from the request itself and
    # suggests substitutions, without the detection call.
    substitution_sites: tuple[str, ...] = ()
    # One call to this model rewrites the top describe_count results' descriptions
    # and writes the summary, from the records alone; the default ranker still
    # judges every candidate. Unset: the ranker's descriptions and the utility
    # model's summary stand.
    describe_model: str | None = None
    # A stronger describer for requests whose restrictions include one of
    # restriction_route_kinds (the parallel check's result decides).
    describe_restricted_model: str | None = None
    # Send the describer the ranking projection of each record rather than
    # raw markup cut at a length.
    describe_project: bool = False
    # Preferred providers for the default ranking model.
    ranking_provider_order: tuple[str, ...] = ()
    describe_count: int = 8
    # Provider routing for the general model (expansion, follow-up rewriting,
    # summaries). OpenRouter's default balancing sent gpt-oss-120b expansion
    # calls to providers taking up to 5.7s; sorting by throughput measured
    # 0.5s median and 0.6s worst over ten calls.
    llm_provider_sort: str | None = "throughput"
    ranking_max_tokens: int = 100
    embedding_model: str = "openai/text-embedding-3-small"
    app_url: str | None = None
    app_title: str = "ask-samples"
    data_dir: Path = Path("data")
    # Manifest naming where each collection is fetched from. When absent, the
    # data directory is used as-is, which is how tests and local checkouts run.
    sources_manifest: Path = Path("sources.yaml")
    # How often to recheck sources. A check that finds nothing changed costs one
    # conditional request per source and no embedding, so this can be frequent.
    refresh_interval_hours: float = 4.0
    cache_dir: Path | None = None
    # gs://bucket holding this service's collection (corpus .jsonl plus its
    # embedding .npz files), downloaded into data_dir and cache_dir at startup.
    gcs_bucket: str | None = None
    # A prebuilt catalog (tools/build_snapshot.py). When it holds catalog.pkl the
    # service loads it instead of parsing the corpus and building the index.
    snapshot_dir: Path | None = None
    # Folder of the bucket holding the snapshot: snapshot/ for the full one,
    # keyword/ for a keyword-only one served with the VM's vectors or Cosmos.
    gcs_snapshot_prefix: str = "snapshot/"
    # Records and vectors in Azure Cosmos DB (cosmos_catalog.py). Set the
    # endpoint to use it; the partition defaults to the snapshot's collection.
    cosmos_endpoint: str | None = None
    cosmos_key: str | None = None
    cosmos_database: str = "samples"
    cosmos_container: str = "items"
    cosmos_partition: str | None = None
    # Vectors and records on the neuralweb VM (vm_catalog.py, vector_service.py),
    # with only the keyword index (gcs_snapshot_prefix keyword/) in the service.
    vector_url: str | None = None
    vector_token: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    bm25_rank_count: int = 10
    vector_rank_count: int = 10
    # Ceilings for the per-request overrides. A wide retrieval judged by an
    # expensive model is exactly what an evaluation wants and exactly what an
    # abusive request wants, so the endpoint caps what it will accept rather
    # than trusting the caller.
    max_retrieval_count: int = 200
    max_ranking_count: int = 200
    ranking_batch_count: int = 3
    ranking_batch_return_count: int = 3
    max_results: int = 10
    min_score: int = 70
    strong_score_threshold: int = 90
    # Decontextualization is one LLM call on the critical path; if it is slow
    # the original query is searched instead.
    decontextualize_timeout: float = 15.0
    # Query expansion runs beside the ordinary search and adds candidates when it
    # returns; it never delays the first results. A slow expansion is dropped.
    query_expansion: bool = True
    expansion_timeout: float = 6.0
    # Candidates the expanded search may add: from the expanded terms, and per
    # suggested name (only records whose own name matches the suggestion).
    expansion_term_count: int = 10
    rate_limit_per_client_per_minute: int = 10
    rate_limit_global_per_hour: int = 60
    rate_limit_global_per_day: int = 500
    max_concurrent_queries: int = 4
    cors_origins: tuple[str, ...] = ()
    mcp_allowed_hosts: tuple[str, ...] = ()
    trusted_proxy_ips: tuple[str, ...] = ("127.0.0.1", "::1")
    # Which schema.org keys survive the projection sent to the ranking model.
    # The list is corpus-shaped, not universal: a product catalog needs `offers`
    # and `eligibleRegion`, a model catalog needs `about` and `subjectOf`, and a
    # list inherited from one corpus silently blinds the ranker on another.
    # Keeping it in config means changing corpus does not mean changing code.
    ranking_fields: tuple[str, ...] = DEFAULT_RANKING_FIELDS
    # BM25F field weights and length normalisation. `name` is weighted heavily
    # because a query that types a model's name is asking for that model, not
    # for something that mentions it; `body` stays at 1.0 so it can corroborate
    # a match without winning one.
    bm25_field_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_BM25_FIELD_WEIGHTS)
    )
    bm25_field_b: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_BM25_FIELD_B))
    vector_field_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_VECTOR_FIELD_WEIGHTS)
    )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        """From the process environment (and .env), or from `environ` alone: an app
        that hosts this pipeline beside its own settings (qdrss) passes its own."""
        if environ is None:
            load_dotenv()
        env = os.environ if environ is None else environ

        def integer(name: str, default: int) -> int:
            return int(env.get(name, str(default)))

        def values(name: str, default: str = "") -> tuple[str, ...]:
            return tuple(
                value.strip() for value in env.get(name, default).split(",") if value.strip()
            )

        def weights(name: str, default: dict[str, float]) -> dict[str, float]:
            """Per-field tuning as a JSON object, merged over the defaults.

            Merged rather than replaced so setting one field's weight does not
            silently drop the other fields to zero.
            """
            raw = env.get(name, "").strip()
            if not raw:
                return dict(default)
            return {**default, **{k: float(v) for k, v in json.loads(raw).items()}}

        host = env.get("ASKHUB_HOST", "127.0.0.1")
        port = integer("ASKHUB_PORT", 8000)
        local_mcp_hosts = f"{host}:{port},127.0.0.1:{port},localhost:{port}"

        return cls(
            openrouter_api_key=env.get("OPENROUTER_API_KEY") or None,
            openrouter_base_url=env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            llm_model=env.get("OPENROUTER_MODEL", "openai/gpt-oss-120b"),
            ranking_model=env.get("OPENROUTER_RANKING_MODEL") or None,
            ranking_reasoning_effort=env.get("OPENROUTER_RANKING_REASONING_EFFORT") or None,
            llm_reasoning_effort=env.get("OPENROUTER_LLM_REASONING_EFFORT", "low") or None,
            ranking_provider_sort=env.get("OPENROUTER_RANKING_PROVIDER_SORT") or None,
            restriction_ranking_model=env.get("ASKHUB_RESTRICTION_RANKING_MODEL") or None,
            restriction_sites=tuple(
                site.strip() for site in env.get("ASKHUB_RESTRICTION_SITES", "").split(",")
                if site.strip()
            ),
            simple_sites=values("ASKHUB_SIMPLE_SITES"),
            simple_summary_model=env.get("ASKHUB_SIMPLE_SUMMARY_MODEL") or None,
            decision_fallback_model=(
                env.get("ASKHUB_DECISION_FALLBACK_MODEL", "openai/gpt-oss-20b") or None
            ),
            triage_timeout=float(env.get("ASKHUB_TRIAGE_TIMEOUT", "4")),
            describe_model=env.get("ASKHUB_DESCRIBE_MODEL") or None,
            describe_restricted_model=env.get("ASKHUB_DESCRIBE_RESTRICTED_MODEL") or None,
            describe_project=env.get("ASKHUB_DESCRIBE_PROJECT", "").lower() in ("1", "true", "yes"),
            ranking_provider_order=tuple(
                # ";" as well as ",": gcloud --set-env-vars splits values on commas.
                p.strip() for p in re.split(r"[,;]", env.get("OPENROUTER_RANKING_PROVIDER_ORDER", ""))
                if p.strip()
            ),
            substitution_sites=tuple(
                site.strip() for site in env.get("ASKHUB_SUBSTITUTION_SITES", "").split(",")
                if site.strip()
            ),
            describe_count=int(env.get("ASKHUB_DESCRIBE_COUNT", "8")),
            restriction_route_kinds=tuple(
                kind.strip() for kind in
                env.get("ASKHUB_RESTRICTION_ROUTE_KINDS", "gluten,dairy,nuts").split(",") if kind.strip()
            ),
            llm_provider_sort=env.get("OPENROUTER_LLM_PROVIDER_SORT", "throughput") or None,
            ranking_max_tokens=integer("ASKHUB_RANKING_MAX_TOKENS", 100),
            embedding_model=env.get(
                "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
            ),
            app_url=env.get("OPENROUTER_APP_URL") or None,
            app_title=env.get("OPENROUTER_APP_TITLE", "ask-samples"),
            data_dir=Path(env.get("ASKHUB_DATA_DIR", "data")),
            sources_manifest=Path(env.get("ASKHUB_SOURCES_MANIFEST", "sources.yaml")),
            refresh_interval_hours=float(env.get("ASKHUB_REFRESH_INTERVAL_HOURS", "4")),
            cache_dir=(
                Path(value) if (value := env.get("ASKHUB_CACHE_DIR", "").strip()) else None
            ),
            gcs_bucket=env.get("ASKHUB_GCS_BUCKET", "").strip() or None,
            gcs_snapshot_prefix=env.get("ASKHUB_GCS_SNAPSHOT_PREFIX", "snapshot/"),
            cosmos_endpoint=env.get("ASKHUB_COSMOS_ENDPOINT") or None,
            cosmos_key=env.get("ASKHUB_COSMOS_KEY") or None,
            cosmos_database=env.get("ASKHUB_COSMOS_DATABASE", "samples"),
            cosmos_container=env.get("ASKHUB_COSMOS_CONTAINER", "items"),
            cosmos_partition=env.get("ASKHUB_COSMOS_PARTITION") or None,
            vector_url=env.get("ASKHUB_VECTOR_URL", "").strip() or None,
            vector_token=env.get("ASKHUB_VECTOR_TOKEN") or None,
            snapshot_dir=(
                Path(value) if (value := env.get("ASKHUB_SNAPSHOT_DIR", "").strip()) else None
            ),
            host=host,
            port=port,
            bm25_rank_count=integer("ASKHUB_BM25_RANK_COUNT", 10),
            vector_rank_count=integer("ASKHUB_VECTOR_RANK_COUNT", 10),
            max_retrieval_count=integer("ASKHUB_MAX_RETRIEVAL_COUNT", 200),
            max_ranking_count=integer("ASKHUB_MAX_RANKING_COUNT", 200),
            ranking_batch_count=integer("ASKHUB_RANKING_BATCH_COUNT", 3),
            ranking_batch_return_count=integer("ASKHUB_RANKING_BATCH_RETURN_COUNT", 3),
            max_results=integer("ASKHUB_MAX_RESULTS", 10),
            min_score=integer("ASKHUB_MIN_SCORE", 70),
            strong_score_threshold=integer("ASKHUB_STRONG_SCORE_THRESHOLD", 90),
            decontextualize_timeout=float(
                env.get("ASKHUB_DECONTEXTUALIZE_TIMEOUT", "15")
            ),
            query_expansion=env.get("ASKHUB_QUERY_EXPANSION", "true").lower()
            in {"1", "true", "yes", "on"},
            expansion_timeout=float(env.get("ASKHUB_EXPANSION_TIMEOUT", "6")),
            expansion_term_count=integer("ASKHUB_EXPANSION_TERM_COUNT", 10),
            rate_limit_per_client_per_minute=integer("ASKHUB_RATE_LIMIT_PER_CLIENT_PER_MINUTE", 10),
            rate_limit_global_per_hour=integer("ASKHUB_RATE_LIMIT_GLOBAL_PER_HOUR", 60),
            rate_limit_global_per_day=integer("ASKHUB_RATE_LIMIT_GLOBAL_PER_DAY", 500),
            max_concurrent_queries=integer("ASKHUB_MAX_CONCURRENT_QUERIES", 4),
            cors_origins=values("ASKHUB_CORS_ORIGINS"),
            mcp_allowed_hosts=values("ASKHUB_MCP_ALLOWED_HOSTS", local_mcp_hosts),
            trusted_proxy_ips=values("ASKHUB_TRUSTED_PROXY_IPS", "127.0.0.1,::1"),
            ranking_fields=values("ASKHUB_RANKING_FIELDS", ",".join(DEFAULT_RANKING_FIELDS)),
            bm25_field_weights=weights("ASKHUB_BM25_FIELD_WEIGHTS", DEFAULT_BM25_FIELD_WEIGHTS),
            bm25_field_b=weights("ASKHUB_BM25_FIELD_B", DEFAULT_BM25_FIELD_B),
            vector_field_weights=weights(
                "ASKHUB_VECTOR_FIELD_WEIGHTS", DEFAULT_VECTOR_FIELD_WEIGHTS
            ),
        )
