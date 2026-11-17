# qdfeeds — Query Defined Feeds

A natural-language query is a feed. The server keeps a corpus of upstream RSS/Atom
items and, when a feed is fetched, returns the newest items that match the query.

    GET /feed.xml?q=<query>[&since=<date>][&limit=N][&threshold=relevant|strong]

The server keeps state about the feeds (items, fetch validators, embeddings) and
nothing about users: no saved queries, no subscriber tokens, no watermarks. The URL
is the whole definition of a feed.

## Run

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env            # then add your key, see below
qdrss
```

Open <http://127.0.0.1:8000/>: type a query, get the feed URL and a preview of what it
returns now. The URL encodes the query; there is nothing to save on the server.
`/health` shows item count, index age, and per-source fetch status.

### This American Life by section

`/tal` searches *This American Life* in passages of about two minutes and says where each one is: episode, act, and the
second it starts at. The feed has one item per episode and cannot; the site's timed transcripts
(`thisamericanlife.org/<n>/transcript`) can: a block per act, and a start time on every paragraph. `src/qdrss/tal.py` splits an
episode into passages (an Item each, with its act, offsets, transcript anchor and an audio link at `#t=<seconds>` in the
item's `extra` field, emitted by `/feed.xml` as `qd:offsetSeconds`, `qd:act`, `qd:audio` ...). The latest 15 episodes arrive
through the feed (a source with `segmenter: tal`); `scripts/backfill_tal.py` adds the other ~880 once (about 28,000 passages).
The transcripts' act names are the transcribers' and can lag a section (in #899 the block labelled Prologue holds the
introduction and Part One); the offsets come from the paragraph times and are right.

## Keys and models

All model calls go through [OpenRouter](https://openrouter.ai), one key for both
embeddings and ranking. Put it in `.env` (git-ignored) or in the environment:

```bash
OPENROUTER_API_KEY=sk-or-...            # https://openrouter.ai/settings/keys
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_RANKING_MODEL=openai/gpt-oss-20b
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
OPENROUTER_RANKING_PROVIDER_SORT=throughput
OPENROUTER_WRITING_MODEL=openai/gpt-oss-120b   # search page: follow-up rewrites and summaries
```

Without a key the server falls back to hash embeddings and a keyword ranker: the
plumbing works, the relevance doesn't. Don't judge results in that mode.

**Ranking model.** The ranker classifies each candidate strong / relevant / exclude
and returns JSON. Any OpenRouter chat model that honours `response_format:
json_object` works. Open-weight options, cheapest first:

| Model | Notes |
|---|---|
| `openai/gpt-oss-20b` | default; measured ~1 s per batch of 5 with `provider.sort=throughput` (10 s without) |
| `openai/gpt-oss-120b` | same family, better judgement, ~3x the price |
| `google/gemma-3-27b-it` | fast, no reasoning tokens |
| `meta-llama/llama-3.3-70b-instruct` | solid at structured output |
| `mistralai/mistral-small-3.2-24b-instruct` | cheap, fast |
| `deepseek/deepseek-v3.2` | strongest of the open models here; slower |

Only `gpt-oss-20b` has been measured on this workload. Two things matter for reasoning
models like gpt-oss: the request sets `reasoning.effort=low` and a generous
`max_tokens`, because a tight cap makes them spend the budget thinking and return
empty content (which this server reports as a 503, not an empty feed).

**Embedding model.** `openai/text-embedding-3-small` (1536 dims) is what the shipped
corpus was embedded with. Embeddings are cached per item per model, so changing
`OPENROUTER_EMBEDDING_MODEL` re-embeds every item on the next refresh; at 57k items
that is about $0.15 and 15 minutes with text-embedding-3-small, and the old vectors
stay in the store keyed by their model name. Any model OpenRouter's `/embeddings`
endpoint serves can be named here; open-weight embedding models come and go on
OpenRouter, so check their catalogue before switching.

**Cost.** A feed fetch is one embedding call (cached per query) plus
`candidate_count / batch_size` ranking calls per page of 40 (8 by default; up to 40
when few items qualify and every page of the 200 is judged). With gpt-oss-20b that is
about 0.2 cents per uncached fetch, so **1,000 uncached fetches cost about $2**.
Rendered feeds are cached per (query, collection, limit) until the next refresh and
served with an ETag, so repeat polls of the same URL cost nothing and readers that
send `If-None-Match` get a 304. The corpus embedding is a one-time cost per model.

## How a fetch works

1. `since` (or, when absent, now minus `QDRSS_WINDOW_DAYS`) masks the index to items
   **ingested** after that time (and to one collection, if `collection=` is given).
   The mask is applied before top-K, so the candidates are the best *new* items,
   not the new items among the best overall.
2. Vector retrieval over the masked items yields up to `QDRSS_MAX_CANDIDATES` (200)
   candidates, closest first. There is no lexical lane.
3. The ranking model classifies the first `QDRSS_CANDIDATE_COUNT` (40) as `strong`,
   `relevant`, or excluded; while fewer than `limit` qualify it classifies the next 40,
   up to all of them. No per-batch quota. If the model call fails the response is 503, not an
   empty feed.
4. Survivors at or above `threshold` are ordered newest-first by publication time (ingestion
   time for an item with none; item id as tie-break) and capped at `limit`.

## The NYT search page as a conversation

`/nyt/search` is a conversation, kept in the browser (IndexedDB) and listed on the left; a
conversation that starts with a question already in the list replaces the older one. Each
question is one `POST /api/search` with the tab, the conversation's earlier queries and the
last turn's result titles, the answer mode (`summarize` or list) and an optional ranking
model from `/api/models`. The harness then:

1. rewrites a follow-up to stand alone (one call to the writing model, `converse.py`; a failed
   rewrite is said on the page and the question is searched as typed);
2. searches the tab exactly as a feed does, with the chosen ranking model (Cooking and
   Wirecutter ask their NLWeb backends, passing the mode and model on);
3. streams JSON lines: `query`, `results`, then `summary` (one call to the writing model over
   the top 12) or `summary_error`, then `usage`: calls, tokens, model time and dollars by model
   and step, as OpenRouter reported them (`usage.py`; an NLWeb backend's own usage is merged in).

A repeated question is served from the cache and its usage says so. The feed URL a turn offers
is for the rewritten query and always ranks with the default model.

## Semantics to know

- **`since` is on ingestion time, not publication time.** Each item carries
  `<qd:ingestedAt>`; a client that passes the newest one back as `since` gets exactly
  what arrived after it, including late-published items. The first import of a
  corpus stamps every existing item "now", so a fresh server shows its whole back
  catalog as new for one window.
- **A capped response is not an exact delta.** The pool is bounded (40 candidates)
  and the output is capped, so a client advancing `since` past a full page may skip
  matches. Readers that fetch the fixed URL get the trailing window each time and
  dedupe on GUID; that is the intended mode.
- **Items are append-only.** First stored version wins; upstream edits and deletions
  are ignored, and items that roll off a feed stay in the corpus.
- **Identity** is `<source>:<guid>`, else the link, else a hash of title + raw date.
  GUIDs are namespaced by source because upstream ids are only unique per feed.
- `pubDate` is the source publication time. `<qd:category>` and `<qd:source>` carry
  the verdict and the source name.

## Storage: Postgres on the VM (production), Cosmos DB (before), or SQLite

**Production (2026-11-17): no Microsoft service in the path.** Items, vectors and fetch state live in
Postgres on the neuralweb VM (`pgstore.py`, schema `pg_schema.sql`; `QDRSS_PG_HOST=10.128.0.2`,
`QDRSS_PG_PASSWORD`). The ingest job and the web app reach it through Direct VPC egress; the
VM's vector service catches up from it. Postgres does no vector search. A `pg_dump` goes to
`gs://neuralweb-vm-backup/postgres/` nightly and the snapshots are rebuilt from Postgres
(`deploy/vm/`). The Cosmos container was exported (`scripts/export_cosmos.py`) and loaded
(`scripts/load_postgres.py`), the last delta at the cutover. The Cosmos notes below are history.

## Cosmos DB or SQLite (before the cutover)

Two modes behind one interface, chosen by whether `COSMOS_ENDPOINT` is set.

- **Cosmos DB for NoSQL** (production): one `items` container partitioned by
  collection holds each episode with its embedding; the feed's nearest-neighbour
  query runs in Cosmos with the date and collection filters, over a `quantizedFlat`
  vector index (DiskANN's filtered recall was poor at this size).
  The web process holds nothing. The free tier (1000 RU/s, 25 GB) covers tens of
  thousands of episodes. `qdrss-ingest` fetches feeds, stores new items and embeds
  them; run it on a schedule anywhere with the same `.env`, or leave
  `QDRSS_INGEST_IN_APP=true` and the web app runs it daily itself, as a subprocess.
  `scripts/migrate_sqlite_to_cosmos.py` loads an existing SQLite corpus, vectors
  included, without re-embedding.
- **SQLite** (default, tests, offline): items and cached vectors in one file, an
  in-memory matrix rebuilt after each refresh.

**Collections served from memory.** Cosmos vector queries over the NYT collections took
0.5–20 s each (2026-11-17), so the collections the search page reads
(`QDRSS_MEMORY_COLLECTIONS=tal,nytnews,nytimes,american_football,soccer`) are loaded at
startup from snapshots in `QDRSS_MEMORY_SNAPSHOT` (`gs://neuralweb-qdrss-snapshots`) and
searched exactly with numpy, off the event loop. A tab over several collections searches each
and merges by score. Whatever was ingested after a snapshot is fetched from Cosmos at
startup and every five minutes, appended into spare room in the vector buffer. Cosmos stays the
store that ingest writes to. Rebuild snapshots from a Cosmos export with
`scripts/snapshot_from_export.py` (an old snapshot only lengthens the catch-up).

In production this runs on the neuralweb VM, not in Cloud Run: `qdrss-vectors` (`vectors.py`,
a systemd user unit, `~/.config/systemd/user/qdrss-vectors.service`, env in
`~/.config/qdrss-vectors.env`) holds the collections (~6 GB) and answers `POST /search` on the
VM's internal address 10.128.0.2:8090 with a shared token. The web app has
`QDRSS_VECTOR_URL`/`QDRSS_VECTOR_TOKEN` set and reaches it through Direct VPC egress
(`--network default --subnet default --vpc-egress private-ranges-only`); if the VM does not
answer the search fails with 503 "vector search unavailable" (no fallback to Cosmos) and
`/health` counts it (`vector_service.failures`).
Deploy the VM side with `git pull && .venv/bin/pip install . && systemctl --user restart
qdrss-vectors` in `~/services/qdrss-vectors`; it takes about a minute to load.

## Sources

`sources.yaml` lists the upstream feeds, grouped into collections. They are fetched at
startup and every `QDRSS_REFRESH_MINUTES` (1440: once a day) with ETag/Last-Modified.
New episodes are inserted by id, so nothing is ever duplicated; episodes that drop
out of a feed stay in the store; and only items without a vector are embedded, so a
refresh costs embeddings for the new episodes alone; servers that ignore those are caught
by a body hash. A malformed or failing feed keeps its previous items and validators
and does not block the others. On a source's first fetch, ingest follows RFC 5005 `rel="next"` archive pages so a
paginated feed's whole history lands in the store; later fetches read page one only.
A feed the publisher truncates (some NYT and TWiT shows) yields only what it carries.
The index is rebuilt after every refresh from the SQLite store; only items without a
cached embedding are embedded.

## Tests

```bash
pytest
```

Fake providers throughout; the conditional-GET paths use `httpx.MockTransport`.

## License

Copyright (C) 2026 the QD Feeds authors. This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. See [LICENSE](LICENSE).
