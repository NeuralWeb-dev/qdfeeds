# Architecture: the harness owns the workflow

*2026-11-16, revision `qdrss-00041-r7t`. For the step-by-step trace of one request with
measurements, see [life-of-a-query.md](life-of-a-query.md).*

## The approach

The code (the harness) runs every step of a search and decides what happens next. The language model
is never asked to plan, choose a tool or loop. It is called for one narrow judgment at a time,
many times, in parallel: *is this article about this question?*

## A search, step by step

1. **Embed the question.** One call to `openai/text-embedding-3-small` through OpenRouter, about 0.3 s,
   cached per question so a feed reader polling the same URL costs nothing (`src/qdrss/feed.py:34`).
2. **Retrieve, with no model.** A vector search limited to the tab's sources returns up to
   `QDRSS_MAX_CANDIDATES` (200) candidates, closest first. Code drops repeats of a URL
   (`one_per_url`, `feed.py:103`) and reorders by closeness blended with recency (`by_blend`, `feed.py:123`).
3. **Judge, in small calls.** Candidates go to the ranking model (`openai/gpt-oss-20b`, reasoning effort
   low, `src/qdrss/providers.py:161`) five per call. Each call returns one short JSON entry per article:
   strong, relevant or exclude, with a one-sentence reason (`src/qdrss/rank.py:17`, `rank.py:36`). The
   eight calls for a page of 40 run at once, so a page takes about 1.5 s.
4. **Decide whether to go on.** If fewer than `limit` (20) have qualified, the code sends the next 40,
   up to all 200 (`evaluate`, `feed.py:58`).
5. **Assemble.** Code sorts the matches newest first, caps them at the limit, and renders JSON for the
   search page, RSS for feed readers, or the email newsletter (`search_api`, `src/qdrss/app.py:312`).

Cooking and Wirecutter tabs call the NLWeb backends instead. Ingest is a separate scheduled Cloud Run
job (`qdrss-ingest`) that fetches the feeds and embeds new items in batches.

## What the harness owns

- The order of the steps, the stopping rule, the candidate limits and the time window.
- Retries: one retry per failed batch, and skipping OpenRouter providers that keep failing.
- Failure handling: a failed ranking returns HTTP 503, never an empty result list.
- Caching of query vectors and finished results.

## Why this way

- **Cheap and fast.** A small model on tiny inputs, run in parallel: about 0.2 cents per 40
  articles judged. Total time is set by the slowest call, not by a long chain of reasoning.
- **Predictable.** The same steps run for every query, so cost and latency have an upper limit
  (at most 40 ranking calls).
- **Debuggable.** Every decision can be traced. When "what was Bessent's role in Jensen Huang's
  case" found nothing under All, the trace showed the answer was 41st of 40 candidates: the cutoff
  was at fault, not the model. The fix (judging in pages) was a code change, not a prompt change.
- **Replaceable parts.** The embedding model, the ranking model, the store (Cosmos now, Postgres
  next) and the retrieval method can each change without touching the others.

## Limits

- **The model judges only the text it is given.** Most items carry the publisher's RSS summary,
  often one sentence. A 172-character summary of "Bessent Faults Commerce Dept. for Failing to
  Blacklist Megaspeed" never says Megaspeed is part of the Huang story, so a stricter prompt
  rejected it. Longer descriptions are the fix, not prompt wording.
- **A batch biases the judgment.** The model judges an article relative to the other four in its
  call: an article that merely names Bessent was rated strong when nothing else in its batch did.
- **Meaning alone misses rare names.** One vector per article captures what it is mostly about, so
  a question about Bessent and Huang lands nearer fifty Huang articles than the one about both.
  Matching on the words themselves (full-text search alongside the vectors) is the planned fix.
