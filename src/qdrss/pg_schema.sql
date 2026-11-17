-- qdrss in Postgres (replaces the Cosmos container `items`). Idempotent: run at every start.
-- Vectors are float32 (vector(1536)), as Cosmos held them; search is exact, in memory
-- (see memory.py), so there is no vector index here.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS items (
    id              text PRIMARY KEY,          -- "<source>:<guid | link | hash>" (Item.id)
    source          text NOT NULL,
    source_title    text NOT NULL DEFAULT '',
    collection      text NOT NULL,
    url             text NOT NULL DEFAULT '',
    title           text NOT NULL DEFAULT '',
    snippet         text NOT NULL DEFAULT '',
    content         text NOT NULL DEFAULT '',
    published_at    timestamptz,
    published_raw   text NOT NULL DEFAULT '',
    ingested_at     timestamptz NOT NULL,      -- first stored; never changes
    extra           text NOT NULL DEFAULT '',
    embedding       vector(1536),
    embedding_model text,
    modified_at     timestamptz NOT NULL DEFAULT now()  -- last insert or vector update: what catch-up reads
);
CREATE INDEX IF NOT EXISTS items_collection_source ON items (collection, source);
CREATE INDEX IF NOT EXISTS items_modified ON items (modified_at);
CREATE INDEX IF NOT EXISTS items_ingested ON items (collection, ingested_at);
CREATE INDEX IF NOT EXISTS items_unembedded ON items (ingested_at) WHERE embedding IS NULL;

CREATE TABLE IF NOT EXISTS sources (
    name          text PRIMARY KEY,
    url           text NOT NULL,
    etag          text,
    last_modified text,
    sha256        text,
    last_fetch    timestamptz,
    last_error    text
);
