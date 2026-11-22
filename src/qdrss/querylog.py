"""One structured line per query answered, for the private query log.

Each line is a JSON object on stdout. Cloud Run hands stdout to Cloud Logging, which stores a JSON
line as the entry's jsonPayload, so the log viewer finds every query with `jsonPayload.log="query"`.
Cloud Run's own request log has the URL of a feed fetch but not the body of a search-page turn
(POST /api/search), so without this line those queries are recorded nowhere.

A line carries what was asked and how it went: the endpoint, vertical, query (and the standalone
rewrite of a follow-up), status, result count, latency, model spend and the caller's user agent.
Never the caller's address, any other header, or anything from the configuration. Feed readers
polling a standing feed are logged too, marked "client": "program" by their user agent.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from typing import Any, TextIO

QUERY_CHARS = 500
TEXT_CHARS = 200


def client_kind(user_agent: str) -> str:
    """"browser" for a person's browser, "program" for anything else (a feed reader on a schedule,
    a script), so the viewer can tell people from pollers."""
    return "browser" if "mozilla/" in user_agent.lower() else "program"


def emit(record: dict[str, Any], stream: TextIO | None = None) -> None:
    """Write one query-log line. A failure to log never fails the query."""
    user_agent = str(record.get("user_agent") or "")
    line = {
        "log": "query",
        "severity": "WARNING" if str(record.get("status")) in ("error", "500", "503") else "INFO",
        "service": os.environ.get("K_SERVICE") or "qdrss",
        "revision": os.environ.get("K_REVISION"),
        **record,
        "query": str(record.get("query") or "")[:QUERY_CHARS],
        "user_agent": user_agent[:TEXT_CHARS],
        "client": client_kind(user_agent),
    }
    if line.get("error"):
        line["error"] = str(line["error"])[:TEXT_CHARS]
    line["message"] = f"query {line.get('status')}: {line['query'][:80]}"
    text = json.dumps(line, ensure_ascii=False, default=str)
    with contextlib.suppress(Exception):  # logging must not break the answer
        print(text, file=stream or sys.stdout, flush=True)
