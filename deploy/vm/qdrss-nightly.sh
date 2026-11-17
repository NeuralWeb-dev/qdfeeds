#!/usr/bin/env bash
# Nightly on the neuralweb VM (qdrss-nightly.timer):
#  1. back up the qdrss Postgres database to gs://neuralweb-vm-backup/postgres/ (kept 14 days by the bucket's
#     lifecycle rule; `latest.dump` is always the newest);
#  2. rebuild the vector service's in-memory snapshots from Postgres into gs://neuralweb-qdrss-snapshots, so a
#     restart of qdrss-vectors loads a day-old corpus and catches up little.
# Env: QDRSS_PG_HOST, QDRSS_PG_PASSWORD (~/.config/qdrss-vectors.env).
set -euo pipefail
export PATH="$HOME/google-cloud-sdk/bin:/usr/bin:/bin"
REPO="$HOME/services/qdrss-vectors"
DAY="$(date -u +%F)"
BACKUP="gs://neuralweb-vm-backup/postgres"

echo "backup: pg_dump qdrss -> $BACKUP/qdrss-$DAY.dump"
PGPASSWORD="$QDRSS_PG_PASSWORD" pg_dump -h "$QDRSS_PG_HOST" -U qdrss -d qdrss -Fc -Z 6 \
  | gcloud storage cp --quiet - "$BACKUP/qdrss-$DAY.dump"
gcloud storage cp --quiet "$BACKUP/qdrss-$DAY.dump" "$BACKUP/latest.dump"
gcloud storage ls -l "$BACKUP/qdrss-$DAY.dump"

echo "snapshots: Postgres -> gs://neuralweb-qdrss-snapshots"
"$REPO/.venv/bin/python" "$REPO/scripts/snapshot_from_pg.py" "$HOME/services/snapshots" \
  --upload gs://neuralweb-qdrss-snapshots
echo "done"
