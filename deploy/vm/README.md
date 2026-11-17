# The qdrss services on the neuralweb VM

Systemd user units (`~/.config/systemd/user/`, the user lingers so they run without a login):

- `qdrss-vectors.service`: the vector service (`qdrss-vectors`, src/qdrss/vectors.py) on 10.128.0.2:8090, every
  collection in memory, caught up from Postgres every five minutes.
- `qdrss-nightly.timer` / `.service`: `qdrss-nightly.sh`, a `pg_dump` of the qdrss database to
  `gs://neuralweb-vm-backup/postgres/` (14 days kept, `latest.dump` the newest) and fresh snapshots to
  `gs://neuralweb-qdrss-snapshots`.

Both read `~/.config/qdrss-vectors.env` (mode 600): QDRSS_PG_HOST, QDRSS_PG_PASSWORD, QDRSS_VECTOR_TOKEN,
QDRSS_MEMORY_COLLECTIONS, QDRSS_MEMORY_SNAPSHOT, QDRSS_VECTOR_HOST, QDRSS_VECTOR_PORT, QDRSS_SOURCES.
The code runs from a clean clone, `~/services/qdrss-vectors`:

    cd ~/services/qdrss-vectors && git pull && .venv/bin/pip install . && systemctl --user restart qdrss-vectors

Restore a backup: `gcloud storage cp gs://neuralweb-vm-backup/postgres/latest.dump - | pg_restore -h ... -U qdrss -d qdrss --clean`.
