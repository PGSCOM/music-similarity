#!/bin/sh
# Worker: downloads the source parquet once (with the account that can read it) and then copies
# its Spotify previews into the public bucket of the other account (HF_TOKEN). Extra arguments go
# to download_previews.py, e.g. `docker compose run --rm worker --limit 30`.
set -e
if [ ! -f /data/tracks.parquet ]; then
    HF_TOKEN="$HF_SOURCE_TOKEN" hf buckets cp "$SOURCE" /data/tracks.parquet.part
    mv /data/tracks.parquet.part /data/tracks.parquet
fi
exec python download_previews.py /data/tracks.parquet --bucket "$BUCKET" --log /data/previews.tsv "$@"
