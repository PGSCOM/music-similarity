#!/usr/bin/env python3
# /// script
# dependencies = ["huggingface_hub>=1.5", "duckdb"]
# ///
"""
Spotify previews (30 s MP3s) -> a public Hugging Face bucket, one download per second, without
touching the disk: each MP3 is kept in memory and uploaded in batches as <hash>.mp3 at the bucket
root (<hash> is the one in https://p.scdn.co/mp3-preview/<hash>). The parquet is only read.

  uv run download_previews.py privado/tracks_recortado.parquet --bucket PGSCOM/spotify-previews
  uv run download_previews.py privado/tracks_recortado.parquet --limit 3      # quick test

(`uv run` because the venv's huggingface_hub is 0.x, which has no buckets.)

Log: one line per preview, `<hash>\t<status>\t<UTC time>` (status = ok once it is in the bucket,
or the HTTP code of a dead preview). It is written only after the upload succeeds, it makes the
script resumable and it is also uploaded to the bucket as log.tsv, so embed.py --bucket knows which
files exist. If the local log is missing (other machine), it starts from the bucket's copy.
"""
import argparse
import datetime
import pathlib
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import duckdb
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("parquet", type=pathlib.Path)
ap.add_argument("--bucket", default="PGSCOM/spotify-previews")
ap.add_argument("--log", type=pathlib.Path, help="default: work/previews.tsv next to the parquet")
ap.add_argument("--limit", type=int)
ap.add_argument("--rate", type=float, default=1.0, help="downloads per second")
ap.add_argument("--batch", type=int, default=100, help="MP3s per upload (~360 KB each, kept in RAM)")
ap.add_argument("--log-every", type=int, default=10, help="upload log.tsv every N batches")
a = ap.parse_args()

api = HfApi()
api.create_bucket(a.bucket, private=False, exist_ok=True)
log = a.log or a.parquet.parent / "work" / "previews.tsv"
log.parent.mkdir(parents=True, exist_ok=True)
if not log.exists():
    try:
        urllib.request.urlretrieve(f"https://huggingface.co/buckets/{a.bucket}/resolve/log.tsv", log)
        print(f"log restored from the bucket into {log}")
    except urllib.error.HTTPError:
        log.touch()

logged = (f"SELECT column0 FROM read_csv('{log}', header = false, delim = '\t', columns = "
          "{'column0': 'VARCHAR', 'column1': 'VARCHAR', 'column2': 'VARCHAR'})") if log.stat().st_size else "SELECT ''"
# Several tracks can share a preview: one per hash, in the parquet's order (genre, genre_rank).
todo = duckdb.sql(f"""
    SELECT h FROM (
        SELECT regexp_extract(preview_url, 'mp3-preview/([0-9a-f]+)', 1) AS h, min(file_row_number) AS n
        FROM read_parquet('{a.parquet}', file_row_number = true)
        WHERE preview_url IS NOT NULL GROUP BY h)
    WHERE h <> '' AND h NOT IN ({logged})
    ORDER BY n {f'LIMIT {a.limit}' if a.limit else ''}""").fetchall()
print(f"{len(todo):,} previews to download into hf://buckets/{a.bucket}", flush=True)


def now():
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def upload(files, dead, n):
    for attempt in range(5):
        try:
            if files:
                api.batch_bucket_files(a.bucket, add=[(data, f"{h}.mp3") for h, data in files])
            break
        except Exception as e:
            print(f"  upload failed ({e}), retrying in {60 * 2 ** attempt} s", flush=True)
            time.sleep(60 * 2 ** attempt)
    else:
        raise SystemExit("upload keeps failing: stopping (nothing of this batch was logged)")
    with open(log, "a") as f:
        f.writelines(f"{h}\tok\t{now()}\n" for h, _ in files)
        f.writelines(f"{h}\t{code}\t{now()}\n" for h, code in dead)
    if n % a.log_every == 0:
        api.batch_bucket_files(a.bucket, add=[(str(log), "log.tsv")])


def fetch(h):
    """MP3 bytes, or the HTTP code if the preview is gone (404...). Retries throttling and network errors."""
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"https://p.scdn.co/mp3-preview/{h}", timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                return e.code
            err = e
        except Exception as e:
            err = e
        print(f"  {h}: {err}, waiting {30 * 2 ** attempt} s", flush=True)
        time.sleep(30 * 2 ** attempt)
    return None  # not logged: it is retried on the next run


files, dead, batches, ok = [], [], 0, 0
nxt = time.monotonic()
# One upload in the background while the next batch downloads, so the uploads do not slow the rate.
with ThreadPoolExecutor(1) as pool:
    pending = pool.submit(lambda: None)
    try:
        for i, (h,) in enumerate(todo, 1):
            time.sleep(max(0.0, nxt - time.monotonic()))
            nxt = max(nxt, time.monotonic()) + 1 / a.rate
            r = fetch(h)
            if isinstance(r, bytes):
                files.append((h, r))
                ok += 1
            elif r is not None:
                dead.append((h, r))
            if len(files) + len(dead) >= a.batch:
                pending.result()
                batches += 1
                pending = pool.submit(upload, files, dead, batches)
                files, dead = [], []
            if i % 1000 == 0:
                print(f"{i:,}/{len(todo):,} ({ok:,} ok)", flush=True)
    finally:  # also on Ctrl+C: upload what is already in memory and the final log
        pending.result()
        upload(files, dead, a.log_every)
print(f"done: {ok:,} MP3s in hf://buckets/{a.bucket}, log in {log}")
