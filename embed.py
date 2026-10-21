#!/usr/bin/env python3
"""
Audio embeddings with MuQ-MuLan-large (512 dim, audio and text in the same space) for a
tracks.parquet of the project (any catalogue with the same structure).

  venv/bin/python embed.py other/tracks.parquet                # preview_url = URL of an MP3
  venv/bin/python embed.py tracks.parquet                       # preview_url = hf://...parquet#row (FMA mirror)
  venv/bin/python embed.py tracks.parquet --limit 3             # quick test
  venv/bin/python embed.py privado/tracks_top15000.parquet --bucket PGSCOM/spotify-previews
      # Spotify previews already copied by download_previews.py: <hash>.mp3 at the bucket root
      # (only those its log.tsv marks ok; the rest wait for a later run instead of failing)

Output: <folder of the parquet>/embeddings/part-NNNNN.parquet with (track_id, embedding FLOAT[512])
L2-normalized: similarity = dot product. Resumable: it skips what is already in embeddings/ and
whatever is listed in embeddings/failed.txt (missing previews or broken MP3s).
"""
import argparse
import io
import os
import itertools
import pathlib
import queue
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import duckdb
import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before importing torch
import torch
from huggingface_hub import HfFileSystem
from muq import MuQMuLan
from tqdm import tqdm

MODEL = "OpenMuQ/MuQ-MuLan-large"
SR = 24_000

ap = argparse.ArgumentParser()
ap.add_argument("parquet", type=pathlib.Path)
ap.add_argument("--out", type=pathlib.Path, help="output folder (default: embeddings/ next to the parquet)")
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=16,
                help="songs per pass (3 windows of 10 s each); with 16 GB of VRAM 16 fits and 64 does not")
ap.add_argument("--workers", type=int, default=16, help="parallel downloads/decodings")
ap.add_argument("--part-rows", type=int, default=20_000)
ap.add_argument("--bucket", help="public HF bucket with the Spotify previews (see download_previews.py)")
a = ap.parse_args()

out = a.out or a.parquet.parent / "embeddings"
out.mkdir(parents=True, exist_ok=True)
failed = out / "failed.txt"
failed.touch()
tracks = f"read_parquet('{a.parquet}')"
if a.bucket:
    base = f"https://huggingface.co/buckets/{a.bucket}/resolve"
    tracks = f"""(SELECT * REPLACE ('{base}/' || h || '.mp3' AS preview_url) FROM (
        SELECT *, regexp_extract(preview_url, 'mp3-preview/([0-9a-f]+)', 1) AS h FROM {tracks})
        JOIN read_csv('{base}/log.tsv', header=false, delim='\t', names=['h', 'status', 'time']) USING (h)
        WHERE status = 'ok')"""
done = f"SELECT track_id FROM read_parquet('{out}/part-*.parquet')" if any(out.glob("part-*.parquet")) else "SELECT ''"
todo = duckdb.sql(f"""
    SELECT track_id, preview_url FROM {tracks}
    WHERE preview_url IS NOT NULL AND track_id NOT IN ({done})
      AND track_id NOT IN (SELECT column0 FROM read_csv('{failed}', header=false, columns={{'column0': 'VARCHAR'}}))
    ORDER BY macro_genre NULLS LAST, genre_rank, popularity DESC
    {f'LIMIT {a.limit}' if a.limit else ''}""").fetchall()
print(f"{len(todo):,} songs to process", flush=True)


def sources(todo):
    """(track_id, url or bytes of the MP3). The rows hf://file.parquet#n are grouped by file and
    read in streaming, fetching only the row groups that are needed (100 songs each in the FMA mirror)."""
    yield from (r for r in todo if not r[1].startswith("hf://"))
    hf = sorted((u.rsplit("#", 1)[0], int(u.rsplit("#", 1)[1]), tid) for tid, u in todo if u.startswith("hf://"))
    fs = HfFileSystem()
    for path, rows in itertools.groupby(hf, key=lambda r: r[0]):
        want = {n: tid for _, n, tid in rows}
        with fs.open(path.removeprefix("hf://")) as f:
            pf, start = pq.ParquetFile(f), 0
            for rg in range(pf.num_row_groups):
                size = pf.metadata.row_group(rg).num_rows
                if any(start <= n < start + size for n in want):
                    for i, audio in enumerate(pf.read_row_group(rg, columns=["audio"]).column("audio").to_pylist()):
                        if start + i in want:
                            yield want[start + i], audio["bytes"]
                start += size


def load(row):
    tid, src = row
    try:
        if isinstance(src, str):
            with urllib.request.urlopen(src, timeout=30) as r:
                src = r.read()
        wav, _ = librosa.load(io.BytesIO(src), sr=SR, mono=True, duration=30)  # the project uses 30 s clips
        if len(wav) <= SR:  # less than 1 s: broken
            return tid, None
        # Less than 10 s: repeat it to fill a window (MuQ only pads once and the window would
        # stay too short to be stacked with the others).
        return tid, np.tile(wav, -(-10 * SR // len(wav))) if len(wav) < 10 * SR else wav
    except Exception as e:
        tqdm.write(f"  failed {tid}: {e}")
        return tid, None


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MuQMuLan.from_pretrained(MODEL).to(dev).eval()
print(f"model on {dev}", flush=True)

ids, vecs = [], []
part = len(list(out.glob("part-*.parquet")))
bar = tqdm(total=len(todo), unit="song", smoothing=0.05, dynamic_ncols=True)
fails = 0


def flush():
    global ids, vecs, part
    if ids:
        pq.write_table(pa.table({"track_id": ids, "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate(vecs).ravel()), 512)}), out / f"part-{part:05d}.parquet")
        part, ids, vecs = part + 1, [], []


def prefetch(gen, size=512):
    """Reads `gen` in another thread: the download of the next block overlaps with the GPU."""
    q, end = queue.Queue(size), object()

    def fill():
        try:
            for x in gen:
                q.put(x)
        except Exception as e:
            q.put(e)
        q.put(end)
    threading.Thread(target=fill, daemon=True).start()
    while (x := q.get()) is not end:
        if isinstance(x, Exception):
            raise x
        yield x


def latents(x):
    """Windows -> embeddings. If it does not fit in the GPU, split the batch in two and retry."""
    try:
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            return model.mulan_module.get_audio_latents(x.to(dev)).float()
    except torch.OutOfMemoryError:
        if len(x) == 1:
            raise
    # Outside the except: inside, the exception keeps the tensors of the failed attempt and the GPU is not freed.
    torch.cuda.empty_cache()
    tqdm.write(f"  out of GPU memory with {len(x)} windows: splitting the batch (lower --batch)")
    h = len(x) // 2
    return torch.cat([latents(x[:h]), latents(x[h:])])


def run(batch):
    global ids
    # All the 10 s windows of the batch in a single pass; then the mean per song.
    clips = [model._get_all_clips(torch.from_numpy(w)) for _, w in batch]
    lat = latents(torch.cat(clips))
    emb = torch.stack([c.mean(0) for c in lat.split([len(c) for c in clips])])
    vecs.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy().astype(np.float32))
    ids += [t for t, _ in batch]
    bar.update(len(batch))
    if len(ids) >= a.part_rows:
        flush()


with ThreadPoolExecutor(a.workers) as pool:
    batch = []
    # In blocks: pool.map queues everything at once and would pile up audio if the model is slower.
    src = prefetch(sources(todo))
    rows = (r for chunk in iter(lambda: list(itertools.islice(src, a.batch * 4)), []) for r in pool.map(load, chunk))
    for tid, wav in rows:
        if wav is None:
            with open(failed, "a") as f:
                f.write(f"{tid}\n")
            fails += 1
            bar.update(1)
            bar.set_postfix(failed=fails)
            continue
        batch.append((tid, wav))
        if len(batch) == a.batch:
            run(batch)
            batch = []
    if batch:
        run(batch)
flush()
bar.close()
print(f"done: {len(todo) - fails:,} embeddings in {out}, {fails:,} failures (in {failed.name})")
