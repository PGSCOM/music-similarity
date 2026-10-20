#!/usr/bin/env python3
"""
Embeddings de audio con MuQ-MuLan-large (512 dim, audio y texto en el mismo espacio) para un
tracks.parquet del proyecto (cualquier catálogo con la misma estructura).

  venv/bin/python embed.py otro/tracks.parquet                # preview_url = URL de un MP3
  venv/bin/python embed.py publico/tracks.parquet             # preview_url = hf://...parquet#fila (réplica de FMA)
  venv/bin/python embed.py publico/tracks.parquet --limit 3   # prueba rápida

Salida: <carpeta del parquet>/embeddings/part-NNNNN.parquet con (track_id, embedding FLOAT[512])
L2-normalizado: similitud = producto escalar. Reanudable: salta lo que ya está en embeddings/ y lo
apuntado en embeddings/fallidos.txt (previews caídas o MP3 rotos).
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # antes de importar torch
import torch
from huggingface_hub import HfFileSystem
from muq import MuQMuLan
from tqdm import tqdm

MODEL = "OpenMuQ/MuQ-MuLan-large"
SR = 24_000

ap = argparse.ArgumentParser()
ap.add_argument("parquet", type=pathlib.Path)
ap.add_argument("--out", type=pathlib.Path, help="carpeta de salida (por defecto embeddings/ junto al parquet)")
ap.add_argument("--limit", type=int)
ap.add_argument("--batch", type=int, default=16,
                help="canciones por pasada (3 ventanas de 10 s cada una); con 16 GB de VRAM, 16 va bien y 64 no cabe")
ap.add_argument("--workers", type=int, default=16, help="descargas/decodificaciones en paralelo")
ap.add_argument("--part-rows", type=int, default=20_000)
a = ap.parse_args()

out = a.out or a.parquet.parent / "embeddings"
out.mkdir(parents=True, exist_ok=True)
failed = out / "fallidos.txt"
failed.touch()
done = f"SELECT track_id FROM read_parquet('{out}/part-*.parquet')" if any(out.glob("part-*.parquet")) else "SELECT ''"
todo = duckdb.sql(f"""
    SELECT track_id, preview_url FROM read_parquet('{a.parquet}')
    WHERE preview_url IS NOT NULL AND track_id NOT IN ({done})
      AND track_id NOT IN (SELECT column0 FROM read_csv('{failed}', header=false, columns={{'column0': 'VARCHAR'}}))
    ORDER BY macro_genre NULLS LAST, genre_rank, popularity DESC
    {f'LIMIT {a.limit}' if a.limit else ''}""").fetchall()
print(f"{len(todo):,} canciones por procesar", flush=True)


def sources(todo):
    """(track_id, url o bytes del MP3). Las filas hf://fichero.parquet#n se agrupan por fichero y se
    leen en streaming solo los row groups que hacen falta (100 canciones cada uno en la réplica de FMA)."""
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
        wav, _ = librosa.load(io.BytesIO(src), sr=SR, mono=True, duration=30)  # el proyecto usa fragmentos de 30 s
        if len(wav) <= SR:  # menos de 1 s: roto
            return tid, None
        # Menos de 10 s: se repite hasta llenar una ventana (MuQ solo rellena una vez y la ventana
        # quedaría corta para apilarla con las demás).
        return tid, np.tile(wav, -(-10 * SR // len(wav))) if len(wav) < 10 * SR else wav
    except Exception as e:
        tqdm.write(f"  fallo {tid}: {e}")
        return tid, None


dev = "cuda" if torch.cuda.is_available() else "cpu"
model = MuQMuLan.from_pretrained(MODEL).to(dev).eval()
print(f"modelo en {dev}", flush=True)

ids, vecs = [], []
part = len(list(out.glob("part-*.parquet")))
bar = tqdm(total=len(todo), unit="canción", smoothing=0.05, dynamic_ncols=True)
fails = 0


def flush():
    global ids, vecs, part
    if ids:
        pq.write_table(pa.table({"track_id": ids, "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(np.concatenate(vecs).ravel()), 512)}), out / f"part-{part:05d}.parquet")
        part, ids, vecs = part + 1, [], []


def prefetch(gen, size=512):
    """Lee `gen` en otro hilo: la descarga del siguiente bloque se solapa con la GPU."""
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
    """Ventanas -> embeddings. Si no cabe en la GPU, parte el lote en dos y reintenta."""
    try:
        with torch.inference_mode(), torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            return model.mulan_module.get_audio_latents(x.to(dev)).float()
    except torch.OutOfMemoryError:
        if len(x) == 1:
            raise
    # Fuera del except: dentro, la excepción retiene los tensores del intento fallido y la GPU no se libera.
    torch.cuda.empty_cache()
    tqdm.write(f"  sin memoria en la GPU con {len(x)} ventanas: parto el lote (baja --batch)")
    h = len(x) // 2
    return torch.cat([latents(x[:h]), latents(x[h:])])


def run(batch):
    global ids
    # Todas las ventanas de 10 s del lote en una sola pasada; luego la media por canción.
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
    # Por bloques: pool.map encola todo de golpe y acumularía audio si el modelo va más lento.
    src = prefetch(sources(todo))
    rows = (r for chunk in iter(lambda: list(itertools.islice(src, a.batch * 4)), []) for r in pool.map(load, chunk))
    for tid, wav in rows:
        if wav is None:
            with open(failed, "a") as f:
                f.write(f"{tid}\n")
            fails += 1
            bar.update(1)
            bar.set_postfix(fallos=fails)
            continue
        batch.append((tid, wav))
        if len(batch) == a.batch:
            run(batch)
            batch = []
    if batch:
        run(batch)
flush()
bar.close()
print(f"hecho: {len(todo) - fails:,} embeddings en {out}, {fails:,} fallos (en {failed.name})")
