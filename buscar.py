#!/usr/bin/env python3
"""
Busca canciones parecidas con los embeddings de MuQ-MuLan (similitud = producto escalar).

  python buscar.py --cancion "Jahzzar - Siesta"      # parecidas a una canción del catálogo
  python buscar.py --texto "piano triste y lento"     # por descripción (carga el modelo)
  python buscar.py --audio mi_clip.mp3 -k 20          # por un audio cualquiera (carga el modelo)

Datos: tracks.parquet + embeddings.parquet de --datos (carpeta local) o, por defecto, del dataset
público de Hugging Face. También se importa como módulo (lo usa preparar_web.py).
"""
import argparse
import pathlib

import duckdb
import numpy as np
import pyarrow.parquet as pq

DATASET = "PGSCOM/fma-muq-mulan"
MODEL = "OpenMuQ/MuQ-MuLan-large"
SR = 24_000


def cargar(datos=None):
    """-> (tracks: tabla DuckDB, ids: array de track_id, emb: matriz N×512 normalizada)."""
    if datos:
        tp, ep = pathlib.Path(datos) / "tracks.parquet", pathlib.Path(datos) / "embeddings.parquet"
    else:
        from huggingface_hub import hf_hub_download
        tp, ep = (hf_hub_download(DATASET, f, repo_type="dataset") for f in ("tracks.parquet", "embeddings.parquet"))
    t = pq.read_table(ep)
    emb = np.asarray(t.column("embedding").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
    tracks = duckdb.read_parquet(str(tp))
    return tracks, np.asarray(t.column("track_id").to_pylist()), emb


_model = None


def modelo():
    global _model
    if _model is None:
        import torch
        from muq import MuQMuLan
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = MuQMuLan.from_pretrained(MODEL).to(dev).eval()
    return _model


def vec_texto(q):
    import torch
    with torch.inference_mode():
        v = modelo()(texts=[q])[0].float().cpu().numpy()
    return v / np.linalg.norm(v)


def vec_audio(path):
    import librosa
    import torch
    wav, _ = librosa.load(path, sr=SR, mono=True, duration=30)  # mismos 30 s que embed.py
    if len(wav) < 10 * SR:  # MuQ trocea en ventanas de 10 s: los audios cortos se repiten
        wav = np.tile(wav, -(-10 * SR // len(wav)))
    with torch.inference_mode():
        v = modelo()(wavs=[torch.from_numpy(wav).to(modelo().device)])[0].float().cpu().numpy()
    return v / np.linalg.norm(v)


def encontrar(tracks, consulta):
    """'artista - título' (o parte) -> track_id de la coincidencia más escuchada, o None."""
    r = tracks.filter(f"(main_artist || ' - ' || name) ILIKE $$%{consulta.replace('$', '')}%$$ "
                      "AND audio_row IS NOT NULL").order("popularity DESC").limit(1).fetchone()
    return r and r[0]


def vecinos(tracks, ids, emb, v, k=10, excluir=None):
    """Las k canciones más parecidas al vector v -> lista de dicts ordenada por similitud."""
    sim = emb @ v
    top = np.argsort(-sim)[: k + 1]
    top = [i for i in top if ids[i] != excluir][:k]
    pos = {ids[i]: (n, float(sim[i])) for n, i in enumerate(top)}
    rows = tracks.filter(f"track_id IN ({', '.join(repr(str(ids[i])) for i in top)})") \
                 .select("track_id, name, main_artist, macro_genre, license, audio_row").fetchall()
    cols = ["track_id", "name", "main_artist", "macro_genre", "license", "audio_row"]
    out = [dict(zip(cols, r), similitud=pos[r[0]][1]) for r in rows]
    return sorted(out, key=lambda d: -d["similitud"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--cancion", help="'artista - título' o parte")
    q.add_argument("--id", help="track_id exacto")
    q.add_argument("--texto")
    q.add_argument("--audio", type=pathlib.Path)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--datos", help="carpeta con tracks.parquet y embeddings.parquet (por defecto: el dataset de HF)")
    a = ap.parse_args()

    tracks, ids, emb = cargar(a.datos)
    excluir = None
    if a.cancion or a.id:
        excluir = a.id or encontrar(tracks, a.cancion)
        idx = np.flatnonzero(ids == excluir)
        if not len(idx):
            raise SystemExit(f"No encuentro '{a.cancion or a.id}' entre las canciones con embedding")
        name, artist = tracks.filter(f"track_id = '{excluir}'").select("name, main_artist").fetchone()
        print(f"Parecidas a: {artist} - {name}\n")
        v = emb[idx[0]]
    else:
        v = vec_texto(a.texto) if a.texto else vec_audio(a.audio)
    for n, r in enumerate(vecinos(tracks, ids, emb, v, a.k, excluir), 1):
        print(f"{n:3}. {r['similitud']:.3f}  {r['main_artist']} - {r['name']}  [{r['macro_genre'] or '-'}]")
