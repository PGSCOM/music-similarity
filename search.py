#!/usr/bin/env python3
"""
Find similar songs using the MuQ-MuLan embeddings (similarity = dot product).

  python search.py --song "Jahzzar - Siesta"      # similar to a song in the catalogue
  python search.py --text "sad piano ballad"       # by description (loads the model)
  python search.py --audio my_clip.mp3 -k 20       # by any audio file (loads the model)

Data: tracks.parquet + embeddings.parquet from --data (a local folder) or, by default, from the
public Hugging Face dataset. Also importable as a module (prepare_web.py uses it).
"""
import argparse
import pathlib

import duckdb
import numpy as np
import pyarrow.parquet as pq

DATASET = "PGSCOM/fma-muq-mulan"
MODEL = "OpenMuQ/MuQ-MuLan-large"
SR = 24_000


def load(data=None):
    """-> (tracks: DuckDB relation, ids: array of track_id, emb: N x 512 normalized matrix)."""
    if data:
        tp, ep = pathlib.Path(data) / "tracks.parquet", pathlib.Path(data) / "embeddings.parquet"
    else:
        from huggingface_hub import hf_hub_download
        tp, ep = (hf_hub_download(DATASET, f, repo_type="dataset") for f in ("tracks.parquet", "embeddings.parquet"))
    t = pq.read_table(ep)
    emb = np.asarray(t.column("embedding").combine_chunks().flatten(), dtype=np.float32).reshape(t.num_rows, -1)
    tracks = duckdb.read_parquet(str(tp))
    return tracks, np.asarray(t.column("track_id").to_pylist()), emb


_model = None


def model():
    global _model
    if _model is None:
        import torch
        from muq import MuQMuLan
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _model = MuQMuLan.from_pretrained(MODEL).to(dev).eval()
    return _model


def text_vector(q):
    import torch
    with torch.inference_mode():
        v = model()(texts=[q])[0].float().cpu().numpy()
    return v / np.linalg.norm(v)


def audio_vector(path):
    import librosa
    import torch
    wav, _ = librosa.load(path, sr=SR, mono=True, duration=30)  # same 30 s as embed.py
    if len(wav) < 10 * SR:  # MuQ splits into 10 s windows: short clips are repeated
        wav = np.tile(wav, -(-10 * SR // len(wav)))
    with torch.inference_mode():
        v = model()(wavs=[torch.from_numpy(wav).to(model().device)])[0].float().cpu().numpy()
    return v / np.linalg.norm(v)


def find(tracks, query):
    """'artist - title' (or part of it) -> track_id of the most played match, or None."""
    r = tracks.filter(f"(main_artist || ' - ' || name) ILIKE $$%{query.replace('$', '')}%$$ "
                      "AND audio_row IS NOT NULL").order("popularity DESC").limit(1).fetchone()
    return r and r[0]


def neighbors(tracks, ids, emb, v, k=10, exclude=None):
    """The k songs closest to vector v -> list of dicts sorted by similarity."""
    sim = emb @ v
    top = np.argsort(-sim)[: k + 1]
    top = [i for i in top if ids[i] != exclude][:k]
    pos = {ids[i]: (n, float(sim[i])) for n, i in enumerate(top)}
    rows = tracks.filter(f"track_id IN ({', '.join(repr(str(ids[i])) for i in top)})") \
                 .select("track_id, name, main_artist, macro_genre, license, audio_row").fetchall()
    cols = ["track_id", "name", "main_artist", "macro_genre", "license", "audio_row"]
    out = [dict(zip(cols, r), similarity=pos[r[0]][1]) for r in rows]
    return sorted(out, key=lambda d: -d["similarity"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--song", help="'artist - title' or part of it")
    q.add_argument("--id", help="exact track_id")
    q.add_argument("--text")
    q.add_argument("--audio", type=pathlib.Path)
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--data", help="folder with tracks.parquet and embeddings.parquet (default: the HF dataset)")
    a = ap.parse_args()

    tracks, ids, emb = load(a.data)
    exclude = None
    if a.song or a.id:
        exclude = a.id or find(tracks, a.song)
        idx = np.flatnonzero(ids == exclude)
        if not len(idx):
            raise SystemExit(f"Cannot find '{a.song or a.id}' among the songs with an embedding")
        name, artist = tracks.filter(f"track_id = '{exclude}'").select("name, main_artist").fetchone()
        print(f"Similar to: {artist} - {name}\n")
        v = emb[idx[0]]
    else:
        v = text_vector(a.text) if a.text else audio_vector(a.audio)
    for n, r in enumerate(neighbors(tracks, ids, emb, v, a.k, exclude), 1):
        print(f"{n:3}. {r['similarity']:.3f}  {r['main_artist']} - {r['name']}  [{r['macro_genre'] or '-'}]")
