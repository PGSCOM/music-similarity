#!/usr/bin/env python3
"""
Prepares the data files of the static demo (web/):

  pip install umap-learn scikit-learn
  python prepare_web.py [--data folder]

  web/emb.bin    vectors as int8 (N x 512, fixed SCALE; the ranking barely changes: ~98 % of the top-10)
  web/map.bin    2D position of each song (UMAP over the embeddings), uint16 x, y
  web/meta.json  artist, title, genre, license, audio row in the mirror and play count; same order
"""
import argparse
import json
import pathlib

import numpy as np

import search

SCALE = 670  # 127 / 99.999 percentile of |value|: only extreme values are clipped
WEB = pathlib.Path(__file__).parent / "web"


def layout(emb):
    import umap
    from sklearn.decomposition import PCA
    x = PCA(50, random_state=0).fit_transform(emb)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0, low_memory=True).fit_transform(x)
    lo, hi = np.percentile(xy, [0.1, 99.9], axis=0)  # a few stray points would shrink everything else
    xy = np.clip((xy - lo) / (hi - lo), 0, 1)
    return np.round(xy * 65535).astype(np.uint16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="folder with tracks.parquet and embeddings.parquet (default: the HF dataset)")
    a = ap.parse_args()
    tracks, ids, emb = search.load(a.data)

    np.clip(np.round(emb * SCALE), -127, 127).astype(np.int8).tofile(WEB / "emb.bin")
    meta = tracks.select("track_id, main_artist, name, coalesce(macro_genre, '-') AS genre, coalesce(license, '-') AS license, "
                         "audio_row, coalesce(popularity, 0) AS popularity").fetchnumpy()
    pos = {t: i for i, t in enumerate(meta["track_id"])}
    order = [pos[t] for t in ids]
    genres, licenses = sorted(set(meta["genre"])), sorted(set(meta["license"]))
    gi, li = {g: i for i, g in enumerate(genres)}, {l: i for i, l in enumerate(licenses)}
    rows = [[str(meta["main_artist"][i]), str(meta["name"][i]), gi[meta["genre"][i]],
             li[meta["license"][i]], int(meta["audio_row"][i]), int(meta["popularity"][i])] for i in order]
    (WEB / "meta.json").write_text(json.dumps(
        {"scale": SCALE, "dim": emb.shape[1], "genres": genres, "licenses": licenses,
         "fields": ["artist", "title", "genre", "license", "audio_row", "plays"], "rows": rows},
        ensure_ascii=False, separators=(",", ":")))
    layout(emb).tofile(WEB / "map.bin")
    print(len(ids), "songs ->", WEB)
