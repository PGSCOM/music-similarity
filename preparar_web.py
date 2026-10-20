#!/usr/bin/env python3
"""
Prepara los datos de la demo estática (web/):

  pip install umap-learn scikit-learn
  python preparar_web.py [--datos carpeta]

  web/emb.bin    vectores en int8 (N×512, escala fija ESCALA; el ranking casi no cambia: ~98 % del top-10)
  web/mapa.bin   posición 2D de cada canción (UMAP sobre los embeddings), uint16 x, y
  web/meta.json  artista, título, género, licencia, fila del audio en la réplica y escuchas; mismo orden
"""
import argparse
import json
import pathlib

import numpy as np

import buscar

ESCALA = 670  # 127 / percentil 99,999 de |valor|: recorta solo valores extremos
WEB = pathlib.Path(__file__).parent / "web"


def mapa(emb):
    import umap
    from sklearn.decomposition import PCA
    x = PCA(50, random_state=0).fit_transform(emb)
    xy = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=0, low_memory=True).fit_transform(x)
    lo, hi = np.percentile(xy, [0.1, 99.9], axis=0)  # unos pocos puntos sueltos encogerían el resto
    xy = np.clip((xy - lo) / (hi - lo), 0, 1)
    return np.round(xy * 65535).astype(np.uint16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", help="carpeta con tracks.parquet y embeddings.parquet (por defecto: el dataset de HF)")
    a = ap.parse_args()
    tracks, ids, emb = buscar.cargar(a.datos)

    np.clip(np.round(emb * ESCALA), -127, 127).astype(np.int8).tofile(WEB / "emb.bin")
    meta = tracks.select("track_id, main_artist, name, coalesce(macro_genre, '-') AS genero, coalesce(license, '-') AS license, "
                         "audio_row, coalesce(popularity, 0) AS popularity").fetchnumpy()
    pos = {t: i for i, t in enumerate(meta["track_id"])}
    orden = [pos[t] for t in ids]
    generos, licencias = sorted(set(meta["genero"])), sorted(set(meta["license"]))
    gi, li = {g: i for i, g in enumerate(generos)}, {l: i for i, l in enumerate(licencias)}
    filas = [[str(meta["main_artist"][i]), str(meta["name"][i]), gi[meta["genero"][i]],
              li[meta["license"][i]], int(meta["audio_row"][i]), int(meta["popularity"][i])] for i in orden]
    (WEB / "meta.json").write_text(json.dumps(
        {"escala": ESCALA, "dim": emb.shape[1], "generos": generos, "licencias": licencias,
         "campos": ["artista", "titulo", "genero", "licencia", "fila_audio", "escuchas"], "filas": filas},
        ensure_ascii=False, separators=(",", ":")))
    mapa(emb).tofile(WEB / "mapa.bin")
    print(len(ids), "canciones ->", WEB)
