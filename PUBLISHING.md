# Publishing

The project is published as three artifacts, each in its own repository:

| Artifact | Where | Local folder |
|---|---|---|
| Browser demo (static Space) | [huggingface.co/spaces/PGSCOM/MusicSearcher](https://huggingface.co/spaces/PGSCOM/MusicSearcher) | `MusicSearcher/` |
| Dataset (tracks + embeddings) | [huggingface.co/datasets/PGSCOM/fma-muq-mulan](https://huggingface.co/datasets/PGSCOM/fma-muq-mulan) | `fma-muq-mulan/` |
| Code and pipeline | [github.com/PGSCOM/music-similarity](https://github.com/PGSCOM/music-similarity) | `music-similarity/` |

The Space was renamed from `buscador-musical` to `MusicSearcher`; the old URL still redirects.

## Rebuilding the dataset

```bash
cd music-similarity
python build_fma.py                    # -> tracks.parquet (106,574 songs, ~2 min)
python embed.py tracks.parquet         # -> embeddings/part-*.parquet (a GPU is recommended)
```

`build_fma.py` writes `tracks.parquet` in this folder; `embed.py` writes `embeddings/part-*.parquet`.
Copy the result into the dataset repository and push it:

```bash
cp tracks.parquet ../fma-muq-mulan/tracks.parquet
cat embeddings/part-*.parquet > ../fma-muq-mulan/embeddings.parquet
cd ../fma-muq-mulan && git add -A && git commit -m "..." && git push
```

## Rebuilding the app data

```bash
cd music-similarity
python prepare_web.py                  # -> web/emb.bin, web/map.bin, web/meta.json
python export_web.py web/model         # -> web/model/mel.onnx + web/model/audio_q4f16.onnx
```

`prepare_web.py` reads `tracks.parquet` + `embeddings.parquet` (from the dataset by default, or
from a local folder with `--data`). It writes the int8 vectors, the UMAP map and `meta.json`.

Copy the results into the Space repository and push (the big files go through Git LFS):

```bash
cp web/emb.bin web/map.bin web/meta.json ../MusicSearcher/
mkdir -p ../MusicSearcher/model && cp web/model/*.onnx ../MusicSearcher/model/
cd ../MusicSearcher && git add -A && git commit -m "..." && git push
```

## Pushing the code

```bash
cd music-similarity && git push
```

## Things that must stay in sync

- **The macro genres.** `genres.py` is the single source of truth for the 18 macro genres and it
  feeds `build_fma.py` (`macro_genre`, `macro_genres`). If you change a name there, rebuild
  `tracks.parquet` *and* the app data: the app reads the genre names from `meta.json`, which
  `prepare_web.py` derives from the dataset.
- **`web/index.html` and the Space copy.** The Space serves `index.html` verbatim; keep the file in
  `music-similarity/web/` and the one in `MusicSearcher/` identical (the Space repository is the
  deployed copy).
- **The cards.** The dataset card lives in `music-similarity/dataset/README.md` and is copied to
  `fma-muq-mulan/README.md`; the Space card lives in `music-similarity/web/README.md` and is
  copied to `MusicSearcher/README.md`. The YAML front matter at the top is what the Hub renders.

## Licenses

- Code: MIT.
- MuQ-MuLan weights: CC BY-NC 4.0, so the embeddings and the demo are non-commercial only.
- FMA metadata: CC BY 4.0. Audio is not redistributed: it is streamed from the public
  [`benjamin-paine/free-music-archive-large`](https://huggingface.co/datasets/benjamin-paine/free-music-archive-large)
  mirror, and each song keeps the license of its artist.
