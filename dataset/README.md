---
license: cc-by-nc-4.0
pretty_name: FMA + MuQ-MuLan embeddings (music similarity)
task_categories:
  - audio-classification
  - feature-extraction
tags:
  - music
  - music-similarity
  - embeddings
  - fma
  - muq-mulan
size_categories:
  - 100K<n<1M
configs:
  - config_name: tracks
    data_files: tracks.parquet
  - config_name: embeddings
    data_files: embeddings.parquet
---

# FMA + MuQ-MuLan embeddings

The 106,574 songs of the [Free Music Archive](https://github.com/mdeff/fma) with metadata sorted by
genre and popularity, plus the audio embedding of each one (512 dimensions,
[MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)) to search for similar songs.
Code: [github.com/PGSCOM/music-similarity](https://github.com/PGSCOM/music-similarity) ·
Demo: [PGSCOM/MusicSearcher](https://huggingface.co/spaces/PGSCOM/MusicSearcher).

## Files

**`tracks.parquet`**: one row per song.
- `popularity`: play count in FMA. `genres`: FMA genres (with their parents). `macro_genre`: one of
  18 macro genres. `genre_rank`: position by play count within its genre. The rows are sorted by
  `(macro_genre, genre_rank)`.
- `preview_url` and `audio_row`: where the 30 s clip lives in the
  [`benjamin-paine/free-music-archive-large`](https://huggingface.co/datasets/benjamin-paine/free-music-archive-large)
  mirror (file#row and global row). NULL for the ~1,550 songs the mirror does not have.
- `license`: the Creative Commons license chosen by the artist.

**`embeddings.parquet`**: `track_id` + `embedding` (512 floats, L2-normalized; similarity = dot
product). Mean of the 3 windows of 10 s of the clip. Comparable with text embeddings from the same
model (`MuQMuLan(texts=[...])`).

## Usage

```python
import duckdb
duckdb.sql("""
  WITH q AS (SELECT embedding FROM 'hf://datasets/PGSCOM/fma-muq-mulan/embeddings.parquet' WHERE track_id = '62460')
  SELECT t.main_artist, t.name, list_dot_product(e.embedding, q.embedding) AS similarity
  FROM 'hf://datasets/PGSCOM/fma-muq-mulan/embeddings.parquet' e
  JOIN 'hf://datasets/PGSCOM/fma-muq-mulan/tracks.parquet' t USING (track_id), q
  ORDER BY similarity DESC LIMIT 10""")
```

Or with `search.py` from the repo: `python search.py --song "Jahzzar - Siesta"`.

## License

- FMA metadata: CC BY 4.0 (Defferrard et al., *FMA: A Dataset for Music Analysis*, ISMIR 2017).
- Audio: not included; each song keeps the license of its artist (column `license`).
- Embeddings: generated with CC BY-NC 4.0 weights, so this dataset is **non-commercial use only**.
