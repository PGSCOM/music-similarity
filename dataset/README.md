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

# FMA + embeddings MuQ-MuLan

Las 106.574 canciones del [Free Music Archive](https://github.com/mdeff/fma) con metadatos
ordenados por género y popularidad, y el embedding de audio de cada una (512 dimensiones,
[MuQ-MuLan-large](https://huggingface.co/OpenMuQ/MuQ-MuLan-large)) para buscar canciones parecidas.
Código: [github.com/PGSCOM/music-similarity](https://github.com/PGSCOM/music-similarity) ·
Demo: [PGSCOM/buscador-musical](https://huggingface.co/spaces/PGSCOM/buscador-musical).

## Ficheros

**`tracks.parquet`**: una fila por canción.
- `popularity`: escuchas en FMA. `genres`: géneros de FMA (con sus padres). `macro_genre`: uno de
  18 géneros grandes. `genre_rank`: puesto por escuchas dentro de su género. Las filas van
  ordenadas por `(macro_genre, genre_rank)`.
- `preview_url` y `audio_row`: dónde está el clip de 30 s en la réplica
  [`benjamin-paine/free-music-archive-large`](https://huggingface.co/datasets/benjamin-paine/free-music-archive-large)
  (fichero#fila y fila global). NULL en las ~1.550 canciones que la réplica no tiene.
- `license`: licencia Creative Commons elegida por el artista.

**`embeddings.parquet`**: `track_id` + `embedding` (512 floats, L2-normalizado; similitud =
producto escalar). Media de las 3 ventanas de 10 s del clip. Son comparables con embeddings de
texto del mismo modelo (`MuQMuLan(texts=[...])`).

## Uso

```python
import duckdb
duckdb.sql("""
  WITH q AS (SELECT embedding FROM 'hf://datasets/PGSCOM/fma-muq-mulan/embeddings.parquet' WHERE track_id = '62460')
  SELECT t.main_artist, t.name, list_dot_product(e.embedding, q.embedding) AS similitud
  FROM 'hf://datasets/PGSCOM/fma-muq-mulan/embeddings.parquet' e
  JOIN 'hf://datasets/PGSCOM/fma-muq-mulan/tracks.parquet' t USING (track_id), q
  ORDER BY similitud DESC LIMIT 10""")
```

O con `buscar.py` del repo: `python buscar.py --cancion "Jahzzar - Siesta"`.

## Licencias

- Metadatos de FMA: CC BY 4.0 (Defferrard et al., *FMA: A Dataset for Music Analysis*, ISMIR 2017).
- Audio: no se incluye; cada canción tiene la licencia de su artista (columna `license`).
- Embeddings: generados con pesos CC BY-NC 4.0, así que este dataset es de **uso no comercial**.
