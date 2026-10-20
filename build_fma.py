#!/usr/bin/env python3
"""
Dataset público de ejemplo: Free Music Archive (FMA, https://github.com/mdeff/fma) con la MISMA
estructura que el catálogo privado del proyecto, para poder publicarlo.

  venv/bin/python publico/build_fma.py   # -> publico/tracks.parquet (106.574 canciones)

Equivalencias con un catálogo comercial: popularity = escuchas en FMA, preview_url = fila del clip de 30 s en la
réplica pública de fma_large en Hugging Face (`hf://datasets/.../train-00000-of-00171.parquet#17`,
lo lee embed.py en streaming; NULL si la réplica no la tiene), genres = géneros de FMA (con sus padres). Los géneros
grandes salen de las mismas MACRO_RULES. Columnas extra: `license` (licencia CC de cada canción) y
`audio_row` (fila global en la réplica: el clip se puede pedir a la API datasets-server de HF).
Metadatos de FMA: CC BY 4.0. Audio: la licencia que eligió cada artista (columna `license`).
"""
import csv
import io
import pathlib
import sys
import urllib.request
import zipfile

import duckdb

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # generos.py: junto a este script (repo) o un nivel arriba (bucket)
from generos import MACRO_RULES  # noqa: E402

URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
WORK = HERE / "work"
OUT = HERE / "tracks.parquet"
# Réplica de fma_large con el audio dentro (Parquet, 171 ficheros). Fijada a un commit para que las
# referencias #fila no cambien.
MIRROR = ("hf://datasets/benjamin-paine/free-music-archive-large@d4cb3e133a7c5a007ddd20458aa30aef8968cf27"
          "/data/*.parquet")

WORK.mkdir(exist_ok=True)
zpath = WORK / "fma_metadata.zip"
if not zpath.exists():
    print("descargando fma_metadata.zip (342 MB) ...", flush=True)
    urllib.request.urlretrieve(URL, zpath)
with zipfile.ZipFile(zpath) as z:
    for f in ("tracks.csv", "genres.csv", "raw_tracks.csv"):
        (WORK / f).write_bytes(z.read(f"fma_metadata/{f}"))

# tracks.csv tiene 3 filas de cabecera: (album|artist|set|track) + (campo) + track_id.
with open(WORK / "tracks.csv", newline="") as f:
    top, sub = next(csv.reader(f)), next(csv.reader(io.StringIO(f.readline())))
cols = ["track_id"] + [f"{a}_{b}" for a, b in zip(top[1:], sub[1:])]

con = duckdb.connect()
con.execute(f"""CREATE TABLE raw AS FROM read_csv('{WORK / 'tracks.csv'}', skip=3, header=false,
                all_varchar=true, names={cols})""")
con.execute(f"CREATE TABLE gen AS FROM read_csv('{WORK / 'genres.csv'}')")

# Fila de la réplica de cada canción. Leer solo las columnas de texto de la réplica tarda ~2 min:
# se guarda en work/mirror.parquet.
mirror = WORK / "mirror.parquet"
if not mirror.exists():
    print("leyendo el índice de la réplica de fma_large ...", flush=True)
    con.execute(f"""COPY (SELECT filename || '#' || file_row_number AS preview_url, url, title, artist, listens
                        FROM read_parquet('{MIRROR}', filename=true, file_row_number=true)) TO '{mirror}'""")
con.execute(f"""
    CREATE TABLE au AS
    WITH m AS (SELECT *, replace(url, 'https:', 'http:') AS u,
                      row_number() OVER (ORDER BY split_part(preview_url, '#', 1),
                                         split_part(preview_url, '#', 2)::INT) - 1 AS audio_row
               FROM '{mirror}'),
         rt AS (SELECT track_id::VARCHAR AS track_id, replace(track_url, 'https:', 'http:') AS u
                FROM read_csv('{WORK / 'raw_tracks.csv'}')),
         by_url AS (SELECT rt.track_id, any_value((m.preview_url, m.audio_row)) AS a
                    FROM m JOIN rt USING (u) WHERE u <> '' GROUP BY 1),
         -- ~1.000 filas de la réplica no tienen url: por (título, artista, escuchas), solo si es única.
         by_meta AS (SELECT r.track_id, any_value((m.preview_url, m.audio_row)) AS a
                     FROM m JOIN raw r ON m.title = r.track_title AND m.artist = r.artist_name
                                      AND m.listens = r.track_listens::BIGINT
                     WHERE m.url = '' AND r.track_id NOT IN (SELECT track_id FROM by_url)
                     GROUP BY 1 HAVING count(*) = 1)
    SELECT track_id, a[1] AS preview_url, a[2] AS audio_row FROM (FROM by_url UNION ALL FROM by_meta)""")
con.execute("CREATE TABLE rules (prio INT, macro_genre TEXT, rx TEXT)")
con.executemany("INSERT INTO rules VALUES (?, ?, ?)", [(i, m, rx) for i, (m, rx) in enumerate(MACRO_RULES)])
con.execute("""
    CREATE TABLE t AS
    WITH g AS (  -- géneros de cada canción, en minúsculas como los de Spotify (hip-hop -> hip hop)
      SELECT r.track_id, list(lower(replace(gen.title, '-', ' ')) ORDER BY gen."#tracks" DESC) AS genres
      FROM (SELECT track_id, unnest(string_split(trim(track_genres_all, '[]'), ', ')) AS gid FROM raw) r
      JOIN gen ON gen.genre_id::VARCHAR = r.gid GROUP BY 1
    ), gm AS (
      SELECT g.genre, arg_min(r.macro_genre, r.prio) AS macro_genre, min(r.prio) AS prio
      FROM (SELECT DISTINCT unnest(genres) AS genre FROM g) g
      JOIN rules r ON regexp_matches(g.genre, r.rx) GROUP BY 1
    ), tm AS (   -- mismo criterio que el catálogo privado: el género grande que más se repite
      SELECT track_id, list(macro_genre ORDER BY n DESC, prio) AS macro_genres FROM (
        SELECT x.track_id, gm.macro_genre, count(*) AS n, min(gm.prio) AS prio
        FROM (SELECT track_id, unnest(genres) AS genre FROM g) x JOIN gm USING (genre) GROUP BY 1, 2)
      GROUP BY 1
    )
    SELECT r.track_id, r.track_title AS name, [r.artist_name] AS artists, r.artist_name AS main_artist,
           [r.artist_id] AS artist_ids, NULL::VARCHAR AS isrc, r.track_listens::BIGINT AS popularity,
           r.track_duration::BIGINT * 1000 AS duration_ms, NULL::BOOLEAN AS explicit,
           au.preview_url,
           r.album_title AS album, coalesce(r.album_date_released, r.track_date_created)[:10] AS release_date,
           []::VARCHAR[] AS genres_spotify, []::VARCHAR[] AS genres_mb_recording, []::VARCHAR[] AS genres_mb_artist,
           NULL::VARCHAR AS mb_source, coalesce(g.genres, []) AS genres,
           lower(replace(r.track_genre_top, '-', ' ')) AS primary_genre,
           lower(r.artist_name) || '|' || lower(r.track_title) AS song_key, 1::BIGINT AS n_versions,
           tm.macro_genres[1] AS macro_genre, tm.macro_genres, r.track_license AS license, au.audio_row
    FROM raw r LEFT JOIN g USING (track_id) LEFT JOIN tm USING (track_id) LEFT JOIN au USING (track_id)
""")
con.execute(f"""COPY (SELECT * EXCLUDE (license, audio_row), CASE WHEN macro_genre IS NOT NULL THEN row_number() OVER
                        (PARTITION BY macro_genre ORDER BY popularity DESC, track_id::INT) END AS genre_rank,
                        license, audio_row
                      FROM t ORDER BY macro_genre NULLS LAST, genre_rank, popularity DESC, track_id::INT)
                TO '{OUT}' (FORMAT parquet, COMPRESSION zstd)""")
for f in ("tracks.csv", "genres.csv", "raw_tracks.csv"):
    (WORK / f).unlink()

print(con.sql(f"""SELECT count(*) AS canciones, count(macro_genre) AS con_genero, count(preview_url) AS con_audio,
                        count(DISTINCT preview_url) AS audios_distintos FROM '{OUT}'"""))
print(con.sql(f"SELECT macro_genre, count(*) AS canciones, max(popularity) AS max_escuchas FROM '{OUT}' GROUP BY 1 ORDER BY 2 DESC"))
print(con.sql(f"SELECT license, count(*) AS n FROM '{OUT}' GROUP BY 1 ORDER BY 2 DESC LIMIT 8"))
