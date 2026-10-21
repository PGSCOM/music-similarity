#!/usr/bin/env python3
"""
Public example dataset: Free Music Archive (FMA, https://github.com/mdeff/fma) with the SAME
structure as the private catalogue of the project, so that it can be published.

  venv/bin/python build_fma.py   # -> tracks.parquet (106,574 songs)

Equivalences with a commercial catalogue: popularity = FMA play count, preview_url = row of the
30 s clip in the public mirror of fma_large on Hugging Face (`hf://datasets/.../train-00000-of-00171.parquet#17`,
read by embed.py in streaming; NULL when the mirror does not have it), genres = FMA genres (with their
parents). The macro genres come from the same MACRO_RULES. Extra columns: `license` (the CC license of
each song) and `audio_row` (global row in the mirror: the clip can be fetched from the HF datasets-server API).
FMA metadata: CC BY 4.0. Audio: the license chosen by each artist (column `license`).
"""
import csv
import io
import pathlib
import sys
import urllib.request
import zipfile

import duckdb

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # genres.py: next to this script (repo) or one level up (bucket)
from genres import MACRO_RULES  # noqa: E402

URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
WORK = HERE / "work"
OUT = HERE / "tracks.parquet"
# Mirror of fma_large with the audio inside (Parquet, 171 files). Pinned to a commit so that the
# #row references never change.
MIRROR = ("hf://datasets/benjamin-paine/free-music-archive-large@d4cb3e133a7c5a007ddd20458aa30aef8968cf27"
          "/data/*.parquet")

WORK.mkdir(exist_ok=True)
zpath = WORK / "fma_metadata.zip"
if not zpath.exists():
    print("downloading fma_metadata.zip (342 MB) ...", flush=True)
    urllib.request.urlretrieve(URL, zpath)
with zipfile.ZipFile(zpath) as z:
    for f in ("tracks.csv", "genres.csv", "raw_tracks.csv"):
        (WORK / f).write_bytes(z.read(f"fma_metadata/{f}"))

# tracks.csv has 3 header rows: (album|artist|set|track) + (field) + track_id.
with open(WORK / "tracks.csv", newline="") as f:
    top, sub = next(csv.reader(f)), next(csv.reader(io.StringIO(f.readline())))
    cols = ["track_id"] + [f"{a}_{b}" for a, b in zip(top[1:], sub[1:])]

con = duckdb.connect()
con.execute(f"""CREATE TABLE raw AS FROM read_csv('{WORK / 'tracks.csv'}', skip=3, header=false,
                all_varchar=true, names={cols})""")
con.execute(f"CREATE TABLE gen AS FROM read_csv('{WORK / 'genres.csv'}')")

# Row of the mirror for each song. Reading only the text columns of the mirror takes ~2 min:
# it is cached in work/mirror.parquet.
mirror = WORK / "mirror.parquet"
if not mirror.exists():
    print("reading the index of the fma_large mirror ...", flush=True)
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
         -- ~1.000 rows of the mirror have no url: matched by (title, artist, plays), only if unique.
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
    WITH g AS (  -- genres of each song, lowercased like the Spotify ones (hip-hop -> hip hop)
      SELECT r.track_id, list(lower(replace(gen.title, '-', ' ')) ORDER BY gen."#tracks" DESC) AS genres
      FROM (SELECT track_id, unnest(string_split(trim(track_genres_all, '[]'), ', ')) AS gid FROM raw) r
      JOIN gen ON gen.genre_id::VARCHAR = r.gid GROUP BY 1
    ), gm AS (
      SELECT g.genre, arg_min(r.macro_genre, r.prio) AS macro_genre, min(r.prio) AS prio
      FROM (SELECT DISTINCT unnest(genres) AS genre FROM g) g
      JOIN rules r ON regexp_matches(g.genre, r.rx) GROUP BY 1
    ), tm AS (   # same criterion as the private catalogue: the macro genre that repeats the most
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

print(con.sql(f"""SELECT count(*) AS songs, count(macro_genre) AS with_genre, count(preview_url) AS with_audio,
                        count(DISTINCT preview_url) AS distinct_audios FROM '{OUT}'"""))
print(con.sql(f"SELECT macro_genre, count(*) AS songs, max(popularity) AS max_plays FROM '{OUT}' GROUP BY 1 ORDER BY 2 DESC"))
print(con.sql(f"SELECT license, count(*) AS n FROM '{OUT}' GROUP BY 1 ORDER BY 2 DESC LIMIT 8"))
