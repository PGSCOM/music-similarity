#!/usr/bin/env python3
"""
Summary of a project database (any tracks*.parquet in this folder or its subfolders).

  venv/bin/python analyze.py                        # shows what is there and asks which one to scan
  venv/bin/python analyze.py tracks.parquet
"""
import pathlib
import sys

import duckdb

HERE = pathlib.Path(__file__).resolve().parent

if len(sys.argv) > 1:
    db = pathlib.Path(sys.argv[1])
else:
    dbs = sorted([*HERE.glob("tracks*.parquet"), *HERE.glob("*/tracks*.parquet")])
    if not dbs:
        sys.exit("There is no tracks*.parquet here nor in the subfolders")
    for i, p in enumerate(dbs, 1):
        print(f"  {i}) {p.relative_to(HERE)}  ({p.stat().st_size / 1e9:.2f} GB)")
    db = dbs[int(input("Which one do I scan? ")) - 1]

con = duckdb.connect()
con.execute("SET memory_limit='4GB'; SET threads=4")
con.execute(f"CREATE VIEW t AS FROM read_parquet('{db}')")
total, = con.execute("SELECT count(*) FROM t").fetchone()
print(f"\n{db}: {total:,} songs, {db.stat().st_size / 1e9:.2f} GB")

cols = con.execute("SELECT column_name, column_type FROM (DESCRIBE t)").fetchall()
filled = con.execute("SELECT " + ", ".join(
    f"count(*) FILTER ({c} IS NOT NULL{' AND len(' + c + ') > 0' if ty.endswith('[]') else ''})" for c, ty in cols)
    + " FROM t").fetchone()
print("\nColumns (with data):")
for (c, ty), n in zip(cols, filled):
    print(f"  {c:22s} {ty:12s} {n / total:6.1%}")

print("\nBy macro genre:")
con.sql("""SELECT coalesce(macro_genre, '(no genre)') AS macro_genre, count(*) AS songs,
                        count(DISTINCT main_artist) AS artists,
                        count(*) FILTER (preview_url IS NOT NULL) AS with_audio,
                        median(popularity) AS median_popularity, max(popularity) AS max_popularity
                 FROM t GROUP BY 1 ORDER BY 2 DESC""").show(max_rows=50)

print("Popularity (percentiles 50/90/99/max):",
      con.execute("SELECT quantile_cont(popularity, [0.5, 0.9, 0.99]), max(popularity) FROM t").fetchone())

print("\nBy release decade:")
con.sql("""SELECT try_cast(left(release_date, 4) AS INT) // 10 * 10 AS decade, count(*) AS songs
           FROM t GROUP BY 1 ORDER BY 1 NULLS LAST""").show(max_rows=40)

print("Artists with the most songs:")
print(con.sql("SELECT main_artist, count(*) AS songs FROM t GROUP BY 1 ORDER BY 2 DESC LIMIT 15"))

emb = db.parent / "embeddings"
if any(emb.glob("part-*.parquet")):
    n, = con.execute(f"SELECT count(*) FROM read_parquet('{emb}/part-*.parquet') SEMI JOIN t USING (track_id)").fetchone()
    print(f"Embeddings done: {n:,} of {total:,} ({n / total:.1%})")
