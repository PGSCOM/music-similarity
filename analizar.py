#!/usr/bin/env python3
"""
Resumen de una base de datos del proyecto (cualquier tracks*.parquet de esta carpeta o sus subcarpetas).

  venv/bin/python analizar.py                        # muestra las que hay y pregunta cuál escanear
  venv/bin/python analizar.py tracks.parquet
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
        sys.exit("No hay ningún tracks*.parquet aquí ni en las subcarpetas")
    for i, p in enumerate(dbs, 1):
        print(f"  {i}) {p.relative_to(HERE)}  ({p.stat().st_size / 1e9:.2f} GB)")
    db = dbs[int(input("¿Cuál escaneo? ")) - 1]

con = duckdb.connect()
con.execute("SET memory_limit='4GB'; SET threads=4")
con.execute(f"CREATE VIEW t AS FROM read_parquet('{db}')")
total, = con.execute("SELECT count(*) FROM t").fetchone()
print(f"\n{db}: {total:,} canciones, {db.stat().st_size / 1e9:.2f} GB")

cols = con.execute("SELECT column_name, column_type FROM (DESCRIBE t)").fetchall()
llenas = con.execute("SELECT " + ", ".join(
    f"count(*) FILTER ({c} IS NOT NULL{' AND len(' + c + ') > 0' if ty.endswith('[]') else ''})" for c, ty in cols)
    + " FROM t").fetchone()
print("\nColumnas (con dato):")
for (c, ty), n in zip(cols, llenas):
    print(f"  {c:22s} {ty:12s} {n / total:6.1%}")

print("\nPor género grande:")
con.sql("""SELECT coalesce(macro_genre, '(sin género)') AS macro_genre, count(*) AS canciones,
                        count(DISTINCT main_artist) AS artistas,
                        count(*) FILTER (preview_url IS NOT NULL) AS con_audio,
                        median(popularity) AS popularidad_mediana, max(popularity) AS popularidad_max
                 FROM t GROUP BY 1 ORDER BY 2 DESC""").show(max_rows=50)

print("Popularidad (percentiles 50/90/99/máx):",
      con.execute("SELECT quantile_cont(popularity, [0.5, 0.9, 0.99]), max(popularity) FROM t").fetchone())

print("\nPor década de publicación:")
con.sql("""SELECT try_cast(left(release_date, 4) AS INT) // 10 * 10 AS decada, count(*) AS canciones
           FROM t GROUP BY 1 ORDER BY 1 NULLS LAST""").show(max_rows=40)

print("Artistas con más canciones:")
print(con.sql("SELECT main_artist, count(*) AS canciones FROM t GROUP BY 1 ORDER BY 2 DESC LIMIT 15"))

emb = db.parent / "embeddings"
if any(emb.glob("part-*.parquet")):
    n, = con.execute(f"SELECT count(*) FROM read_parquet('{emb}/part-*.parquet') SEMI JOIN t USING (track_id)").fetchone()
    print(f"Embeddings hechos: {n:,} de {total:,} ({n / total:.1%})")
