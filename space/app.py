"""Demo: buscador de canciones parecidas (Free Music Archive + MuQ-MuLan). Usa buscar.py."""
import json
import urllib.request

import gradio as gr

import buscar

MIRROR = "benjamin-paine/free-music-archive-large"
N_AUDIOS = 5

tracks, ids, emb = buscar.cargar()
buscar.modelo()  # se carga al arrancar para que la primera búsqueda no tarde


def audio_url(row):
    """URL firmada del clip de 30 s en la réplica de FMA (la sirve la API datasets-server de HF)."""
    url = ("https://datasets-server.huggingface.co/rows?dataset=" + MIRROR +
           f"&config=default&split=train&offset={row}&length=1")
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return json.load(r)["rows"][0]["row"]["audio"][0]["src"]
    except Exception:
        return None


def mostrar(res, cabecera=""):
    tabla = [[f"{r['similitud']:.3f}", r["main_artist"], r["name"], r["macro_genre"] or "-", r["license"]]
             for r in res]
    audios = [gr.Audio(value=audio_url(r["audio_row"]), label=f"{n}. {r['main_artist']} - {r['name']}")
              for n, r in enumerate(res[:N_AUDIOS], 1)]
    return [cabecera, tabla, *audios]


def por_texto(q):
    return mostrar(buscar.vecinos(tracks, ids, emb, buscar.vec_texto(q)), f"Canciones que suenan a «{q}»")


def por_cancion(q):
    tid = buscar.encontrar(tracks, q)
    if tid is None or tid not in set(ids):
        return mostrar([], f"No encuentro «{q}» en el catálogo")
    i = int((ids == tid).nonzero()[0][0])
    name, artist = tracks.filter(f"track_id = '{tid}'").select("name, main_artist").fetchone()
    return mostrar(buscar.vecinos(tracks, ids, emb, emb[i], excluir=tid), f"Parecidas a: {artist} - {name}")


def por_audio(path):
    if not path:
        return mostrar([], "Sube o graba un audio")
    return mostrar(buscar.vecinos(tracks, ids, emb, buscar.vec_audio(path)), "Parecidas a tu audio")


with gr.Blocks(title="Buscador de canciones parecidas") as demo:
    gr.Markdown(
        "# Buscador de canciones parecidas\n"
        f"{len(ids):,} canciones con licencia Creative Commons del "
        "[Free Music Archive](https://github.com/mdeff/fma), convertidas en vectores con "
        "[MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large). Busca por descripción, por una "
        "canción del catálogo o con tu propio audio. "
        "[Código](https://github.com/PGSCOM/music-similarity) · "
        f"[Datos](https://huggingface.co/datasets/{buscar.DATASET})")
    with gr.Tab("Describe"):
        q_texto = gr.Textbox(label="Descripción (mejor en inglés)", placeholder="sad piano ballad")
        b_texto = gr.Button("Buscar")
        gr.Examples(["sad piano ballad", "aggressive metal with fast drums", "chill lo-fi hip hop beat",
                     "acoustic folk guitar and female vocals", "orchestral film music"], q_texto)
    with gr.Tab("Parecidas a…"):
        q_cancion = gr.Textbox(label="Canción del catálogo: «artista - título» o parte", placeholder="Jahzzar - Siesta")
        b_cancion = gr.Button("Buscar")
        gr.Examples(["Jahzzar - Siesta", "Kevin MacLeod", "Broke For Free"], q_cancion)
    with gr.Tab("Sube un audio"):
        q_audio = gr.Audio(label="Audio (se usan los primeros 30 s)", type="filepath")
        b_audio = gr.Button("Buscar")
    cabecera = gr.Markdown()
    tabla = gr.Dataframe(headers=["similitud", "artista", "título", "género", "licencia"], wrap=True)
    audios = [gr.Audio(interactive=False) for _ in range(N_AUDIOS)]

    salidas = [cabecera, tabla, *audios]
    for b, q, f in ((b_texto, q_texto, por_texto), (b_cancion, q_cancion, por_cancion), (b_audio, q_audio, por_audio)):
        b.click(f, q, salidas)
    q_texto.submit(por_texto, q_texto, salidas)
    q_cancion.submit(por_cancion, q_cancion, salidas)

demo.launch()
