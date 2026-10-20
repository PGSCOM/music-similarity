---
title: Buscador de canciones parecidas
emoji: 🎧
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app.py
license: cc-by-nc-4.0
models:
  - OpenMuQ/MuQ-MuLan-large
datasets:
  - PGSCOM/fma-muq-mulan
  - benjamin-paine/free-music-archive-large
short_description: Busca música por descripción, canción o audio (MuQ-MuLan)
---

Demo de [music-similarity](https://github.com/PGSCOM/music-similarity): 105.000 canciones del Free
Music Archive convertidas en vectores con MuQ-MuLan. Busca por texto, por una canción del catálogo
o con tu propio audio. Gradio 5 porque MuQ necesita `transformers<5` (y este, `huggingface_hub` 0.x).
