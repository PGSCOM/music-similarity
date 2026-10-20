---
title: Suena a…
emoji: 🎧
colorFrom: gray
colorTo: yellow
sdk: static
app_file: index.html
license: cc-by-nc-4.0
models:
  - OpenMuQ/MuQ-MuLan-large
datasets:
  - PGSCOM/fma-muq-mulan
  - benjamin-paine/free-music-archive-large
short_description: 104.921 canciones en un mapa; busca con tu audio
---

Demo de [music-similarity](https://github.com/PGSCOM/music-similarity). Todo corre en el navegador:
la búsqueda es un producto escalar contra 104.921 vectores (int8), y tu audio pasa por la parte de
audio de MuQ-MuLan exportada a ONNX (WebGPU si está disponible, si no WebAssembly).
Datos: `python preparar_web.py`. Modelo: `python exportar_web.py web/modelo`.
