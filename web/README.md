---
title: MusicSearcher
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
short_description: 104,921 songs on a map; search with your own audio
---

Demo of [music-similarity](https://github.com/PGSCOM/music-similarity). Everything runs in the
browser: the search is a dot product against 104,921 vectors (int8), and your audio goes through the
audio part of MuQ-MuLan exported to ONNX (WebGPU if available, WebAssembly otherwise).
Data: `python prepare_web.py`. Model: `python export_web.py web/model`.
