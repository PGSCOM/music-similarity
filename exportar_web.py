#!/usr/bin/env python3
"""
Exporta la parte de audio de MuQ-MuLan a ONNX para buscar con tu propio audio en el navegador (web/).

  pip install onnx onnxruntime onnx-ir
  python exportar_web.py web/modelo     # -> mel.onnx + audio_fp32/fp16/q8/q4/q4f16.onnx

Dos modelos: mel.onnx (clip de 10 s a 24 kHz, 1 × 240000 -> espectrograma mel normalizado; siempre fp32,
en fp16 se desborda) y audio_*.onnx (mel -> vector de 512 normalizado; el que se cuantiza). El navegador
trocea los primeros 30 s igual que MuQMuLan y embed.py y hace la media.
Comprueba cada versión contra PyTorch con canciones reales de la réplica e imprime su tamaño.
"""
import io
import pathlib
import sys

import librosa
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfFileSystem
from muq import MuQMuLan

MODEL = "OpenMuQ/MuQ-MuLan-large"
SR, CLIP = 24_000, 240_000
PRUEBA = ("datasets/benjamin-paine/free-music-archive-large@d4cb3e133a7c5a007ddd20458aa30aef8968cf27/"
          "data/train-00000-of-00171.parquet")


class Mel(torch.nn.Module):
    """clip (1 × 240000) -> mel normalizado (1 × 128 × 1000), igual que MuQ pero sin torch.stft:
    MelSpectrogram(n_fft=2048, hop=240, Hann, center/reflect, potencia 2) + dB + quitar la última trama
    + (x - media) / desviación. La DFT con ventana es una convolución de paso 240 (cos y sin)."""
    def __init__(self, muq):
        super().__init__()
        sp, n = muq.preprocessor_melspec_2048.mel_stft.spectrogram, 2048
        k = torch.arange(n, dtype=torch.float64)[:, None] * torch.arange(n // 2 + 1, dtype=torch.float64)
        w = sp.window.double()[:, None]
        self.n, self.hop = n, sp.hop_length
        self.register_buffer("cos", (torch.cos(2 * torch.pi * k / n) * w).T[:, None].float())   # 1025 × 1 × 2048
        self.register_buffer("sin", (-torch.sin(2 * torch.pi * k / n) * w).T[:, None].float())
        self.register_buffer("fb", muq.preprocessor_melspec_2048.mel_stft.mel_scale.fb.T.contiguous())  # 128 × 1025
        self.media, self.desv = muq.stat["melspec_2048_mean"], muq.stat["melspec_2048_std"]

    def forward(self, wav):
        x = F.pad(wav[:, None], (self.n // 2, self.n // 2), mode="reflect")
        pot = F.conv1d(x, self.cos, stride=self.hop) ** 2 + F.conv1d(x, self.sin, stride=self.hop) ** 2
        db = 10 * torch.log10(torch.matmul(self.fb, pot).clamp(min=1e-10))
        return (db[..., :-1] - self.media) / self.desv


class Codificador(torch.nn.Module):
    """mel normalizado -> vector de 512 normalizado (resto de get_audio_latents de MuQ-MuLan)."""
    def __init__(self, m):
        super().__init__()
        self.a, self.lat = m.mulan_module.audio, m.mulan_module.audio_to_latents

    def forward(self, mel):
        _, ocultas, _ = self.a.model.model.encoder(mel, is_features_only=True)
        h, _ = self.a.transformer(self.a.proj(ocultas[self.a.use_layer_idx]), return_all_layers=True)
        return F.normalize(self.lat(h.mean(dim=-2)), dim=-1)


def clips(wav):
    """Como embed.py + MuQMuLan._get_all_clips: repite si dura < 10 s y trocea en ventanas de 10 s."""
    if len(wav) < CLIP:
        wav = np.tile(wav, -(-CLIP // len(wav)))
    out = [wav[i:i + CLIP] for i in range(0, len(wav) - CLIP + 1, CLIP)]
    if len(wav) % CLIP:
        resto = wav[len(wav) // CLIP * CLIP:]
        out.append(np.concatenate([resto, wav[:CLIP - len(resto)]]))
    return out


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    m = MuQMuLan.from_pretrained(MODEL).eval()
    with HfFileSystem().open(PRUEBA) as f:
        filas = pq.ParquetFile(f).read_row_group(0, columns=["audio"]).column("audio").to_pylist()[:6]
    wavs = [librosa.load(io.BytesIO(a["bytes"]), sr=SR, mono=True, duration=30)[0] for a in filas]
    with torch.inference_mode():
        # con el mel original (torchaudio); MuQMuLan no renormaliza la media de los trozos, embed.py sí
        ref = [F.normalize(m(wavs=[torch.from_numpy(w)])[0], dim=0).numpy() for w in wavs]

    mel, cod = Mel(m.mulan_module.audio.model.model).eval(), Codificador(m).eval()
    x = torch.from_numpy(wavs[0][:CLIP])[None]
    with torch.no_grad():
        assert torch.allclose(cod(mel(x)), m.mulan_module.get_audio_latents(x), atol=2e-3), "no coincide con MuQ"
        torch.onnx.export(mel, (x,), out / "mel.onnx", dynamo=False, opset_version=18,
                          input_names=["wav"], output_names=["mel"])
        fp32 = out / "audio_fp32.onnx"
        if not fp32.exists():
            torch.onnx.export(cod, (mel(x),), fp32, dynamo=False, opset_version=18,
                              input_names=["mel"], output_names=["embedding"])
        tro = [F.normalize(torch.stack([cod(mel(torch.from_numpy(c)[None]))[0] for c in clips(w)]).mean(0), dim=0)
               for w in wavs]
    print("troceo en PyTorch vs MuQMuLan: coseno min",
          f"{min(float(t.numpy() @ r) for t, r in zip(tro, ref)):.4f}")

    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.transformers.float16 import convert_float_to_float16
    fp16 = out / "audio_fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), fp16)
    # pesos de las MatMul en bloques de 32; el resto (convoluciones) se queda en fp32, o en fp16 en q4f16
    for nombre, base, bits in (("audio_q8.onnx", fp32, 8), ("audio_q4.onnx", fp32, 4), ("audio_q4f16.onnx", fp16, 4)):
        q = MatMulNBitsQuantizer(onnx.load(base), bits=bits, block_size=32, is_symmetric=True,
                                 op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
        q.process()
        onnx.save(q.model.model, out / nombre)

    import onnxruntime as ort
    sm = ort.InferenceSession(out / "mel.onnx", providers=["CPUExecutionProvider"])
    with torch.no_grad():
        mt, et = mel(x), cod(mel(x))[0].numpy()
    mo = sm.run(None, {"wav": x.numpy()})[0]
    print(f"mel ONNX vs PyTorch: diferencia máx {np.abs(mo - mt.numpy()).max():.5f}")
    so = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    eo = so.run(None, {"mel": mt.numpy()})[0][0]
    print(f"codificador ONNX vs PyTorch (mismo mel): coseno {float(eo @ et):.4f}")
    del m, cod, so
    for f in ("audio_fp32.onnx", "audio_fp16.onnx", "audio_q8.onnx", "audio_q4.onnx", "audio_q4f16.onnx"):
        s = ort.InferenceSession(out / f, providers=["CPUExecutionProvider"])
        cos = []
        for w, r in zip(wavs, ref):
            v = np.mean([s.run(None, {"mel": sm.run(None, {"wav": c[None].astype(np.float32)})[0]})[0][0]
                         for c in clips(w)], axis=0)
            cos.append(float(v @ r / np.linalg.norm(v)))
        mb = sum(p.stat().st_size for p in out.glob(f + "*")) / 1e6
        print(f"{f:16} {mb:7.0f} MB   coseno con PyTorch: min {min(cos):.4f}  media {np.mean(cos):.4f}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "web/modelo"))
