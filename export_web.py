#!/usr/bin/env python3
"""
Exports the audio part of MuQ-MuLan to ONNX so that you can search with your own audio in the
browser (web/).

  pip install onnx onnxruntime onnx-ir
  python export_web.py web/model   # -> mel.onnx + audio_fp32/fp16/q8/q4/q4f16.onnx

Two models: mel.onnx (10 s clip at 24 kHz, 1 x 240000 -> normalized mel spectrogram; always fp32,
in fp16 it overflows) and audio_*.onnx (mel -> normalized 512-dim vector; the one that gets
quantized). The browser splits the first 30 s exactly like MuQMuLan and embed.py does and averages
the result.
It checks every version against PyTorch with real songs from the mirror and prints its size.
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
TEST = ("datasets/benjamin-paine/free-music-archive-large@d4cb3e133a7c5a007ddd20458aa30aef8968cf27/"
        "data/train-00000-of-00171.parquet")


class Mel(torch.nn.Module):
    """clip (1 x 240000) -> normalized mel (1 x 128 x 1000), same as MuQ but without torch.stft:
    MelSpectrogram(n_fft=2048, hop=240, Hann, center/reflect, power 2) + dB + drop the last frame
    + (x - mean) / std. The windowed DFT is a stride-240 convolution (cos and sin)."""
    def __init__(self, muq):
        super().__init__()
        sp, n = muq.preprocessor_melspec_2048.mel_stft.spectrogram, 2048
        k = torch.arange(n, dtype=torch.float64)[:, None] * torch.arange(n // 2 + 1, dtype=torch.float64)
        w = sp.window.double()[:, None]
        self.n, self.hop = n, sp.hop_length
        self.register_buffer("cos", (torch.cos(2 * torch.pi * k / n) * w).T[:, None].float())   # 1025 x 1 x 2048
        self.register_buffer("sin", (-torch.sin(2 * torch.pi * k / n) * w).T[:, None].float())
        self.register_buffer("fb", muq.preprocessor_melspec_2048.mel_stft.mel_scale.fb.T.contiguous())  # 128 x 1025
        self.mean, self.std = muq.stat["melspec_2048_mean"], muq.stat["melspec_2048_std"]

    def forward(self, wav):
        x = F.pad(wav[:, None], (self.n // 2, self.n // 2), mode="reflect")
        pot = F.conv1d(x, self.cos, stride=self.hop) ** 2 + F.conv1d(x, self.sin, stride=self.hop) ** 2
        db = 10 * torch.log10(torch.matmul(self.fb, pot).clamp(min=1e-10))
        return (db[..., :-1] - self.mean) / self.std


class Encoder(torch.nn.Module):
    """normalized mel -> normalized 512-dim vector (rest of get_audio_latents of MuQ-MuLan)."""
    def __init__(self, m):
        super().__init__()
        self.a, self.lat = m.mulan_module.audio, m.mulan_module.audio_to_latents

    def forward(self, mel):
        _, hidden, _ = self.a.model.model.encoder(mel, is_features_only=True)
        h, _ = self.a.transformer(self.a.proj(hidden[self.a.use_layer_idx]), return_all_layers=True)
        return F.normalize(self.lat(h.mean(dim=-2)), dim=-1)


def clips(wav):
    """Same as embed.py + MuQMuLan._get_all_clips: repeat if shorter than 10 s and cut into 10 s windows."""
    if len(wav) < CLIP:
        wav = np.tile(wav, -(-CLIP // len(wav)))
    out = [wav[i:i + CLIP] for i in range(0, len(wav) - CLIP + 1, CLIP)]
    if len(wav) % CLIP:
        rest = wav[len(wav) // CLIP * CLIP:]
        out.append(np.concatenate([rest, wav[:CLIP - len(rest)]]))
    return out


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    m = MuQMuLan.from_pretrained(MODEL).eval()
    with HfFileSystem().open(TEST) as f:
        rows = pq.ParquetFile(f).read_row_group(0, columns=["audio"]).column("audio").to_pylist()[:6]
    wavs = [librosa.load(io.BytesIO(a["bytes"]), sr=SR, mono=True, duration=30)[0] for a in rows]
    with torch.inference_mode():
        # with the original mel (torchaudio); MuQMuLan does not renormalize the mean of the pieces, embed.py does
        ref = [F.normalize(m(wavs=[torch.from_numpy(w)])[0], dim=0).numpy() for w in wavs]

    mel, enc = Mel(m.mulan_module.audio.model.model).eval(), Encoder(m).eval()
    x = torch.from_numpy(wavs[0][:CLIP])[None]
    with torch.no_grad():
        assert torch.allclose(enc(mel(x)), m.mulan_module.get_audio_latents(x), atol=2e-3), "does not match MuQ"
        torch.onnx.export(mel, (x,), out / "mel.onnx", dynamo=False, opset_version=18,
                          input_names=["wav"], output_names=["mel"])
        fp32 = out / "audio_fp32.onnx"
        if not fp32.exists():
            torch.onnx.export(enc, (mel(x),), fp32, dynamo=False, opset_version=18,
                              input_names=["mel"], output_names=["embedding"])
        tro = [F.normalize(torch.stack([enc(mel(torch.from_numpy(c)[None]))[0] for c in clips(w)]).mean(0), dim=0)
               for w in wavs]
    print("slicing in PyTorch vs MuQMuLan: minimum cosine",
          f"{min(float(t.numpy() @ r) for t, r in zip(tro, ref)):.4f}")

    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    from onnxruntime.transformers.float16 import convert_float_to_float16
    fp16 = out / "audio_fp16.onnx"
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), fp16)
    # Weights of the MatMuls in blocks of 32; the rest (convolutions) stays in fp32, or in fp16 in q4f16
    for name, base, bits in (("audio_q8.onnx", fp32, 8), ("audio_q4.onnx", fp32, 4), ("audio_q4f16.onnx", fp16, 4)):
        q = MatMulNBitsQuantizer(onnx.load(base), bits=bits, block_size=32, is_symmetric=True,
                                 op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
        q.process()
        onnx.save(q.model.model, out / name)

    import onnxruntime as ort
    sm = ort.InferenceSession(out / "mel.onnx", providers=["CPUExecutionProvider"])
    with torch.no_grad():
        mt, et = mel(x), enc(mel(x))[0].numpy()
    mo = sm.run(None, {"wav": x.numpy()})[0]
    print(f"mel ONNX vs PyTorch: max difference {np.abs(mo - mt.numpy()).max():.5f}")
    so = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    eo = so.run(None, {"mel": mt.numpy()})[0][0]
    print(f"encoder ONNX vs PyTorch (same mel): cosine {float(eo @ et):.4f}")
    del m, enc, so
    for f in ("audio_fp32.onnx", "audio_fp16.onnx", "audio_q8.onnx", "audio_q4.onnx", "audio_q4f16.onnx"):
        s = ort.InferenceSession(out / f, providers=["CPUExecutionProvider"])
        cos = []
        for w, r in zip(wavs, ref):
            v = np.mean([s.run(None, {"mel": sm.run(None, {"wav": c[None].astype(np.float32)})[0]})[0][0]
                         for c in clips(w)], axis=0)
            cos.append(float(v @ r / np.linalg.norm(v)))
        mb = sum(p.stat().st_size for p in out.glob(f + "*")) / 1e6
        print(f"{f:16} {mb:7.0f} MB   cosine with PyTorch: min {min(cos):.4f}  mean {np.mean(cos):.4f}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "web/model"))
