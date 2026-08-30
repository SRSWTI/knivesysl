#!/usr/bin/env python3
"""Two questions about the NVFP4 tier, answered on real weights and real activations.

  Q1: is it worth emitting NVFP4 from the converter (bf16 -> NVFP4, one rounding step)
      instead of the shipped load-time repack (bf16 -> E2M3 -> NVFP4, two steps)?
  Q2: does a Hadamard rotation along K help? W@x == (W@H) @ (H.T@x) for orthogonal H,
      and rotating along K spreads outlier channels across the axis that the per-16
      ue4m3 block scales run along. This is what QuaRot does and what QuTLASS's
      "NVFP4 + Hadamard" curves measure.

Error is reported on the PRODUCT W@x, not on the weight: a rotation changes what "weight
error" even means, and the product is what the model consumes.

Activations MUST be real. A Gaussian has no outlier channels, so it makes any rotation
look worthless; the engine's real residual stream has a 245x per-channel peak-to-median
ratio at layer 32 (see act.py). Set TQ_ACT to the .npy to use.

All quantizers reproduce the engine exactly:
  E2M3  1 sign + 2 exp + 3 mantissa, grid max 7.5, per-128x128 pow2 block scale
  NVFP4 E2M1 {0,.5,1,1.5,2,3,4,6} + per-16 ue4m3 + one fp32 global per 128-row block
        (global constant over K so it folds in the epilogue)
  act   E2M1 + per-16 ue4m3, no global (k_tq_nvf4_quant_x)
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

E2M1 = np.array([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=np.float64)
E2M3 = np.unique(np.array([m / 8.0 for m in range(8)]
                          + [(1 + m / 8.0) * (1 << (e - 1))
                             for e in (1, 2, 3) for m in range(8)], dtype=np.float64))


def q_grid(x, grid):
    s = np.sign(x)
    idx = np.abs(np.abs(x)[..., None] - grid).argmin(axis=-1)
    return s * grid[idx]


def ue4m3(v):
    """Positive scale -> ue4m3 (4 exp bias 7, 3 mantissa), exactly tq_f2ue4m3."""
    v = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v)
    pos = v > 0
    if not pos.any():
        return out
    m, e = np.frexp(v[pos])
    E, f = e + 6, np.rint((2.0 * m - 1.0) * 8.0)
    bump = f > 7
    f, E = np.where(bump, 0, f), np.where(bump, E + 1, E)
    hi = E > 15
    E, f = np.where(hi, 15, E), np.where(hi, 7, f)
    out[pos] = np.where(E < 1, 0.0, np.ldexp(1.0 + f / 8.0, E - 7))
    return out


def to_e2m3(W):
    out = np.zeros_like(W)
    for r in range(0, W.shape[0], 128):
        for c in range(0, W.shape[1], 128):
            b = W[r:r + 128, c:c + 128]
            am = np.abs(b).max()
            s = 2.0 ** np.ceil(np.log2(am / 7.5)) if am > 0 else 1.0
            out[r:r + 128, c:c + 128] = q_grid(b / s, E2M3) * s
    return out


def to_nvfp4(W):
    M, K = W.shape
    out = np.zeros_like(W)
    for r in range(0, M, 128):
        rows = W[r:r + 128]
        am = np.abs(rows).max()
        g = am / 6.0 if am > 0 else 1.0
        b = rows.reshape(rows.shape[0], K // 16, 16)
        sb = ue4m3(np.abs(b).max(axis=2) / 6.0 / g)
        sb = np.where(sb <= 0, 2.0 ** -6, sb)
        eff = (sb * g)[:, :, None]
        out[r:r + 128] = (q_grid(b / eff, E2M1) * eff).reshape(rows.shape)
    return out


def quant_act(X):
    """X is [K][N]; K contiguous per column, matching the engine's token-major pack."""
    K, N = X.shape
    Xt = X.T.reshape(N, K // 16, 16)
    sb = ue4m3(np.abs(Xt).max(axis=2) / 6.0)
    sb = np.where(sb <= 0, 2.0 ** -6, sb)
    eff = sb[:, :, None]
    return (q_grid(Xt / eff, E2M1) * eff).reshape(N, K).T


def hadamard(n=256):
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def rot(X, H, axis):
    """Block-diagonal rotation along `axis`. K = 20*256 here, so 256-blocks tile it and
    the engine's shipped 256-wide FWHT (tq_fwht256) computes it in 8 butterfly stages."""
    b = H.shape[0]
    X = np.moveaxis(X, axis, -1)
    s = X.shape
    return np.moveaxis((X.reshape(-1, s[-1] // b, b) @ H).reshape(s), -1, axis)


def main() -> int:
    snap = glob.glob(os.environ.get("TQ_HF_HUB", os.path.expanduser("~/.cache/huggingface/hub")) + "/"
                     + os.environ.get("TQ_HF_MODEL", "models--Qwen--Qwen3.8-27B") + "/snapshots/*/")[0]
    idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    want = sorted({n for n in idx
                   if any(k in n for k in ("mlp.gate_proj", "mlp.down_proj",
                                           "q_proj", "linear_attn.in_proj_qkv"))
                   and (".0." in n or ".3." in n)})[:4]
    if not want:
        print("no matching tensors", file=sys.stderr); return 1
    from safetensors import safe_open
    import torch

    actf = os.environ.get("TQ_ACT", "/tmp/act.npy")
    A0 = np.load(actf).astype(np.float64)
    a = np.abs(A0); ch = a.max(axis=0)
    print(f"activations: {actf} {A0.shape}  per-channel peak/median = "
          f"{ch.max() / np.median(ch):.1f}x")
    H = hadamard(256)
    blocks = [int(x) for x in os.environ.get("TQ_HBLK", "16,32,64,256").split(",")]
    print(f"\n{'layer.tensor':30s} {'shipped':>8s} {'direct':>8s}"
          + "".join(f" {('had' + str(b)):>8s}" for b in blocks))
    print("-" * (30 + 18 + 9 * len(blocks)))
    for name in want:
        with safe_open(snap + idx[name], framework="pt") as f:
            W = f.get_tensor(name).to(torch.float64).numpy()
        if W.ndim != 2 or W.shape[1] % 256 or W.shape[0] % 128:
            continue
        W = W[:512]
        K = W.shape[1]
        A = A0 if A0.shape[1] == K else np.resize(A0, (A0.shape[0], K))
        A = A / np.sqrt((A * A).mean(axis=1, keepdims=True))   # stand in for RMSNorm
        X = A.T
        ref = W @ X

        def perr(Wq, Xq):
            d = Wq @ Xq - ref
            return float(np.sqrt((d * d).mean()) / np.sqrt((ref * ref).mean()))

        Xq = quant_act(X)
        e_ship = perr(to_nvfp4(to_e2m3(W)), Xq)
        e_dir = perr(to_nvfp4(W), Xq)
        # Rotation block size vs the 16-element scale group: if spreading an outlier
        # across more groups is what hurts, error must rise monotonically with block size.
        hs = []
        for b in blocks:
            Hb = hadamard(b)
            hs.append(perr(to_nvfp4(rot(W, Hb, 1)), quant_act(rot(X, Hb, 0))))
        print(f"{name.split('layers.')[1][:30]:30s} {e_ship:8.5f} {e_dir:8.5f}"
              + "".join(f" {h:8.5f}" for h in hs))


if __name__ == "__main__":
    raise SystemExit(main())
