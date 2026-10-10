#!/usr/bin/env python3
"""Export RIFE v4.6 IFNet weights (the app's ncnn weights) and golden outputs.

Reads the PyTorch reference from a directory holding rife46.py / ncnn_weights.py
(default ~/.cache/claude-tmp/rife-npu, MIT-licensed RIFE weights) and writes:

  build/weights/rife46_f32.bin   fp32, torch layouts       (host/arm64 reference)
  build/weights/rife46_hvx.bin   fp16, HVX-arranged layout (DSP)
  tests/golden/                  inputs (RGBA8) + golden flow/mask (+ intermediates for pair 0)

Layouts are documented in inc/lfrife_weights.h.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

BLOCK_CIN = [7, 12, 12, 12]
BLOCK_C = [192, 128, 96, 64]
BLOCK_SCALE = [8, 4, 2, 1]
LAYERS_PER_BLOCK = 11
HDR_FMT = "<8sIIIIQ4I4I4I4I"  # magic ver kind nlayers res total cin[4] c[4] scale[4] pad[4]
LAYER_FMT = "<IIIIQQQQ"
HDR_SIZE = struct.calcsize(HDR_FMT)
LAYER_SIZE = struct.calcsize(LAYER_FMT)
assert HDR_SIZE == 96 and LAYER_SIZE == 48, (HDR_SIZE, LAYER_SIZE)


def collect_layers(net):
    """-> list of (type, cin, cout, stride, weight(np f32), bias(np f32)) in blob order."""
    out = []
    for b in [net.block0, net.block1, net.block2, net.block3]:
        for conv in (b.conv0[0][0], b.conv0[1][0]):
            out.append((0, conv.in_channels, conv.out_channels, 2,
                        conv.weight.detach().numpy().copy(), conv.bias.detach().numpy().copy()))
        for rc in b.convblock:
            assert float(rc.beta.min()) == 1.0 and float(rc.beta.max()) == 1.0
            c = rc.conv
            out.append((1, c.in_channels, c.out_channels, 1,
                        c.weight.detach().numpy().copy(), c.bias.detach().numpy().copy()))
        d = b.lastconv[0]
        out.append((2, d.in_channels, d.out_channels, 2,
                    d.weight.detach().numpy().copy(), d.bias.detach().numpy().copy()))
    assert len(out) == 44
    return out


def hvx_bias(b, nob):
    """fp32 [ob][2][32]: even lanes then odd lanes (matches vmpy(hf,hf)->sf pair split)."""
    bp = np.zeros(nob * 64, np.float32)
    bp[:len(b)] = b
    bp = bp.reshape(nob, 32, 2)  # [ob][i][even/odd]
    return np.ascontiguousarray(bp.transpose(0, 2, 1)).astype(np.float32)


def hvx_conv_weights(w):
    """w[co][ci][3][3] -> fp16 [ob][tap][ci][64]."""
    co, ci, _, _ = w.shape
    nob = (co + 63) // 64
    wp = np.zeros((nob * 64, ci, 3, 3), np.float32)
    wp[:co] = w
    wp = wp.reshape(nob, 64, ci, 9).transpose(0, 3, 2, 1)  # ob, tap, ci, lane
    return np.ascontiguousarray(wp).astype(np.float16), nob


def deconv_lane(pv, i, j, ch):
    return ((pv * 2 + i) * 2 + j) * 5 + ch


def hvx_deconv(w, b):
    """w[ci][24][4][4] -> fp16 [pu][tap6][ci][64], bias fp32 [2][32]."""
    ci, co, _, _ = w.shape
    assert co == 24
    wp = np.zeros((2, 6, ci, 64), np.float32)
    bias = np.zeros(64, np.float32)
    ky_of = {0: {-1: 3, 0: 1}, 1: {0: 2, 1: 0}}  # [pu][dy] (also valid for kx with pv)
    for pu in range(2):
        for dyi in range(2):
            dy = dyi - 1 + pu
            ky = ky_of[pu][dy]
            for dxi in range(3):
                dx = dxi - 1
                tap = dyi * 3 + dxi
                for pv in range(2):
                    if dx not in ky_of[pv]:
                        continue
                    kx = ky_of[pv][dx]
                    for i in range(2):
                        for j in range(2):
                            for ch in range(5):
                                lane = deconv_lane(pv, i, j, ch)
                                wp[pu, tap, :, lane] = w[:, ch * 4 + i * 2 + j, ky, kx]
    for pv in range(2):
        for i in range(2):
            for j in range(2):
                for ch in range(5):
                    bias[deconv_lane(pv, i, j, ch)] = b[ch * 4 + i * 2 + j]
    return wp.astype(np.float16), bias.reshape(32, 2).T.copy().astype(np.float32)


def align(n, a=128):
    return (n + a - 1) // a * a


def write_blob(path, kind, layers):
    n = len(layers)
    data = bytearray()
    table = []
    base = align(HDR_SIZE + n * LAYER_SIZE)
    for typ, cin, cout, stride, w, b in layers:
        if kind == 0:
            wb = w.astype(np.float32).tobytes()
            bb = b.astype(np.float32).tobytes()
        elif typ == 2:
            wp, bp = hvx_deconv(w, b)
            wb, bb = wp.tobytes(), bp.tobytes()
        else:
            wp, nob = hvx_conv_weights(w)
            wb, bb = wp.tobytes(), hvx_bias(b, nob).tobytes()
        woff = base + len(data)
        data += wb
        data += b"\0" * (align(len(data)) - len(data))
        boff = base + len(data)
        data += bb
        data += b"\0" * (align(len(data)) - len(data))
        table.append((typ, cin, cout, stride, woff, len(wb), boff, len(bb)))
    total = base + len(data)
    hdr = struct.pack(HDR_FMT, b"LFRIFE46", 1, kind, n, 0, total, *BLOCK_CIN, *BLOCK_C, *BLOCK_SCALE,
                      0, 0, 0, 0)
    out = bytearray(hdr)
    for t in table:
        out += struct.pack(LAYER_FMT, *t)
    out += b"\0" * (base - len(out))
    out += data
    assert len(out) == total
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path}: {total} bytes, kind {kind}")


def run_golden(ref_dir, net, args):
    import cv2
    import torch
    from rife46 import Rife46

    h, w = args.height, args.width
    model = Rife46(net, h, w, with_frame=False).eval()
    os.makedirs(args.golden, exist_ok=True)
    pairs = {}
    for line in open(os.path.join(ref_dir, "calib", "README.txt")):
        parts = line.split()
        if parts and parts[0].isdigit():
            pairs[int(parts[0])] = float([p for p in parts if p.startswith("t=")][0][2:])
    manifest = []
    for idx in args.pairs:
        t = pairs[idx]
        imgs = []
        for k in (0, 1):
            a = np.load(os.path.join(ref_dir, "calib", f"{idx:03d}_img{k}.npy"))[0]  # 3,H,W
            a8 = np.clip(a.transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            a8 = cv2.resize(a8, (w, h), interpolation=cv2.INTER_AREA)  # H,W,3
            imgs.append(a8)
        name = f"pair{idx:03d}_{w}x{h}"
        for k, a8 in enumerate(imgs):
            rgba = np.concatenate([a8, np.full((h, w, 1), 255, np.uint8)], 2)
            rgba.tofile(os.path.join(args.golden, f"{name}_img{k}.rgba"))
        x = [torch.from_numpy(a8.transpose(2, 0, 1).astype(np.float32) / 255.0)[None] for a8 in imgs]
        ts = torch.full((1, 1, h, w), t)
        with torch.no_grad():
            flow, mask = model(x[0], x[1], ts)
        flow.numpy().astype(np.float32).tofile(os.path.join(args.golden, f"{name}_flow.f32"))
        mask.numpy().astype(np.float32).tofile(os.path.join(args.golden, f"{name}_mask.f32"))
        entry = {"name": name, "t": t, "w": w, "h": h,
                 "flow_absmax": float(flow.abs().max()), "flow_absmean": float(flow.abs().mean())}
        if idx == args.pairs[0]:
            dump_intermediates(net, model, x, ts, os.path.join(args.golden, name + "_dbg"), entry)
        manifest.append(entry)
        print(entry["name"], "t=%.3f" % t, "flow |max| %.2f |mean| %.2f" % (entry["flow_absmax"],
                                                                        entry["flow_absmean"]))
    with open(os.path.join(args.golden, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(args.golden, "manifest.txt"), "w") as f:
        for e in manifest:
            f.write("%s %.6f\n" % (e["name"], e["t"]))


def dump_intermediates(net, model, x, ts, outdir, entry):
    """Per block: net input x (after downsample), feature after conv1, tmp (pre-upsample), cumulative flow/mask."""
    import torch
    import torch.nn.functional as F
    os.makedirs(outdir, exist_ok=True)
    blocks = [net.block0, net.block1, net.block2, net.block3]
    img0, img1 = x
    flow = mask = None
    wi0, wi1 = img0, img1
    for i in range(4):
        s = net.scale_list[i]
        blk = blocks[i]
        if flow is None:
            xin = torch.cat((img0, img1, ts), 1)
            fin = None
        else:
            xin = torch.cat((wi0, wi1, ts, mask), 1)
            fin = flow
        if s != 1:
            xin = F.interpolate(xin, scale_factor=1.0 / s, mode="bilinear", align_corners=False)
        if fin is not None:
            if s != 1:
                fin = F.interpolate(fin, scale_factor=1.0 / s, mode="bilinear", align_corners=False) / s
            xin = torch.cat((xin, fin), 1)
        feat0 = blk.conv0(xin)
        feat = blk.convblock(feat0)
        tmp = blk.lastconv(feat)
        up = F.interpolate(tmp, scale_factor=s, mode="bilinear", align_corners=False) if s != 1 else tmp
        f0, m0 = up[:, :4] * s, up[:, 4:5]
        if flow is None:
            flow, mask = f0, m0
        else:
            flow, mask = flow + f0, mask + m0
        for nm, t in (("xin", xin), ("feat", feat), ("tmp", tmp), ("flow", flow), ("mask", mask)):
            t.detach().numpy().astype(np.float32).tofile(os.path.join(outdir, f"b{i}_{nm}.f32"))
        wi0 = model.warp(img0, flow[:, :2])
        wi1 = model.warp(img1, flow[:, 2:4])
    entry["dbg"] = os.path.basename(outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.expanduser("~/.cache/claude-tmp/rife-npu"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "weights"))
    ap.add_argument("--golden", default=os.path.join(ROOT, "tests", "golden"))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=192)
    ap.add_argument("--pairs", type=int, nargs="*", default=[0, 2, 3, 11])
    ap.add_argument("--no-golden", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.ref)
    os.chdir(args.ref)  # ncnn_weights loads flownet_v4.6.pkl relative to cwd
    from ncnn_weights import load_ifnet_from_ncnn

    net = load_ifnet_from_ncnn()
    layers = collect_layers(net)
    os.makedirs(args.out, exist_ok=True)
    write_blob(os.path.join(args.out, "rife46_f32.bin"), 0, layers)
    write_blob(os.path.join(args.out, "rife46_hvx.bin"), 1, layers)
    if not args.no_golden:
        run_golden(args.ref, net, args)


if __name__ == "__main__":
    main()
