#!/usr/bin/env python3
"""Extrae frames de los .mov de HQ-WMCA-RGB conservando el subattack type en la ruta.
output: /mnt/d1/data/HQ-WMCA/frames/<Bonafide|SUBTIPO>/<video>/frameNNN.jpg"""
import os, sys, cv2
SRC = "/mnt/d1/data/HQ-WMCA/HQ-WMCA-RGB/HQ-WMCA-RGB"
DST = "/mnt/d1/data/HQ-WMCA/frames"
N = 12  # frames por vídeo
jobs = []
for cat in ("Bonafide", "Impersonation", "Obfuscation"):
    d = os.path.join(SRC, cat)
    if not os.path.isdir(d): continue
    if cat == "Bonafide":
        jobs += [("Bonafide", os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".mov")]
    else:
        for st in sorted(os.listdir(d)):
            sd = os.path.join(d, st)
            if os.path.isdir(sd):
                jobs += [(st, os.path.join(sd, f)) for f in os.listdir(sd) if f.endswith(".mov")]
print(f"{len(jobs)} vídeos a procesar", flush=True)
ok = fail = 0
for i, (st, path) in enumerate(jobs, 1):
    name = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(DST, st, name)
    if os.path.isdir(out) and len(os.listdir(out)) >= N: ok += 1; continue
    os.makedirs(out, exist_ok=True)
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total < 1: fail += 1; cap.release(); continue
    idxs = [int(total * k / (N + 1)) for k in range(1, N + 1)]
    got = 0
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        r, fr = cap.read()
        if r: cv2.imwrite(os.path.join(out, f"frame{j:03d}.jpg"), fr); got += 1
    cap.release()
    ok += got > 0; fail += got == 0
    if i % 200 == 0: print(f"  [{i}/{len(jobs)}] ok={ok} fail={fail}", flush=True)
print(f"FIN: ok={ok} fail={fail}", flush=True)
