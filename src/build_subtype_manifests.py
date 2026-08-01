#!/usr/bin/env python3
"""
Manifiestos CON SUBTIPO DE ATAQUE — requisito para el curriculum de IBT.

IBT necesita >=2 clases spoof para escalonar. Este script genera, por dataset, un CSV con
una columna `subtype` (identificador de la clase de ataque; 'live' para bonafide):

  dataset,split,label,subtype,path

Datasets soportados (los que SÍ tienen subtipos):
  - CelebA-Spoof : desde metas/intra_test/*_label.json, campo [40] (10 tipos). 162k bonafide en train.
  - OULU-NPU     : AccessType = último campo de la carpeta v_{live,spoof}_g{N}_{ses}_{user}_{AT}
                   (1=real, 2=print1, 3=print2, 4=replay1, 5=replay2). Tiene dev OFICIAL.
  - SiW          : carpeta v_{live,spoof}_g{subj}-{sensor}-{tipo}-{medio}-{ses}
                   (tipo 1=live, 2=print, 3=replay); subtipo = tipo_medio (6 clases de ataque).

NO soportados (sin subtipos utilizables):
  - CASIA-CeFA   : el train de 4@1 tiene UN solo tipo de ataque y dev/test están anonimizados.

Uso:  python build_subtype_manifests.py [celeba|oulu|siw|all]
Salida: manifests/<dataset>_subtype.csv
"""
import os, sys, csv, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = os.path.join(HERE, "manifests"); os.makedirs(MAN, exist_ok=True)
ROOT = os.environ.get("FAS_DATA_ROOT", "/mnt/d1/data")
HDR = ["dataset", "split", "label", "subtype", "path"]

CELEBA_TYPES = {0: "live", 1: "Photo", 2: "Poster", 3: "A4", 4: "FaceMask", 5: "UpperBodyMask",
                6: "RegionMask", 7: "PC", 8: "Pad", 9: "Phone", 10: "3DMask"}
OULU_TYPES = {"1": "live", "2": "print1", "3": "print2", "4": "replay1", "5": "replay2"}
SIW_TYPES = {"1": "live", "2": "print", "3": "replay"}


def write(name, rows):
    p = os.path.join(MAN, f"{name}_subtype.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)
    from collections import Counter
    c = Counter((r[1], r[3]) for r in rows)
    print(f"-> {p}  ({len(rows)} rows)")
    for (sp, st), n in sorted(c.items()):
        print(f"     {sp:5s} {st:14s} {n:8d}")
    return p


def celeba():
    """Uses the COMPLETE copy (74 parts), the only one that ships the metas/ folder."""
    base = os.path.join(ROOT, "CelebA-Spoof", "_redownload", "CelebA_Spoof")
    metas = os.path.join(base, "metas", "intra_test")
    assert os.path.isdir(metas), f"faltan metas en {metas} (¿copia incompleta?)"
    rows = []
    for split, fn in [("train", "train_label.json"), ("test", "test_label.json")]:
        d = json.load(open(os.path.join(metas, fn)))
        for rel, v in d.items():
            st = CELEBA_TYPES.get(v[40], f"t{v[40]}")
            rows.append(["CelebA-Spoof", split, int(v[43]), st, os.path.join(base, rel)])
    return write("CelebA-Spoof", rows)


def hqwmca():
    """HQ-WMCA (RGB): 10 very diverse PAIs (rigid/flexible masks, mannequin, paper, print, replay,
    gafas, maquillaje, tatuaje, peluca) + bonafide. Frames ya extraídos en frames/<subtipo>/<video>/.
    NO hay split oficial -> se talla train/dev/test DISJUNTO POR SUJETO (campo 3 del nombre WMCA:
    1_01_<SUJETO>_...). 70/15/15. Todos los subtipos presentes en cada split (los de pocos subjects,
    e.g. Print with 3, may fall unevenly; acceptable for the within-dataset design."""
    base = os.path.join(ROOT, "HQ-WMCA", "frames")
    assert os.path.isdir(base), f"no existe {base} (¿extracción pendiente?)"
    import random as _rnd
    # 1) collect every subject and assign each a split deterministically
    subj_of = lambda vid: vid.split("_")[2] if len(vid.split("_")) > 2 else vid
    subjects = set()
    for st in os.listdir(base):
        for vid in os.listdir(os.path.join(base, st)):
            subjects.add(subj_of(vid))
    subs = sorted(subjects); _rnd.Random(0).shuffle(subs)
    n = len(subs); n_te = int(n * 0.15); n_dv = int(n * 0.15)
    split_of = {}
    for i, s in enumerate(subs):
        split_of[s] = "test" if i < n_te else "dev" if i < n_te + n_dv else "train"
    # 2) rows
    rows = []
    for st in sorted(os.listdir(base)):
        label = 0 if st == "Bonafide" else 1
        sub = "live" if st == "Bonafide" else st
        sd = os.path.join(base, st)
        for vid in sorted(os.listdir(sd)):
            sp = split_of[subj_of(vid)]
            for fr in _frames(os.path.join(sd, vid)):
                rows.append(["HQ-WMCA", sp, label, sub, os.path.join(sd, vid, fr)])
    print(f"  [HQ-WMCA] {n} subjects -> train/dev/test disjunto")
    return write("HQ-WMCA", rows)


def casia_surf():
    """CASIA-SURF: su protocolo OFICIAL ya es zero-shot cross-attack.
       Training = ataques 04/05/06 · Val y Testing = ataques 01/02/03 (no vistos en train).
       El val oficial contiene los tipos no vistos -> la selección honesta está alineada con el test.
       The `color` (RGB) modality is used, for consistency with the rest of the benchmark."""
    base = os.path.join(ROOT, "CASIA-SURF", "Data-001", "Data")
    assert os.path.isdir(base), f"no existe {base}"
    rows = []
    for split, sub in [("train", "Training"), ("dev", "Val"), ("test", "Testing")]:
        for part in ("real_part", "fake_part"):
            pd = os.path.join(base, sub, part)
            if not os.path.isdir(pd): continue
            for subj in sorted(os.listdir(pd)):
                sd = os.path.join(pd, subj)
                if not os.path.isdir(sd): continue
                for rs in sorted(os.listdir(sd)):
                    if not rs.endswith(".rssdk"): continue
                    st = "live" if rs.startswith("real") else rs[:-len(".rssdk")]
                    label = 0 if st == "live" else 1
                    cd = os.path.join(sd, rs, "color")
                    for fr in _frames(cd):
                        rows.append(["CASIA-SURF", split, label, st, os.path.join(cd, fr)])
    return write("CASIA-SURF", rows)


def casia_fasd():
    """CASIA-FASD: the subtype comes from the video number in the filename (sNvVfF.png).
    Estándar CASIA-FASD (12 vídeos/sujeto):
      real  = 1, 2, HR_1     warped = 3, 4, HR_2     cut = 5, 6, HR_3     replay = 7, 8, HR_4
    OJO: la extracción metió TODOS los HR_* en spoof/, así que HR_1 (que es REAL) quedó etiquetado
    como ataque (~10.4k frames). Aquí la etiqueta se deriva del nº de vídeo, NO de la carpeta,
    lo que CORRIGE ese error. Se usan solo los ficheros sin prefijo (las variantes b*/f* aparecen
    only in train/live, undocumented, and would skew the balance)."""
    base = os.path.join(ROOT, "CASIA-FASD", "casia-fasd")
    VID2ST = {"1": "live", "2": "live", "HR_1": "live",
              "3": "warped", "4": "warped", "HR_2": "warped",
              "5": "cut", "6": "cut", "HR_3": "cut",
              "7": "replay", "8": "replay", "HR_4": "replay"}
    pat = re.compile(r"^s(\d+)v(HR_\d+|\d+)f(\d+)\.png$")   # sin prefijo b/f
    rows = []; fixed = 0
    for split in ("train", "test"):
        for cls in ("live", "spoof"):
            d = os.path.join(base, split, cls)
            if not os.path.isdir(d): continue
            for f in sorted(os.listdir(d)):
                m = pat.match(f)
                if not m: continue
                st = VID2ST.get(m.group(2))
                if st is None: continue
                label = 0 if st == "live" else 1
                if cls == "spoof" and label == 0: fixed += 1   # HR_1 rescatado
                rows.append(["CASIA-FASD", split, label, st, os.path.join(d, f)])
    print(f"  [CASIA-FASD] {fixed} frames HR_1 relabelled from attack to bonafide (correction)")
    return write("CASIA-FASD", rows)


def _frames(dirpath):
    try:
        return [f for f in os.listdir(dirpath) if f.lower().endswith((".jpg", ".png"))]
    except OSError:
        return []


def oulu():
    """AccessType is the last field of the folder name. Keeps OULU's OFFICIAL dev split."""
    base = os.path.join(ROOT, "oulu")
    rows = []
    for split, sub in [("train", "train_jpeg_256"), ("dev", "dev_jpeg_256"), ("test", "test_jpeg_256")]:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            print(f"  aviso: no existe {d}"); continue
        for vid in sorted(os.listdir(d)):
            parts = vid.split("_")               # v_spoof_g1_3_16_4
            if len(parts) < 6: continue
            at = parts[-1]
            st = OULU_TYPES.get(at, f"at{at}")
            label = 0 if at == "1" else 1
            for fr in _frames(os.path.join(d, vid)):
                rows.append(["OULU-NPU", split, label, st, os.path.join(d, vid, fr)])
    return write("OULU-NPU", rows)


def replay():
    """Idiap Replay-Attack: subtipo = dispositivo_medio (5 clases). Tiene train/dev/test OFICIALES.
    Carpeta: v_spoof_g{fixed|hand}_attack_{dev}_client..._session..._{dev}_{photo|video}_{luz}"""
    base = os.path.join(ROOT, "replay")
    rows = []
    for split, sub in [("train", "train_jpeg_256"), ("dev", "dev_jpeg_256"), ("test", "test_jpeg_256")]:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            print(f"  aviso: no existe {d}"); continue
        for vid in sorted(os.listdir(d)):
            p = vid.split("_")
            if "spoof" in vid and len(p) >= 9:
                st = f"{p[4]}_{p[8]}"; label = 1
            elif "live" in vid:
                st = "live"; label = 0
            else:
                continue
            for fr in _frames(os.path.join(d, vid)):
                rows.append(["replay", split, label, st, os.path.join(d, vid, fr)])
    return write("replay", rows)


def siw():
    """subtipo = tipo_medio (print_1/2, replay_1..4). Sin dev oficial: se talla luego por sujeto."""
    base = os.path.join(ROOT, "siw")
    rows = []
    for split, sub in [("train", "train_jpeg_256"), ("test", "test_jpeg_256")]:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            print(f"  aviso: no existe {d}"); continue
        for vid in sorted(os.listdir(d)):
            m = re.search(r"g(\d+)-(\d+)-(\d+)-(\d+)-(\d+)", vid)
            if not m: continue
            typ, med = m.group(3), m.group(4)
            label = 0 if typ == "1" else 1
            st = "live" if typ == "1" else f"{SIW_TYPES.get(typ, typ)}{med}"
            for fr in _frames(os.path.join(d, vid)):
                rows.append(["SiW", split, label, st, os.path.join(d, vid, fr)])
    return write("SiW", rows)


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
    if which in ("hqwmca", "all"): hqwmca()
    if which in ("surf", "all"): casia_surf()
    if which in ("fasd", "all"): casia_fasd()
    if which in ("celeba", "all"): celeba()
    if which in ("replay", "all"): replay()
    if which in ("oulu", "all"): oulu()
    if which in ("siw", "all"): siw()
