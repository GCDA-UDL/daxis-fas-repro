# Datasets: where to put them and what structure the code expects

You only need this for **Level 2** (retraining / re-extracting embeddings). Level 1 (reproducing
every number and figure in the paper) works without any dataset.

## Where to change the path

A single environment variable:

```bash
export FAS_DATA_ROOT=/path/to/your/datasets      # default: /mnt/d1/data
```

It is used by `src/build_subtype_manifests.py` (which builds the manifests) and
`src/00_extract_embeddings.py`. It is defined in `src/config.py`; if you would rather hard-code it,
edit `FAS_DATA_ROOT` there.

The datasets are **not redistributed**: each requires its own licence agreement with the
institution that publishes it.

## Expected structure per dataset

The code derives the **attack type (PAI)** from the folder structure or from the annotation files,
so the layout matters. `$ROOT` = `FAS_DATA_ROOT`.

---

### HQ-WMCA (the primary dataset in the paper)
Source: Idiap. Only the **RGB modality** is used, extracting frames from the `.mov` files.

```
$ROOT/HQ-WMCA/
  HQ-WMCA-RGB/HQ-WMCA-RGB/
    Bonafide/            *.mov
    Impersonation/       Flexiblemask/ Mannequin/ Papermask/ Print/ Replay/ Rigidmask/   (*.mov)
    Obfuscation/         Glasses/ Makeup/ Tattoo/ Wig/                                    (*.mov)
  frames/                <- produced by the extraction script
    Bonafide/<video>/frame000.jpg ...
    Rigidmask/<video>/...
```

Mandatory first step (the `.mov` files must be turned into frames):
```bash
python src/extract_hqwmca_frames.py     # 12 frames/video, ~2900 videos
```
- **PAI** = the folder (`Rigidmask`, `Makeup`, …); `Bonafide` → genuine class.
- **Subject** = field 3 of the WMCA filename (`1_01_<SUBJECT>_...`). It is used to make
  train/dev/test **subject-disjoint** (70/15/15), since this dataset ships no official splits.

---

### CelebA-Spoof
Source: the official repository (Google Drive, 74 `.zip` parts).

```
$ROOT/CelebA-Spoof/CelebA_Spoof/
  Data/train/<id>/{live,spoof}/*.png|jpg
  Data/test/<id>/{live,spoof}/*.png|jpg
  metas/intra_test/{train_label.json,test_label.json}     <- REQUIRED
```
> Warning: the `metas/` folder is **only in the complete archive**. If your download was
> interrupted (ours was: 48 of 74 parts) you will have `Data/` but no `metas/`, and without it
> there are no attack types. Check that the zip lists `CelebA_Spoof/metas`.

- **PAI** = field `[40]` of the JSON (0 Live, 1 Photo, 2 Poster, 3 A4, 4 FaceMask, 5 UpperBodyMask,
  6 RegionMask, 7 PC, 8 Pad, 9 Phone, 10 3DMask); `[43]` = live/spoof.
- **Subject** = the `<id>` in the path.

---

### CASIA-SURF
Source: the CASIA-SURF challenge organisers. The `color` modality is used.

```
$ROOT/CASIA-SURF/Data-001/Data/
  Training/{real_part,fake_part}/<subject>/<NN_x_y>.rssdk/color/*.jpg
  Val/      same
  Testing/  same
```
- **PAI** = the name of the `.rssdk` folder (`04_en_b`, `05_enm_s`, `06_enm_b` in train;
  `01_e_s`, `02_e_b`, `03_en_s` in val/test).
- **Its official protocol is already cross-attack**: the training attacks and the val/test attacks
  are **disjoint by design**. The code therefore does **not** filter the dev split for this dataset
  — filtering it by the training subtypes would leave dev with no attacks at all and the EER
  threshold would not exist.

---

### CASIA-FASD
Source: CASIA. Starts from already-extracted frames.

```
$ROOT/CASIA-FASD/casia-fasd/
  train/{live,spoof}/sNvVfF.png
  test/{live,spoof}/sNvVfF.png
```
Filename `s<subject>v<video>f<frame>.png`, with `<video>` ∈ {1..8, HR_1..HR_4}.

- **PAI** = the video number: 1, 2, HR_1 → **genuine**; 3, 4, HR_2 → *warped*; 5, 6, HR_3 → *cut*;
  7, 8, HR_4 → *replay*.
- Important correction: the usual export places **every** `HR_*` video in `spoof/`, so
  **10,359 frames of genuine faces (HR_1) end up labelled as attacks**. The adapter derives the
  label from the **video number, not the folder**, which corrects this. If you compare against
  published CASIA-FASD results, note that many carry that error.
- Filename variants prefixed `b*`/`f*` (present only in `train/live`, undocumented) are excluded.

---

### OULU-NPU / Replay-Attack / SiW
All three are used as frames in per-video folders:

```
$ROOT/oulu/{train,dev,test}_jpeg_256/v_{live,spoof}_g<N>_<ses>_<user>_<AT>/frameNNNNNN.jpg
$ROOT/replay/{train,dev,test}_jpeg_256/v_{live,spoof}_g<fixed|hand>_attack_<dev>_client..._<dev>_<photo|video>_<light>/*.jpg
$ROOT/siw/{train,test}_jpeg_256/v_{live,spoof}_g<subj>-<sensor>-<type>-<medium>-<ses>/*.jpg
```

| dataset | PAI (where it comes from) | subject | splits |
|---|---|---|---|
| **OULU-NPU** | last field = AccessType: 1 genuine, 2/3 print, 4/5 replay | field 5 | official (train/dev/test) |
| **Replay-Attack** | `device_medium`: highdef/mobile/print × photo/video | folder | official |
| **SiW** | `type_medium` after the `g`: type 1 genuine, 2 print, 3 replay | `g<subj>` | train/test; dev carved by subject |

> Note on saturation: Replay-Attack and SiW reach AUC ≈ 100 with all PAIs. That is not a
> problem for the coverage law — restricting the training subset still produces variance — but it
> does explain why on Replay-Attack the AUC does not correlate while the ACER does.

---

## Quick check

```bash
export FAS_DATA_ROOT=/path/to/your/datasets
python src/build_subtype_manifests.py all
```
This prints, per dataset, the number of rows per split and per PAI. If a dataset yields 0 rows or
is missing PAIs, the structure does not match the layout above. The resulting manifests are written
to `$DAXIS_MANIFESTS` (default: `manifests/` at the repo root).

Once the manifests exist:
```bash
python src/00_extract_embeddings.py HQ-WMCA train resnet50 4000 cuda:0
python src/01_daxis_map.py HQ-WMCA        # regenerates the axes and the orderings
```
