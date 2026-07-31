# Datasets: dónde ponerlos y qué estructura espera el código

Solo hace falta leer esto para el **Nivel 2** (re-entrenar / re-extraer embeddings). El Nivel 1
(reproducir todos los números y figuras del paper) funciona sin ningún dataset.

## Dónde se cambia la ruta

Una sola variable de entorno:

```bash
export FAS_DATA_ROOT=/ruta/a/tus/datasets      # por defecto: /mnt/d1/data
```

La usan `src/build_subtype_manifests.py` (construye los manifiestos) y
`src/00_extract_embeddings.py`. Está definida en `src/config.py`; si prefieres fijarla en código,
edita ahí `FAS_DATA_ROOT`.

Los datasets **no se redistribuyen**: cada uno requiere su propio acuerdo de licencia con la
institución que lo publica.

## Estructura esperada por dataset

El código deriva el **tipo de ataque (PAI)** de la estructura de carpetas o de los ficheros de
anotación, así que la disposición importa. `$ROOT` = `FAS_DATA_ROOT`.

---

### HQ-WMCA — *dataset principal del paper*
Fuente: Idiap. Se usa **solo la modalidad RGB**, extrayendo frames de los `.mov`.

```
$ROOT/HQ-WMCA/
  HQ-WMCA-RGB/HQ-WMCA-RGB/
    Bonafide/            *.mov
    Impersonation/       Flexiblemask/ Mannequin/ Papermask/ Print/ Replay/ Rigidmask/   (*.mov)
    Obfuscation/         Glasses/ Makeup/ Tattoo/ Wig/                                    (*.mov)
  frames/                <- lo genera el script de extracción
    Bonafide/<video>/frame000.jpg ...
    Rigidmask/<video>/...
```

**Paso previo obligatorio** (los `.mov` hay que convertirlos a frames):
```bash
python src/extract_hqwmca_frames.py     # 12 frames/vídeo, ~2900 vídeos
```
- **PAI** = carpeta (`Rigidmask`, `Makeup`, …); `Bonafide` → clase real.
- **Sujeto** = campo 3 del nombre WMCA (`1_01_<SUJETO>_...`). Se usa para que train/dev/test
  sean **disjuntos por sujeto** (70/15/15), porque este dataset no trae splits oficiales.

---

### CelebA-Spoof
Fuente: repositorio oficial (Google Drive, 74 partes `.zip`).

```
$ROOT/CelebA-Spoof/CelebA_Spoof/
  Data/train/<id>/{live,spoof}/*.png|jpg
  Data/test/<id>/{live,spoof}/*.png|jpg
  metas/intra_test/{train_label.json,test_label.json}     <- IMPRESCINDIBLE
```
> **Aviso**: la carpeta `metas/` está **solo en el archivo completo**. Si tu descarga se
> interrumpió (a nosotros nos pasó: 48 de 74 partes), tendrás `Data/` pero no `metas/`, y sin
> ella no hay tipos de ataque. Verifica que el zip lista `CelebA_Spoof/metas`.

- **PAI** = campo `[40]` del JSON (0 Live, 1 Photo, 2 Poster, 3 A4, 4 FaceMask, 5 UpperBodyMask,
  6 RegionMask, 7 PC, 8 Pad, 9 Phone, 10 3DMask); `[43]` = live/spoof.
- **Sujeto** = el `<id>` de la ruta.

---

### CASIA-SURF
Fuente: organizadores del CASIA-SURF challenge. Se usa la modalidad `color`.

```
$ROOT/CASIA-SURF/Data-001/Data/
  Training/{real_part,fake_part}/<sujeto>/<NN_x_y>.rssdk/color/*.jpg
  Val/      idem
  Testing/  idem
```
- **PAI** = nombre de la carpeta `.rssdk` (`04_en_b`, `05_enm_s`, `06_enm_b` en train;
  `01_e_s`, `02_e_b`, `03_en_s` en val/test).
- **Su protocolo oficial ya es cross-attack**: los ataques de train y los de val/test son
  **disjuntos por diseño**. Por eso el código **no filtra el dev** en este dataset — si lo
  filtrara por los subtipos de train, el dev se quedaría sin ataques y el umbral EER no existiría.

---

### CASIA-FASD
Fuente: CASIA. Se parte de frames ya extraídos.

```
$ROOT/CASIA-FASD/casia-fasd/
  train/{live,spoof}/sNvVfF.png
  test/{live,spoof}/sNvVfF.png
```
Nombre `s<sujeto>v<vídeo>f<frame>.png`, con `<vídeo>` ∈ {1..8, HR_1..HR_4}.

- **PAI** = número de vídeo: 1, 2, HR_1 → **real**; 3, 4, HR_2 → *warped*; 5, 6, HR_3 → *cut*;
  7, 8, HR_4 → *replay*.
- ⚠️ **Corrección importante**: la extracción habitual mete **todos** los `HR_*` en `spoof/`, así
  que **10.359 frames de caras auténticas (HR_1) quedan etiquetados como ataque**. El adaptador
  deriva la etiqueta del **número de vídeo, no de la carpeta**, lo que lo corrige. Si comparas con
  resultados publicados de CASIA-FASD, ten en cuenta que muchos arrastran ese error.
- Las variantes de nombre con prefijo `b*`/`f*` (solo en `train/live`, sin documentar) se excluyen.

---

### OULU-NPU / Replay-Attack / SiW
Los tres se usan como frames en carpetas por vídeo:

```
$ROOT/oulu/{train,dev,test}_jpeg_256/v_{live,spoof}_g<N>_<ses>_<user>_<AT>/frameNNNNNN.jpg
$ROOT/replay/{train,dev,test}_jpeg_256/v_{live,spoof}_g<fixed|hand>_attack_<disp>_client..._<disp>_<photo|video>_<luz>/*.jpg
$ROOT/siw/{train,test}_jpeg_256/v_{live,spoof}_g<subj>-<sensor>-<tipo>-<medio>-<ses>/*.jpg
```

| dataset | PAI (de dónde sale) | sujeto | splits |
|---|---|---|---|
| **OULU-NPU** | último campo = AccessType: 1 real, 2/3 print, 4/5 replay | campo 5 | oficiales (train/dev/test) |
| **Replay-Attack** | `dispositivo_medio`: highdef/mobile/print × photo/video | carpeta | oficiales |
| **SiW** | `tipo_medio` tras la `g`: tipo 1 real, 2 print, 3 replay | `g<subj>` | train/test; el dev se talla por sujeto |

> **Nota sobre saturación**: Replay-Attack y SiW alcanzan AUC ≈ 100 con todos los PAIs. No es un
> problema para la ley de cobertura —restringir el subconjunto de entrenamiento sigue generando
> varianza— pero sí explica por qué en Replay-Attack el AUC no correlaciona y el ACER sí.

---

## Comprobación rápida

```bash
export FAS_DATA_ROOT=/ruta/a/tus/datasets
python src/build_subtype_manifests.py all
```
Imprime, por dataset, el número de filas por split y por PAI. Si un dataset da 0 filas o le faltan
PAIs, la estructura no coincide con la de arriba. Los manifiestos resultantes se escriben en
`$DAXIS_MANIFESTS` (por defecto `manifests/` en la raíz del repo).

Con los manifiestos listos:
```bash
python src/00_extract_embeddings.py HQ-WMCA train resnet50 4000 cuda:0
python src/01_daxis_map.py HQ-WMCA        # regenera los ejes y los órdenes
```
