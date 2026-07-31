# Angular Coverage Governs Training-Set Selection in Face PAD — paquete de reproducción

Código y resultados derivados del artículo *"What to Train On, Not How to Order It: Angular
Coverage Governs Training-Set Selection in Face Presentation Attack Detection"*.

**Idea en una frase**: para cada tipo de ataque de presentación (PAI) se calcula su *eje
discriminante* `d_k = normalizar(media(ataque k) − media(bonafide))` sobre una red congelada, y la
*cobertura angular* de un conjunto de entrenamiento `S` es `cov(S) = media_k max_{j∈S} <d_k, d_j>`
—es decir, si cada ataque que verá el sistema tiene alguno parecido en el entrenamiento—. Esa
cantidad, calculable **sin entrenar nada**, predice el rendimiento del detector.

---

## Nivel 1 — reproducir el artículo (sin datasets, sin GPU)

Todo lo necesario está incluido: los resultados de **las 331 ejecuciones controladas** y los ejes
discriminantes ya calculados.

```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn, matplotlib
python src/reproduce_all.py            # pocos minutos
```

Regenera las estadísticas de la ley de cobertura, el análisis de olvido, el experimento del piloto,
la comprobación de taxonomía y **todas las figuras del artículo** en `figures/`.

Valores esperados (compara con tu salida para verificar):

| resultado | valor |
|---|---|
| Cobertura → AUC (HQ-WMCA, 39 celdas) | `r = +0.795`, `p = 1.5e-09` |
| Cobertura → ACER | `r = −0.856`, `p = 3.6e-12` |
| R² solo nº de PAIs → + cobertura | `0.469 → 0.773` |
| Olvido ↔ ángulo (180 transiciones) | `r = +0.275`, `p = 1.9e-04` |
| Piloto de 25 imágenes/PAI | pérdida de cobertura `+0.004` |
| DBSCAN no supervisado vs partición de PAIs | `ARI = 0.145` (y `−0.105` vs real/ataque) |

Análisis individuales:
```bash
python src/07_coverage_law.py HQ-WMCA      # la ley
python src/06_retro_forgetting.py HQ-WMCA  # olvido vs ángulo
python src/09_pilot.py HQ-WMCA             # planificación de captura
python src/11_geometry_viz.py HQ-WMCA 3000 # PCA/t-SNE/DBSCAN + taxonomía
python src/12_axial_space.py HQ-WMCA       # coordenadas axiales vs LDA (resultado negativo)
python src/14_paper_figures.py             # figuras
```

## Nivel 2 — re-entrenar desde cero (necesita datasets + GPU)

Requiere los siete datasets de anti-spoofing. **No se redistribuyen aquí**: cada uno tiene su propio
acuerdo de licencia. La estructura de carpetas que espera el código y **dónde se cambia la ruta**
están en **[`docs/DATASETS.md`](docs/DATASETS.md)**.

```bash
export FAS_DATA_ROOT=/ruta/a/tus/datasets
export DAXIS_OUT=/ruta/de/salida

pip install torch torchvision pillow opencv-python

python src/build_subtype_manifests.py all          # manifiestos con PAI por muestra
python src/00_extract_embeddings.py HQ-WMCA train resnet50 4000 cuda:0
python src/01_daxis_map.py HQ-WMCA                 # ejes + matriz de cosenos + órdenes
python src/08_daxis_select.py HQ-WMCA              # selección por cobertura (facility location)
python src/13_budget_select.py HQ-WMCA             # selección a presupuesto de imágenes fijo
```
Entrenamiento de una celda concreta:
```bash
python src/ibt_generic.py --manifest manifests/HQ-WMCA_subtype.csv --dataset HQ-WMCA \
       --design within --order standard --seed 0 --model resnet50 \
       --epochs_per_iter 3 --cap_per_class 4000 --batch_size 64 --tag mi-prueba
```

Coste aproximado de la campaña completa: **331 ejecuciones ≈ 200 GPU-hora** en RTX 2080 Ti.

---

## Contenido

```
src/                 código
  config.py            rutas (variables de entorno; por defecto apuntan dentro del repo)
  daxis_ext.py         geometría: ejes, cosenos, órdenes, scores por muestra
  ibt_generic.py       entrenador (soporta curriculum, órdenes custom y filtrado)
  build_subtype_manifests.py   adaptadores de los 7 datasets a un formato común
  00..14_*.py          extracción, geometría, análisis y figuras (numerados por orden de uso)
  reproduce_all.py     ejecuta todo el Nivel 1
results/
  results_daxis_campaign.csv        una fila por iteración de cada ejecución de selección
  results_curriculum_campaign.csv   ídem para los experimentos de curriculum
  artifacts/           ejes (axes_*.npz), órdenes, celdas, y embeddings de HQ-WMCA
figures/             figuras del artículo
docs/DATASETS.md     estructura esperada de cada dataset y dónde cambiar la ruta
```

### Sobre los embeddings incluidos
`results/artifacts/HQ-WMCA_train_resnet50.npz` (79 MB) son las features de la red congelada del
dataset principal, guardadas en **float16** para que el paquete quepa. La geometría no se ve
afectada: la diferencia máxima en la matriz de cosenos frente a float32 es **1.5e-06**, porque solo
se usan medias y productos escalares normalizados.

Los embeddings de los otros datasets no se incluyen (≈1 GB); se regeneran con
`00_extract_embeddings.py` una vez tengas los datasets.

---

## Notas de honestidad

Estos puntos están en el artículo y se repiten aquí para que no haya sorpresas al reproducir:

- **La ley es correlacional.** Se establece sobre ablaciones controladas del conjunto de
  entrenamiento, no sobre intervenciones en la cobertura a número de PAIs fijo.
- **No replica en dos de siete datasets, y ambos casos vienen con su causa.** Replay-Attack está
  saturado (AUC 99–100 con cualquier subconjunto, sin varianza que correlacionar) aunque sí sigue
  la ley en ACER (`r = −0.795`); CASIA-SURF es degenerado para este test (3 PAIs → cobertura
  0.984–0.993, sin variación). Se reportan, no se ocultan.
- **Maximizar la cobertura a ciegas NO gana de forma fiable.** El objetivo ignora cuántos datos
  aporta cada PAI: a presupuesto fijo, la estrategia de mayor cobertura pierde contra una de menor
  cobertura que conserva los PAIs grandes. La cobertura es un buen *descriptor* y un mal *objetivo*.
- **El curriculum guiado por geometría no funciona.** Diez variantes, incluido *rehearsal* angular,
  pierden contra entrenamiento conjunto con cómputo igualado.
- **El ACER es más informativo que el AUC** en estos análisis, porque el AUC satura cerca de 100.
- **CASIA-FASD trae un error de etiquetado** en la extracción habitual (10.359 caras reales
  marcadas como ataque); el adaptador de este repo lo corrige. Tenlo en cuenta al comparar con
  números publicados.

## Licencia y cita

Código y resultados derivados: **MIT** (ver `LICENSE`). No cubre los datasets.
Para citar, ver `CITATION.cff`.
