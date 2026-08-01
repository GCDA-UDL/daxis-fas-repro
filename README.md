# Angular coverage for training-set selection in face PAD — reproducibility package

Code and derived results for the paper "What to Train On, Not How to Order It: Angular Coverage
Governs Training-Set Selection in Face Presentation Attack Detection".

For each presentation attack instrument (PAI) we compute a discriminant axis on a frozen network,

    d_k = normalise( mean(features of attack k) - mean(features of bonafide) )

and define the angular coverage of a training set S as `cov(S) = mean_k max_{j in S} <d_k, d_j>`,
i.e. whether every attack the system will face has something similar in training. Coverage requires
no training to compute, and it predicts detector performance.

## Level 1: reproducing the paper (no datasets, no GPU)

The outcomes of all 331 controlled runs and the pre-computed discriminant axes are bundled here, so
the analysis runs on its own.

```bash
pip install -r requirements.txt        # numpy, scipy, scikit-learn, matplotlib
python src/reproduce_all.py
```

This takes a few minutes and regenerates the coverage-law statistics, the forgetting analysis, the
pilot experiment, the taxonomy check, and the paper figures into `figures/`.

Values to compare your output against:

| result | value |
|---|---|
| Coverage vs AUC (HQ-WMCA, 39 cells) | r = +0.795, p = 1.5e-09 |
| Coverage vs ACER | r = -0.856, p = 3.6e-12 |
| R² from PAI count alone, then adding coverage | 0.469 to 0.773 |
| Forgetting vs angle (180 transitions) | r = +0.275, p = 1.9e-04 |
| Pilot of 25 images per PAI | coverage loss +0.004 |
| Unsupervised DBSCAN against the PAI partition | ARI = 0.145 (and -0.105 against bonafide/attack) |

Individual analyses:

```bash
python src/07_coverage_law.py HQ-WMCA      # the law
python src/06_retro_forgetting.py HQ-WMCA  # forgetting against angle
python src/09_pilot.py HQ-WMCA             # capture planning
python src/11_geometry_viz.py HQ-WMCA 3000 # PCA/t-SNE/DBSCAN and the taxonomy
python src/12_axial_space.py HQ-WMCA       # axial coordinates against LDA (a negative result)
python src/14_paper_figures.py             # figures
```

## Level 2: retraining from scratch (datasets and a GPU required)

This needs the seven face anti-spoofing datasets. They are not redistributed here, since each
carries its own licence agreement. The layout the code expects, and where to point it, are in
[`docs/DATASETS.md`](docs/DATASETS.md).

```bash
export FAS_DATA_ROOT=/path/to/your/datasets
export DAXIS_OUT=/path/to/output

pip install torch torchvision pillow opencv-python

python src/build_subtype_manifests.py all          # manifests carrying a PAI label per sample
python src/00_extract_embeddings.py HQ-WMCA train resnet50 4000 cuda:0
python src/01_daxis_map.py HQ-WMCA                 # axes, cosine matrix, orderings
python src/08_daxis_select.py HQ-WMCA              # coverage selection (facility location)
python src/13_budget_select.py HQ-WMCA             # selection at a fixed image budget
```

Training a single cell:

```bash
python src/ibt_generic.py --manifest manifests/HQ-WMCA_subtype.csv --dataset HQ-WMCA \
       --design within --order standard --seed 0 --model resnet50 \
       --epochs_per_iter 3 --cap_per_class 4000 --batch_size 64 --tag my-run
```

The full campaign is 331 runs, roughly 200 GPU-hours on RTX 2080 Ti.

## Contents

```
src/                 code
  config.py            paths (environment variables, defaulting inside the repo)
  daxis_ext.py         geometry: axes, cosines, orderings, per-sample scores
  ibt_generic.py       trainer (curricula, custom orderings, filtering)
  build_subtype_manifests.py   adapters mapping the 7 datasets to a common format
  00..14_*.py          extraction, geometry, analysis, figures (numbered by order of use)
  reproduce_all.py     runs all of Level 1
results/
  results_daxis_campaign.csv        one row per iteration of each selection run
  results_curriculum_campaign.csv   the same for the curriculum experiments
  artifacts/           axes (axes_*.npz), orderings, cells, HQ-WMCA embeddings
figures/             paper figures
docs/DATASETS.md     expected layout of each dataset, and where to change the path
```

`results/artifacts/HQ-WMCA_train_resnet50.npz` (79 MB) holds the frozen-network features of the
primary dataset, stored as float16 to keep the package to a reasonable size. This does not affect
the geometry: the largest change in the cosine matrix against float32 is 1.5e-06, since only means
and normalised dot products are involved. Embeddings for the other datasets come to about 1 GB and
are not included; regenerate them with `00_extract_embeddings.py`.

## Limitations and negative results

All of these appear in the paper. They are repeated here so that nothing about the numbers you
obtain comes as a surprise.

The law is correlational. It rests on controlled ablations of the training set, not on
interventions on coverage at a fixed number of PAIs.

It does not replicate on two of the seven datasets, and in both cases the reason is identifiable.
Replay-Attack is saturated: with any reasonable subset its AUC sits at 99-100, leaving no variance
to correlate, although it does follow the law on ACER (r = -0.795). CASIA-SURF is degenerate for
this test, since three PAIs admit only m=2 subsets whose coverage spans 0.984 to 0.993. Both are
reported rather than dropped.

Maximising coverage directly does not reliably win. The objective ignores how much data each PAI
contributes, so at a fixed image budget the highest-coverage strategy loses to a lower-coverage one
that keeps the large PAIs. Coverage works as a descriptor and fails as an objective.

Geometry-driven curricula do not work. Ten variants, angular rehearsal among them, lose to
compute-matched joint training.

ACER carries more signal than AUC in these analyses, because AUC saturates near 100 on most PAD
benchmarks.

CASIA-FASD carries a labelling error in the usual frame export, where 10,359 genuine faces are
marked as attacks. The adapter here corrects it, which is worth knowing when comparing against
published numbers.

## Licence and citation

Code and derived results are MIT (see `LICENSE`); this does not extend to the datasets. For
citation details see `CITATION.cff`.
