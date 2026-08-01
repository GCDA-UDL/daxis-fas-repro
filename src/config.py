"""
Centralised paths for the reproducibility package.

By default everything points INSIDE the repo, so the analyses and figures reproduce with no
configuration at all. Retraining additionally needs the datasets and an output directory:

    export FAS_DATA_ROOT=/path/to/datasets     # original images
    export DAXIS_OUT=/path/to/output           # checkpoints + results of new runs
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- input: resultados ya calculados (incluidos en el repo) ---
RESULTS_DIR = os.environ.get("DAXIS_RESULTS", os.path.join(ROOT, "results"))
ART = os.environ.get("DAXIS_ARTIFACTS", os.path.join(RESULTS_DIR, "artifacts"))
RES_DAXIS = os.path.join(RESULTS_DIR, "results_daxis_campaign.csv")
RES_CURRICULUM = os.path.join(RESULTS_DIR, "results_curriculum_campaign.csv")

# --- output ---
FIG_DIR = os.environ.get("DAXIS_FIGURES", os.path.join(ROOT, "figures"))
OUT = os.environ.get("DAXIS_OUT", os.path.join(ROOT, "runs"))

# --- datos originales (solo para re-train / re-extraer embeddings) ---
FAS_DATA_ROOT = os.environ.get("FAS_DATA_ROOT", "/mnt/d1/data")
MANIFEST_DIR = os.environ.get("DAXIS_MANIFESTS", os.path.join(ROOT, "manifests"))

for _d in (FIG_DIR, OUT):
    os.makedirs(_d, exist_ok=True)
