"""Rutas centralizadas del paquete de reproducción.

Por defecto todo apunta DENTRO del repo, así que los análisis y las figuras se reproducen sin
configurar nada. Para re-entrenar hace falta apuntar a los datasets y a un directorio de salida:

    export FAS_DATA_ROOT=/ruta/a/los/datasets     # imágenes originales
    export DAXIS_OUT=/ruta/de/salida              # checkpoints + results de las nuevas ejecuciones
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- entrada: resultados ya calculados (incluidos en el repo) ---
RESULTS_DIR = os.environ.get("DAXIS_RESULTS", os.path.join(ROOT, "results"))
ART = os.environ.get("DAXIS_ARTIFACTS", os.path.join(RESULTS_DIR, "artifacts"))
RES_DAXIS = os.path.join(RESULTS_DIR, "results_daxis_campaign.csv")
RES_CURRICULUM = os.path.join(RESULTS_DIR, "results_curriculum_campaign.csv")

# --- salida ---
FIG_DIR = os.environ.get("DAXIS_FIGURES", os.path.join(ROOT, "figures"))
OUT = os.environ.get("DAXIS_OUT", os.path.join(ROOT, "runs"))

# --- datos originales (solo para re-entrenar / re-extraer embeddings) ---
FAS_DATA_ROOT = os.environ.get("FAS_DATA_ROOT", "/mnt/d1/data")
MANIFEST_DIR = os.environ.get("DAXIS_MANIFESTS", os.path.join(ROOT, "manifests"))

for _d in (FIG_DIR, OUT):
    os.makedirs(_d, exist_ok=True)
