#!/usr/bin/env python3
"""
Reproduce EVERY number and figure in the paper from the bundled results.

Needs neither the original datasets nor a GPU: it starts from results/ (the campaign CSVs) and
results/artifacts/ (the pre-computed discriminant axes). Typical runtime: a few minutes.

    python reproduce_all.py

To RETRAIN from scratch (needs datasets + GPU) see the README, "Level 2".
"""
import subprocess, sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
STEPS = [
    ("Ley de coverage (HQ-WMCA): r y R2 de la Tabla II",   ["07_coverage_law.py", "HQ-WMCA"]),
    ("Olvido vs angulo (Sec. VIII-A): r=+0.275",            ["06_retro_forgetting.py", "HQ-WMCA"]),
    ("Piloto de captura (Tabla V): 25 imgs/PAI",            ["09_pilot.py", "HQ-WMCA"]),
    ("Geometria + taxonomia (Fig. 3): ARI DBSCAN",          ["11_geometry_viz.py", "HQ-WMCA", "3000"]),
    ("Figuras del paper (Figs. 1, 2, 4)",                   ["14_paper_figures.py"]),
]
fail = 0
for i, (desc, cmd) in enumerate(STEPS, 1):
    print(f"\n{'='*78}\n[{i}/{len(STEPS)}] {desc}\n{'='*78}", flush=True)
    t = time.time()
    r = subprocess.run([PY] + cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  !! FALLO (codigo {r.returncode})"); fail += 1
    else:
        print(f"  ok ({time.time()-t:.0f}s)")
print(f"\n{'='*78}")
print("TODO REPRODUCIDO" if not fail else f"{fail} paso(s) fallaron")
print("Figuras en ../figures/ · compara con las del paper para verificar.")
sys.exit(1 if fail else 0)
