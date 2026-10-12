"""Étape 2 : vérifier que l'estimation retrouve les vrais paramètres (données synthétiques)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.banister import PARAM_NAMES, ajuster, predire
from src.simulate import VRAIS_PARAMS, simuler

w, perf, jours, vraie = simuler()
est, sol = ajuster(w, perf, jours)

print(f"{'param':>6} {'vrai':>9} {'estimé':>9}")
for nom, v, e in zip(PARAM_NAMES, VRAIS_PARAMS, est):
    print(f"{nom:>6} {v:9.3f} {e:9.3f}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(vraie, label="performance vraie", color="0.6")
ax.plot(predire(est, w), label="modèle ajusté")
ax.scatter(jours, perf, s=12, label="mesures bruitées", color="C3")
ax.set_xlabel("jour")
ax.set_ylabel("performance")
ax.legend()
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/synthetique_ajustement.png", dpi=150)
print("figure enregistrée dans figures/synthetique_ajustement.png")
