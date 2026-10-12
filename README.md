# Modélisation de la forme sportive (modèle de Banister)

> Projet personnel en cours — statut : étape 2/4 (validation sur données synthétiques).

## Objectif
Modéliser comment la charge d'entraînement produit à la fois de la **forme** (effet lent) et de la **fatigue** (effet rapide),
estimer les paramètres du modèle de Banister sur des données d'entraînement, mesurer leur incertitude, puis comparer avec un modèle de machine learning.

## Le modèle
p(t) = p0 + k1 · Σ w(s) e^(−(t−s)/τ1) − k2 · Σ w(s) e^(−(t−s)/τ2)

| Paramètre | Rôle |
|---|---|
| w(s) | charge d'entraînement du jour s |
| τ1 | durée de l'effet de forme (long) |
| τ2 | durée de la fatigue (court) |
| k1, k2 | gains de forme et de fatigue |

## Plan
- [x] Étape 1 : structure du dépôt
- [ ] Étape 2 : estimation validée sur données synthétiques (`scripts/run_synthetic.py`)
- [ ] Étape 3 : estimation sur données réelles + incertitude (bootstrap)
- [ ] Étape 4 : comparaison avec un modèle de ML + validation temporelle

## Utilisation
```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_synthetic.py
python -m pytest
```

## Résultats
_À compléter au fil du projet (graphiques, paramètres estimés, erreur de prédiction)._

## Limites
_À compléter : identifiabilité des paramètres, taille des données, biais de mesure de la performance._
