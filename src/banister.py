"""Modèle de Banister (impulsion-réponse) : forme et fatigue d'un sportif.

p(t) = p0 + k1 * sum_{s<t} w(s) exp(-(t-s)/tau1) - k2 * sum_{s<t} w(s) exp(-(t-s)/tau2)

- w(s)  : charge d'entraînement du jour s (0 les jours de repos)
- tau1  : durée caractéristique de l'effet de forme (long, ~ 30-60 jours)
- tau2  : durée caractéristique de la fatigue (court, ~ 5-15 jours)
- k1,k2 : gains associés (k2 > k1 en général : la fatigue est plus intense mais plus brève)
"""
import numpy as np
from scipy.optimize import least_squares

PARAM_NAMES = ["p0", "k1", "k2", "tau1", "tau2"]


def _filtre_exponentiel(w, tau):
    """f[t] = sum_{s<t} w[s] * exp(-(t-s)/tau), calculé par récurrence en O(n)."""
    a = np.exp(-1.0 / tau)
    f = np.zeros(len(w))
    for t in range(1, len(w)):
        f[t] = a * (f[t - 1] + w[t - 1])
    return f


def predire(params, w):
    """Performance prédite pour chaque jour, à partir de la charge w."""
    p0, k1, k2, tau1, tau2 = params
    return p0 + k1 * _filtre_exponentiel(w, tau1) - k2 * _filtre_exponentiel(w, tau2)


def ajuster(w, perf, jours_mesures, n_departs=20, graine=0):
    """Estime (p0, k1, k2, tau1, tau2) par moindres carrés non linéaires.

    w             : charge pour TOUS les jours (longueur n)
    perf          : performances mesurées (seulement certains jours)
    jours_mesures : indices des jours où perf est mesurée
    n_departs     : nombre de points de départ aléatoires (le problème a des minima locaux)
    """
    rng = np.random.default_rng(graine)
    perf = np.asarray(perf, dtype=float)
    idx = np.asarray(jours_mesures, dtype=int)

    def residus(params):
        return predire(params, w)[idx] - perf

    bas = [-np.inf, 0.0, 0.0, 5.0, 1.0]
    haut = [np.inf, np.inf, np.inf, 120.0, 40.0]
    meilleur = None
    for _ in range(n_departs):
        x0 = [
            perf.mean(),
            rng.uniform(0.01, 0.5),
            rng.uniform(0.01, 1.0),
            rng.uniform(20, 70),
            rng.uniform(3, 20),
        ]
        sol = least_squares(residus, x0, bounds=(bas, haut))
        if meilleur is None or sol.cost < meilleur.cost:
            meilleur = sol
    return meilleur.x, meilleur
