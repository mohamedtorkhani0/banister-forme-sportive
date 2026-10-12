"""Génère des données synthétiques dont on CONNAÎT les vrais paramètres.

Sert à vérifier que l'estimation retrouve les bons paramètres avant de
toucher aux vraies données.
"""
import numpy as np
from .banister import predire

VRAIS_PARAMS = (100.0, 0.10, 0.28, 45.0, 11.0)  # p0, k1, k2, tau1, tau2


def charge_realiste(n_jours=240, graine=0):
    """Charge hebdomadaire plausible : 4-5 séances/semaine, semaines dures puis légères."""
    rng = np.random.default_rng(graine)
    w = np.zeros(n_jours)
    for j in range(n_jours):
        jour_semaine = j % 7
        semaine = j // 7
        facteur_semaine = 0.6 if semaine % 4 == 3 else 1.0  # 1 semaine légère sur 4
        if jour_semaine in (1, 3, 5) or (jour_semaine == 6 and rng.random() < 0.5):
            w[j] = rng.gamma(shape=6.0, scale=10.0 * facteur_semaine)
    return w


def simuler(params=VRAIS_PARAMS, n_jours=240, bruit=2.0, tous_les=4, graine=0):
    """Retourne (charge, perf_mesuree, jours_mesures, perf_vraie)."""
    rng = np.random.default_rng(graine)
    w = charge_realiste(n_jours, graine)
    vraie = predire(params, w)
    jours = np.arange(5, n_jours, tous_les)  # un test de performance tous les 4 jours
    perf = vraie[jours] + rng.normal(0, bruit, size=len(jours))
    return w, perf, jours, vraie
