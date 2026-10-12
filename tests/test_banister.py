import numpy as np
from src.banister import ajuster, predire
from src.simulate import VRAIS_PARAMS, simuler


def test_sans_charge_performance_constante():
    w = np.zeros(50)
    assert np.allclose(predire((100, 0.1, 0.3, 40, 10), w), 100)


def test_une_seance_fatigue_puis_forme():
    # une séance au jour 0 : fatigue d'abord dominante, effet de forme plus tard
    w = np.zeros(120)
    w[0] = 100
    p = predire((0.0, 0.1, 0.3, 45.0, 10.0), w)
    assert p[1] < 0          # juste après : fatigue > forme
    assert p[60] > 0         # beaucoup plus tard : la fatigue s'est dissipée, un peu de forme reste


def test_retrouve_les_parametres_sans_bruit():
    w, perf, jours, _ = simuler(bruit=0.0)
    est, _ = ajuster(w, perf, jours)
    assert np.allclose(predire(est, w)[jours], perf, atol=1e-3)
