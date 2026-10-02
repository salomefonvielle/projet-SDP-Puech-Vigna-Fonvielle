"""Apprentissage de modèles MR-Sort par programmation linéaire mixte (Gurobi).

Conventions :
- X : tableau (m, n) des performances, plus grand = meilleur sur chaque critère.
- y : tableau (m,) des classes, entiers de 0 (pire) à p-1 (meilleure).
- Un modèle MR-Sort est un triplet (w, lam, B) où B est un tableau (p-1, n) :
  B[k] est la frontière entre la classe k et la classe k+1 (B[k] <= B[k+1]).
"""

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Modèle MR-Sort
# ---------------------------------------------------------------------------

def classify(X, w, lam, B):
    """Affecte chaque objet à une classe selon le modèle MR-Sort (w, lam, B).

    x est au moins dans la classe k+1 si la coalition des critères où x >= B[k]
    pèse au moins lam. Les frontières étant ordonnées, la classe est le nombre
    de frontières franchies.
    """
    X = np.atleast_2d(X)
    scores = ((X[:, None, :] >= B[None, :, :]) * w).sum(axis=2)  # (m, p-1)
    return (scores >= lam - 1e-9).sum(axis=1)


def random_model(n, p, rng):
    """Tire un modèle MR-Sort aléatoire : poids, seuil et frontières ordonnées."""
    w = rng.random(n)
    w /= w.sum()
    lam = rng.uniform(0.5, 1.0)
    B = np.sort(rng.random((p - 1, n)), axis=0)
    return w, lam, B


def random_objects(m, n, rng, decimals=2):
    """Tire m objets uniformément dans [0, 1]^n (valeurs arrondies)."""
    return np.round(rng.random((m, n)), decimals)


# ---------------------------------------------------------------------------
# Apprentissage (questions vii et viii)
# ---------------------------------------------------------------------------

def learn_mrsort(X, y, p, eps=1e-3, time_limit=60, verbose=False):
    """Apprend (w, lam, B) qui restitue au mieux le learning set (X, y).

    Les données sont normalisées dans [0, 1] critère par critère, ce qui permet
    de prendre M = 2 (question v). On maximise la marge alpha (question iv).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    m, n = X.shape

    # Normalisation dans [0, 1]
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / span
    M = 2.0

    model = gp.Model("inv-MR-Sort")
    model.Params.OutputFlag = int(verbose)
    model.Params.TimeLimit = time_limit

    w = model.addVars(n, lb=0, ub=1, name="w")
    lam = model.addVar(lb=0, ub=1, name="lambda")
    b = model.addVars(p - 1, n, lb=0, ub=1 + eps, name="b")
    alpha = model.addVar(lb=-1, ub=1, name="alpha")

    model.addConstr(w.sum() == 1)
    # Frontières ordonnées : b^{k} <= b^{k+1}
    for k in range(1, p - 1):
        for i in range(n):
            model.addConstr(b[k - 1, i] <= b[k, i])

    def score(j, k):
        """Ajoute delta_i(x), w_i(x) pour l'objet j et la frontière k ; renvoie s_k(x)."""
        d = model.addVars(n, vtype=GRB.BINARY)
        wx = model.addVars(n, lb=0, ub=1)
        for i in range(n):
            diff = Xn[j, i] - b[k, i]
            model.addConstr(diff >= M * (d[i] - 1))        # (1)
            model.addConstr(diff <= M * d[i] - eps)        # (2)
            model.addConstr(wx[i] <= w[i])                 # (3)
            model.addConstr(wx[i] <= d[i])                 # (4)
            model.addConstr(wx[i] >= d[i] + w[i] - 1)      # (4)
        return wx.sum()

    for j in range(m):
        c = y[j]
        if c >= 1:          # x doit franchir la frontière inférieure de sa classe
            model.addConstr(score(j, c - 1) >= lam + alpha)
        if c <= p - 2:      # x ne doit pas franchir la frontière supérieure
            model.addConstr(score(j, c) <= lam - alpha)

    model.setObjective(alpha, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"Pas de solution (statut Gurobi {model.Status})")

    w_val = np.array([w[i].X for i in range(n)])
    B_val = np.array([[b[k, i].X for i in range(n)] for k in range(p - 1)])
    # Gurobi respecte x_i >= b_i à sa tolérance près (~1e-6) : on abaisse b de
    # eps/2, ce qui reste sous les x validés et au-dessus des x non validés.
    B_val = B_val - eps / 2
    B_val = lo + B_val * span  # retour à l'échelle d'origine
    return w_val, lam.X, B_val, alpha.X


# ---------------------------------------------------------------------------
# Protocole de test du Jalon 1
# ---------------------------------------------------------------------------

def run_experiment(n=4, p=2, m_train=50, m_test=1000, runs=20, seed=0, quiet=False, **kw):
    """Génère un modèle de référence, apprend, mesure l'accord sur un jeu test.

    Renvoie la liste des taux d'accord et la liste des temps de résolution.
    """
    rng = np.random.default_rng(seed)
    scores, times = [], []
    for r in range(runs):
        w0, lam0, B0 = random_model(n, p, rng)
        X = random_objects(m_train, n, rng)
        y = classify(X, w0, lam0, B0)
        t0 = time.perf_counter()
        w1, lam1, B1, alpha = learn_mrsort(X, y, p, **kw)
        times.append(time.perf_counter() - t0)

        T = random_objects(m_test, n, rng)
        acc = np.mean(classify(T, w0, lam0, B0) == classify(T, w1, lam1, B1))
        train_acc = np.mean(classify(X, w1, lam1, B1) == y)
        scores.append(acc)
        if not quiet:
            print(f"run {r + 1:2d} : alpha = {alpha:.3f}, "
                  f"restitution L = {train_acc:.0%}, accord sur T = {acc:.1%}")
    print(f"n={n}, p={p}, |L|={m_train} : qualité moyenne sur {runs} essais "
          f"{np.mean(scores):.1%} (écart-type {np.std(scores):.1%}), "
          f"temps moyen {np.mean(times):.2f} s")
    return scores, times


# ---------------------------------------------------------------------------
# Lecture d'un learning set et ligne de commande
# ---------------------------------------------------------------------------

def load_learning_set(path):
    """Lit un CSV : une ligne par objet, les n premières colonnes sont les
    performances, la dernière est la classe. Une ligne d'en-tête est tolérée.

    Les classes peuvent être des entiers quelconques (0..p-1, 1..p, ...) : elles
    sont renumérotées de 0 à p-1 dans l'ordre croissant.
    """
    raw = np.genfromtxt(path, delimiter=",", dtype=float)
    raw = raw[~np.isnan(raw).any(axis=1)]  # retire l'en-tête éventuel
    X, labels = raw[:, :-1], raw[:, -1]
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)
    return X, y, classes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apprentissage MR-Sort (Gurobi)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_learn = sub.add_parser("learn", help="apprend un modèle sur un learning set CSV")
    p_learn.add_argument("csv")
    p_learn.add_argument("-p", type=int, default=None,
                         help="nombre de catégories (par défaut : déduit des données)")

    p_test = sub.add_parser("test", help="protocole de test du Jalon 1")
    p_test.add_argument("-n", type=int, default=4, help="nombre de critères")
    p_test.add_argument("-p", type=int, default=2, help="nombre de catégories")
    p_test.add_argument("-m", type=int, default=50, help="taille du learning set")
    p_test.add_argument("--runs", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "learn":
        X, y, classes = load_learning_set(args.csv)
        p = args.p or len(classes)
        w, lam, B, alpha = learn_mrsort(X, y, p)
        acc = np.mean(classify(X, w, lam, B) == y)
        np.set_printoptions(precision=3, suppress=True)
        print(f"n = {X.shape[1]} critères, p = {p} catégories, {len(y)} objets")
        print(f"poids w     = {w}")
        print(f"seuil lambda = {lam:.3f}")
        for k in range(p - 1):
            print(f"frontière b^{k + 1} = {B[k]}")
        print(f"marge alpha = {alpha:.3f}, restitution du learning set = {acc:.1%}")
    else:
        run_experiment(n=args.n, p=args.p, m_train=args.m, runs=args.runs)


if __name__ == "__main__":
    main()
