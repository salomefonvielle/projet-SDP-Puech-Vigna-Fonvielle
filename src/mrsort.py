# Projet SDP - apprentissage d'un modèle MR-Sort avec Gurobi
# X : performances (m objets x n critères), y : classes de 0 à p-1
# modèle = (w, lam, B) avec B[k] la frontière entre les classes k et k+1

import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB


def classify(X, w, lam, B):
    # classe = nombre de frontières franchies (poids des critères validés >= lam)
    X = np.atleast_2d(X)
    scores = ((X[:, None, :] >= B[None, :, :]) * w).sum(axis=2)  # (m, p-1)
    return (scores >= lam - 1e-9).sum(axis=1)


def random_model(n, p, rng):
    # modèle aléatoire : poids normalisés, lam entre 0.5 et 1, frontières triées
    w = rng.random(n)
    w /= w.sum()
    lam = rng.uniform(0.5, 1.0)
    B = np.sort(rng.random((p - 1, n)), axis=0)
    return w, lam, B


def random_objects(m, n, rng, decimals=2):
    # m objets au hasard dans [0,1]^n
    return np.round(rng.random((m, n)), decimals)


def learn_mrsort(X, y, p, eps=1e-3, time_limit=60, verbose=False):
    # MILP de la question vii (généralisé à p classes), on maximise alpha
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    m, n = X.shape

    # normalisation dans [0,1] -> M = 2 suffit
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
    # frontières ordonnées
    for k in range(1, p - 1):
        for i in range(n):
            model.addConstr(b[k - 1, i] <= b[k, i])

    def score(j, k):
        # variables delta et w(x) de l'objet j pour la frontière k
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
        if c >= 1:  # franchit la frontière du dessous
            model.addConstr(score(j, c - 1) >= lam + alpha)
        if c <= p - 2:  # ne franchit pas celle du dessus
            model.addConstr(score(j, c) <= lam - alpha)

    model.setObjective(alpha, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        raise RuntimeError(f"Pas de solution (statut Gurobi {model.Status})")

    w_val = np.array([w[i].X for i in range(n)])
    B_val = np.array([[b[k, i].X for i in range(n)] for k in range(p - 1)])
    # on baisse b de eps/2 à cause de la tolérance de Gurobi (sinon x = b pose problème)
    B_val = B_val - eps / 2
    B_val = lo + B_val * span  # retour à l'échelle de départ
    return w_val, lam.X, B_val, alpha.X


def run_experiment(n=4, p=2, m_train=50, m_test=1000, runs=20, seed=0, quiet=False, **kw):
    # protocole du jalon 1 : modèle de référence -> learning set -> apprentissage -> test
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


def load_learning_set(path):
    # CSV : n colonnes de performances puis la classe (classes renumérotées de 0 à p-1)
    raw = np.genfromtxt(path, delimiter=",", dtype=float)
    raw = raw[~np.isnan(raw).any(axis=1)]  # enlève l'en-tête
    X, labels = raw[:, :-1], raw[:, -1]
    classes = np.unique(labels)
    y = np.searchsorted(classes, labels)
    return X, y, classes


def center_frontiers(B, X):
    # met chaque frontière au milieu entre la valeur juste en dessous et celle juste
    # au-dessus (les affectations ne changent pas)
    B = B.copy()
    for k in range(B.shape[0]):
        for i in range(B.shape[1]):
            below = X[X[:, i] < B[k, i], i]
            above = X[X[:, i] >= B[k, i], i]
            if below.size and above.size:
                B[k, i] = (below.max() + above.min()) / 2
            elif above.size:
                B[k, i] = above.min()
    return B


def format_model(w, lam, B, alpha, acc, n, p, m):
    lines = [f"n = {n} critères, p = {p} catégories, {m} objets",
             "poids w      = " + "  ".join(f"{v:.3f}" for v in w),
             f"seuil lambda = {lam:.3f}"]
    for k in range(p - 1):
        lines.append(f"frontière b^{k + 1} = " + "  ".join(f"{v:g}" for v in B[k]))
    lines.append(f"marge alpha = {alpha:.3f}, restitution du learning set = {acc:.1%}")
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Apprentissage MR-Sort (Gurobi)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_learn = sub.add_parser("learn", help="apprend un modèle sur un learning set CSV")
    p_learn.add_argument("csv")
    p_learn.add_argument("-p", type=int, default=None,
                         help="nombre de catégories (par défaut : déduit des données)")
    p_learn.add_argument("-o", "--out", default=None,
                         help="fichier texte où enregistrer les paramètres appris")

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
        B = center_frontiers(B, X)
        acc = np.mean(classify(X, w, lam, B) == y)
        text = format_model(w, lam, B, alpha, acc, X.shape[1], p, len(y))
        print(text)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(f"Modèle MR-Sort appris sur {args.csv}\n" + text + "\n")
    else:
        run_experiment(n=args.n, p=args.p, m_train=args.m, runs=args.runs)


if __name__ == "__main__":
    main()
