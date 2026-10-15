# TP Calcul scientifique - Regression lineaire simple y = a*x + b
# Prototype Python (numpy)
import numpy as np
import time
import sys

# ---------------- Donnees de test (en tete du script) ----------------
# Test 1 : droite exacte y = 2x + 1 (pas de bruit)
x1 = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
y1 = 2.0 * x1 + 1.0
# Test 2 : y = 3x - 5 + bruit gaussien (sigma = 1), 100 points
np.random.seed(42)
x2 = np.linspace(0., 10., 100)
y2 = 3.0 * x2 - 5.0 + np.random.normal(0., 1., x2.size)
# Test 3 : droite horizontale y = 4 + bruit (pente attendue ~ 0)
x3 = np.linspace(-5., 5., 50)
y3 = 4.0 + np.random.normal(0., 0.5, x3.size)
# --------------------------------------------------------------------

def regression_matricielle(x, y):
    """Moindres carres sous forme matricielle : theta = (A^T A)^-1 A^T y"""
    A = np.column_stack((x, np.ones(x.size)))   # matrice [x 1]
    theta = np.linalg.solve(A.T @ A, A.T @ y)   # equations normales
    return theta[0], theta[1]

def regression_sommes(x, y):
    """Formules explicites avec les moyennes (comme dans le code C)"""
    xm, ym = x.mean(), y.mean()
    sxy = np.sum((x - xm) * (y - ym))
    sxx = np.sum((x - xm) ** 2)
    a = sxy / sxx
    b = ym - a * xm
    return a, b

def regression_boucle(x, y):
    """Version boucle Python pure (pour comparer les perfs)"""
    n = len(x)
    sx = sy = 0.0
    for i in range(n):
        sx += x[i]; sy += y[i]
    xm, ym = sx / n, sy / n
    sxy = sxx = 0.0
    for i in range(n):
        dx = x[i] - xm
        sxy += dx * (y[i] - ym)
        sxx += dx * dx
    a = sxy / sxx
    return a, ym - a * xm

def stats(x, y, a, b):
    """Coefficient de determination R^2 et ecart-type des residus"""
    r = y - (a * x + b)
    ss_res = np.sum(r ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    sigma = np.sqrt(ss_res / (x.size - 2))
    return r2, sigma

def lire_binaire(nom):
    """Fichier binaire : int64 n, puis n doubles x, puis n doubles y"""
    with open(nom, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        x = np.fromfile(f, dtype=np.float64, count=n)
        y = np.fromfile(f, dtype=np.float64, count=n)
    return x, y

if __name__ == "__main__":
    for nom, x, y in [("Test 1 (exact)", x1, y1),
                      ("Test 2 (bruit)", x2, y2),
                      ("Test 3 (pente nulle)", x3, y3)]:
        a, b = regression_matricielle(x, y)
        a2, b2 = regression_sommes(x, y)
        r2, sig = stats(x, y, a, b)
        print(f"{nom}: a = {a:.6f}  b = {b:.6f}  R2 = {r2:.6f}  sigma = {sig:.4f}")
        print(f"   (formules sommes : a = {a2:.6f}  b = {b2:.6f}  "
              f"ecart = {abs(a-a2):.1e})")
        ap, bp = np.polyfit(x, y, 1)
        print(f"   (np.polyfit     : a = {ap:.6f}  b = {bp:.6f})")

    # sauvegarde des donnees de test 2 pour le code C
    with open("test2.bin", "wb") as f:
        np.array([x2.size], dtype=np.int64).tofile(f)
        x2.tofile(f); y2.tofile(f)

    # gros jeu de donnees si fichier donne en argument
    if len(sys.argv) > 1:
        x, y = lire_binaire(sys.argv[1])
        print(f"\nGros jeu : n = {x.size}")
        t = time.perf_counter(); a, b = regression_sommes(x, y)
        print(f"numpy (sommes)    : a = {a:.6f} b = {b:.6f}  t = {time.perf_counter()-t:.3f} s")
        t = time.perf_counter(); a, b = regression_matricielle(x, y)
        print(f"numpy (matriciel) : a = {a:.6f} b = {b:.6f}  t = {time.perf_counter()-t:.3f} s")
        m = 2_000_000
        t = time.perf_counter(); a, b = regression_boucle(x[:m].tolist(), y[:m].tolist())
        print(f"boucle python ({m} pts): t = {time.perf_counter()-t:.3f} s")
