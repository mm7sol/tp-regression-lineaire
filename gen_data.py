# Generation d'un gros jeu de donnees binaire : y = 1.5 x + 2 + bruit
import numpy as np, sys
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20_000_000
rng = np.random.default_rng(0)
x = rng.uniform(0., 100., n)
y = 1.5 * x + 2.0 + rng.normal(0., 5., n)
with open("gros.bin", "wb") as f:
    np.array([n], dtype=np.int64).tofile(f); x.tofile(f); y.tofile(f)
print("ecrit gros.bin, n =", n)
