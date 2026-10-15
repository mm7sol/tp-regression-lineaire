import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from reg_lin import x2, y2, x3, y3, regression_matricielle
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for k, (x, y, t, vrai) in enumerate([(x2, y2, "Test 2 : y = 3x - 5 + bruit", (3, -5)),
                                     (x3, y3, "Test 3 : y = 4 + bruit", (0, 4))]):
    a, b = regression_matricielle(x, y)
    ax[k].plot(x, y, ".", ms=4, label="donnees")
    ax[k].plot(x, a*x+b, "r-", label=f"regression a={a:.3f} b={b:.3f}")
    ax[k].plot(x, vrai[0]*x+vrai[1], "k--", lw=1, label="droite theorique")
    ax[k].set_title(t); ax[k].set_xlabel("x"); ax[k].set_ylabel("y"); ax[k].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_tests.png", dpi=150)
fig, ax = plt.subplots(figsize=(7, 3.5))
lab = ["Python\nboucle*", "numpy", "C -O0", "C -O3", "C -Ofast", "OpenMP\n1 thread", "OpenMP\n2 threads"]
val = [3.89, 0.415, 0.1925, 0.0628, 0.0503, 0.0652, 0.0311]
ax.bar(lab, val, color=["#999","#4a7","#47a","#47a","#47a","#a54","#a54"])
ax.set_yscale("log"); ax.set_ylabel("temps (s, echelle log)")
ax.set_title("Temps d'une regression, n = 20 000 000 points")
for i, v in enumerate(val): ax.text(i, v*1.1, f"{v:.3g}", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig("fig_perf.png", dpi=150)
