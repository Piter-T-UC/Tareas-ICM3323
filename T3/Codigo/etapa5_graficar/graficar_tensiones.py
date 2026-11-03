import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon

from etapa1_malla.elementos import ELEMENTOS, tipo_elemento
from etapa4_resolver.tensiones import von_mises
from etapa5_graficar.graficar_elemento import grilla_natural, evaluar


## Triangulacion fina de la malla para graficar: cada elemento se recorre con una grilla
## (n x n) en (xi, eta) y se lleva a (x, y) con x = sum N_i x_i; los valores nodales se
## interpolan igual, v = sum N_i v_i (respeta lados curvos y campos cuadraticos)
## valores: (n_nodos, k). Devuelve Triangulation y v (n_puntos, k)
def triangular(xy_m, conectividad, valores, n=9):
    pts, vals, tris = [], [], []
    base = 0
    for nodos in conectividad:
        XI, ETA = grilla_natural(tipo_elemento(nodos), n)
        N = evaluar(tipo_elemento(nodos), XI, ETA)[0].reshape(len(nodos), -1).T    # (n*n, n_nodos)
        pts.append(N @ xy_m[list(nodos)])
        vals.append(N @ valores[list(nodos)])
        i = base + np.arange(n*n).reshape(n, n)
        a, b, c, d = i[:-1, :-1].ravel(), i[:-1, 1:].ravel(), i[1:, 1:].ravel(), i[1:, :-1].ravel()
        tris.append(np.r_[np.c_[a, b, c], np.c_[a, c, d]])
        base += n*n
    pts, vals, tris = np.vstack(pts), np.vstack(vals), np.vstack(tris)
    p = pts[tris]                       # saca triangulos de area nula (punta de los triangulos)
    area = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return Triangulation(pts[:, 0], pts[:, 1], tris[np.abs(area) > 1e-12*np.abs(area).max()]), vals


## Mapas de color de sxx, syy, txy y von Mises [MPa] con los bordes de los elementos
## sigma_nodos: (n_nodos, 3) [Pa] de tensiones.calc_tensiones
def graficar_tensiones(xy, conectividad, sigma_nodos, titulo="Tensiones"):
    xy_m = np.asarray(getattr(xy, "magnitude", xy), dtype=float)
    unidad = f"{xy.units:~}" if hasattr(xy, "units") else "m"
    campos = np.c_[sigma_nodos, von_mises(sigma_nodos)] / 1e6
    tri, v = triangular(xy_m, conectividad, campos)
    nombres = [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\tau_{xy}$", r"$\sigma_{VM}$"]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10), layout="constrained")
    for k, (ax, nombre) in enumerate(zip(axs.flat, nombres)):
        lo, hi = v[:, k].min(), v[:, k].max()
        if k < 3 and lo < 0 < hi:       # con signo: 0 en blanco
            cf = ax.tricontourf(tri, v[:, k], 21, cmap="RdBu_r", norm=TwoSlopeNorm(0, lo, hi))
        else:
            cf = ax.tricontourf(tri, v[:, k], 21, cmap="RdBu_r" if k < 3 else "viridis")
        for nodos in conectividad:
            c = ELEMENTOS[tipo_elemento(nodos)]["contorno"]
            ax.add_patch(Polygon(xy_m[np.array(nodos)[c]], closed=True, fc="none", ec="k", lw=0.4))
        fig.colorbar(cf, ax=ax, label=f"{nombre} [MPa]")
        i_max = np.argmax(np.abs(campos[:, k]))
        ax.plot(*xy_m[i_max], "*", color="gold", mec="k", ms=12)
        ax.set_title(f"{nombre}:  min {campos[:, k].min():.4g},  max {campos[:, k].max():.4g} MPa"
                     f"   (★ nodo {i_max})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel(f"x [{unidad}]")
        ax.set_ylabel(f"y [{unidad}]")
    fig.suptitle(titulo)
    return fig
