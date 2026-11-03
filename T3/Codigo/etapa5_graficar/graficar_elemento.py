import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from etapa1_malla.elementos import ELEMENTOS
from etapa2_interpolacion.funciones_forma import funciones_forma, NODOS_NATURALES
from etapa3_matrices.matriz_B import jacobiano, matriz_B


## Grilla (n x n) sobre el elemento de referencia. En triangulos: xi = u (1 - v), eta = v
## (cubre el triangulo exacto; las columnas son rayos hacia el nodo 2)
def grilla_natural(tipo, n=61):
    s = np.linspace(0, 1, n) if tipo[0] == "T" else np.linspace(-1, 1, n)
    U, V = np.meshgrid(s, s)
    if tipo[0] == "T":
        return U*(1 - V), V
    return U, V


## N (n_nodos, ny, nx) y dN (2, n_nodos, ny, nx) en todos los puntos de la grilla
def evaluar(tipo, XI, ETA):
    res = [funciones_forma(tipo, a, b) for a, b in zip(XI.ravel(), ETA.ravel())]
    N = np.array([r[0] for r in res]).T.reshape(-1, *XI.shape)
    dN = np.moveaxis(np.array([r[1] for r in res]), 0, -1).reshape(2, -1, *XI.shape)
    return N, dN


## Contorno del elemento de referencia como puntos (xi, eta)
def borde_natural(tipo, n=30):
    c = NODOS_NATURALES[tipo][:ELEMENTOS[tipo]["esquinas"]]
    c = np.vstack([c, c[:1]])
    t = np.linspace(0, 1, n)[:, None]
    return np.vstack([(1 - t)*p + t*q for p, q in zip(c[:-1], c[1:])])


## Puntos naturales (m, 2) -> fisicos con x = sum N_i x_i
def a_fisico(tipo, xy_e, pts):
    return np.array([funciones_forma(tipo, *p)[0] @ xy_e for p in pts])


def _numerar(ax, pts):
    for k, p in enumerate(pts):
        ax.annotate(str(k), p, xytext=(4, 4), textcoords="offset points", fontsize=8, color="#2a78d6")


## Un mapa de color por funcion de forma sobre el elemento de referencia
def graficar_funciones_forma(tipo):
    XI, ETA = grilla_natural(tipo)
    N, _ = evaluar(tipo, XI, ETA)
    nat = NODOS_NATURALES[tipo]
    borde = borde_natural(tipo)
    n = len(nat)
    cols = {3: 3, 4: 2, 6: 3, 8: 4, 9: 3}[n]
    filas = -(-n // cols)
    fig, axs = plt.subplots(filas, cols, figsize=(3.3*cols + 1, 3.2*filas), layout="constrained", squeeze=False)
    norma = TwoSlopeNorm(vmin=-0.4, vcenter=0, vmax=1)
    for i, ax in enumerate(axs.flat):
        if i >= n:
            ax.axis("off")
            continue
        cf = ax.contourf(XI, ETA, N[i], np.linspace(-0.4, 1, 15), cmap="RdBu_r", norm=norma, extend="both")
        ax.contour(XI, ETA, N[i], [0], colors="k", linewidths=0.8, linestyles="--")
        ax.plot(*borde.T, "k", lw=1)
        ax.plot(*nat.T, "o", color="k", ms=4)
        ax.plot(*nat[i], "o", color="gold", mec="k", ms=11)
        _numerar(ax, nat)
        ax.set_title(f"$N_{{{i}}}$  (= 1 en el nodo {i})", fontsize=10)
        ax.set_aspect("equal")
        ax.margins(0.08)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$\eta$")
    fig.colorbar(cf, ax=axs, shrink=0.8, label="$N_i$")
    fig.suptitle(f"Funciones de forma {tipo}: cada $N_i$ vale 1 en su nodo (amarillo) y 0 en los demas"
                 f"   (- - - : N = 0)")
    return fig


## Elemento de referencia -> elemento fisico: lineas de la grilla, det J de fondo y en el punto
## las filas de J (tangentes a las lineas eta = cte y xi = cte)
def graficar_mapeo(tipo, xy_e, punto):
    XI, ETA = grilla_natural(tipo)
    N, dN = evaluar(tipo, XI, ETA)
    X, Y = np.einsum("nij,n->ij", N, xy_e[:, 0]), np.einsum("nij,n->ij", N, xy_e[:, 1])
    detJ = np.linalg.det(np.einsum("anij,nb->ijab", dN, xy_e))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.8), layout="constrained")
    cf = ax1.contourf(X, Y, detJ, 20, cmap="viridis")
    fig.colorbar(cf, ax=ax1, label="det J")
    col = r"$\xi$ = cte" if tipo[0] == "Q" else r"$\xi/(1-\eta)$ = cte"
    for ax, (A, B) in ((ax0, (XI, ETA)), (ax1, (X, Y))):
        for k, j in enumerate(np.linspace(0, len(XI) - 1, 9).astype(int)):
            ax.plot(A[j, :], B[j, :], color="C0", lw=0.9, label=r"$\eta$ = cte" if k == 0 else None)
            ax.plot(A[:, j], B[:, j], color="C3", lw=0.9, label=col if k == 0 else None)

    nat = NODOS_NATURALES[tipo]
    borde = borde_natural(tipo)
    ax0.plot(*borde.T, "k", lw=1.2)
    ax1.plot(*a_fisico(tipo, xy_e, borde).T, "k", lw=1.2)
    for ax, pts in ((ax0, nat), (ax1, xy_e)):
        ax.plot(*pts.T, "o", color="k", ms=5, zorder=5)
        _numerar(ax, pts)

    N0, dN0 = funciones_forma(tipo, *punto)
    J0 = dN0 @ xy_e
    p = N0 @ xy_e
    ax0.plot(*punto, "*", color="gold", mec="k", ms=15, zorder=6)
    ax1.plot(*p, "*", color="gold", mec="k", ms=15, zorder=6)
    esc = 0.3
    for fila, color, txt in ((0, "C0", r"$\partial\mathbf{x}/\partial\xi$ (fila 0 de J)"),
                             (1, "C3", r"$\partial\mathbf{x}/\partial\eta$ (fila 1 de J)")):
        ax1.annotate("", p + esc*J0[fila], p, zorder=7,
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=18))
        ax1.annotate(txt, p + esc*J0[fila], xytext=(5, 5), textcoords="offset points", color=color,
                     fontsize=10, fontweight="bold", zorder=7, bbox=dict(fc="w", ec="none", alpha=0.8))

    ax0.set_title(r"Elemento de referencia ($\xi$, $\eta$)")
    ax1.set_title(f"Elemento fisico (x, y), fondo = det J   (min {detJ.min():.3g}"
                  f"{'  ¡NEGATIVO!' if detJ.min() <= 0 else ''})")
    for ax in (ax0, ax1):
        ax.set_aspect("equal")
        ax.use_sticky_edges = False         # contourf pega los ejes al borde y corta los nodos
        ax.margins(0.08)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8)
    fig.suptitle(f"{tipo}: mapeo isoparametrico  x = Σ N_i x_i   (estrella = punto {tuple(punto)})")
    return fig


## Para cada caso (nombre, u(x, y) -> (ux, uy), eps_exacta): impone u en los nodos, dibuja la
## deformada y compara eps = B u_e con la exacta en el punto y en una grilla de todo el elemento
def graficar_deformacion(tipo, xy_e, casos, punto, escala=1.0):
    XI, ETA = grilla_natural(tipo, 15)
    puntos = np.c_[XI.ravel(), ETA.ravel()]
    borde = borde_natural(tipo)
    fig, axs = plt.subplots(1, len(casos), figsize=(4.3*len(casos), 5.2), layout="constrained", squeeze=False)
    for ax, (nombre, u, eps) in zip(axs[0], casos):
        u_e = np.column_stack(u(xy_e[:, 0], xy_e[:, 1]))
        eps_B = matriz_B(tipo, xy_e, *punto)[0] @ u_e.ravel()
        err = max(abs(matriz_B(tipo, xy_e, *q)[0] @ u_e.ravel() - eps).max() for q in puntos)

        ax.fill(*a_fisico(tipo, xy_e, borde).T, fc="0.88", ec="0.5", label="original")
        ax.plot(*a_fisico(tipo, xy_e + escala*u_e, borde).T, color="C3", lw=1.6, label=f"deformada (x{escala:g})")
        ax.plot(*xy_e.T, "o", color="0.4", ms=3)
        ax.quiver(*xy_e.T, *(escala*u_e).T, angles="xy", scale_units="xy", scale=1, color="C3", width=0.005)
        ax.set_title(nombre, fontsize=11)
        ax.set_aspect("equal")
        ax.margins(0.15)
        ax.set_xlabel(f"ε exacta     = {np.round(eps, 4) + 0.0}\n"
                      f"B·u (punto)  = {np.round(eps_B, 4) + 0.0}\n"
                      f"max |B·u − ε| en el elemento = {err:.1e}", family="monospace", fontsize=8)
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{tipo}: u conocido en los nodos  ->  ε = B u_e   (ε = [εxx, εyy, γxy])")
    return fig


def _matriz(ax, M, titulo, filas, columnas):
    lim = max(abs(M).max(), 1e-12)
    ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v + 0.0:.3g}", ha="center", va="center", fontsize=7 if M.shape[1] > 9 else 9,
                color="w" if abs(v) > 0.6*lim else "k")
    ax.set_xticks(range(M.shape[1]), columnas, fontsize=8)
    ax.set_yticks(range(M.shape[0]), filas)
    ax.set_title(titulo, fontsize=10)


## J, J^-1, D, dN, dN_dx y B en el punto como tablas de color con sus valores
def graficar_matrices(tipo, xy_e, punto, D):
    N, dN = funciones_forma(tipo, *punto)
    J, detJ, dN_dx = jacobiano(dN, xy_e)
    B = matriz_B(tipo, xy_e, *punto)[0]
    n = len(N)
    nodos = [f"N{i}" for i in range(n)]
    fig = plt.figure(figsize=(max(11, 1.2*n + 2), 11), layout="constrained")
    gs = fig.add_gridspec(4, 3)
    _matriz(fig.add_subplot(gs[0, 0]), J, f"J   (det J = {detJ:.4g})", [r"$\xi$", r"$\eta$"], ["x", "y"])
    _matriz(fig.add_subplot(gs[0, 1]), np.linalg.inv(J), r"$J^{-1}$", ["x", "y"], [r"$\xi$", r"$\eta$"])
    _matriz(fig.add_subplot(gs[0, 2]), D/1e9, "D [GPa]", [r"$\sigma_{xx}$", r"$\sigma_{yy}$", r"$\tau_{xy}$"],
            [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\gamma_{xy}$"])
    _matriz(fig.add_subplot(gs[1, :]), dN, "dN  (derivadas naturales, funciones_forma)",
            [r"$\partial N/\partial\xi$", r"$\partial N/\partial\eta$"], nodos)
    _matriz(fig.add_subplot(gs[2, :]), dN_dx, r"dN_dx = $J^{-1}$ dN  (derivadas fisicas, jacobiano)",
            [r"$\partial N/\partial x$", r"$\partial N/\partial y$"], nodos)
    _matriz(fig.add_subplot(gs[3, :]), B, "B  (matriz_B)",
            [r"$\varepsilon_{xx}$", r"$\varepsilon_{yy}$", r"$\gamma_{xy}$"],
            [f"u{i}{c}" for i in range(n) for c in "xy"])
    fig.suptitle(f"{tipo}: matrices en (ξ, η) = {tuple(punto)}")
    return fig
