import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from esfuerzos import esfuerzos_a_lo_largo
from exportar_latex import _DIAG
from igualar_ejes_3d import _igualar_ejes_3d


# ==================================================================
# DIAGRAMAS DE FUERZAS INTERNAS Y TENSIONES (matplotlib)
# ==================================================================
# "res" es la salida de calc_esfuerzos() y "q_locales" la tercera salida de
# gen_f_distribuida() (None si no hay cargas distribuidas).

# componente: (titulo, unidad, divisor desde SI)
_FUERZAS = {"N": ("N", "kN", 1e3), "V_x": ("V_x", "kN", 1e3), "V_y": ("V_y", "kN", 1e3),
            "T": ("T", r"kN$\cdot$m", 1e3), "M_x": ("M_x", r"kN$\cdot$m", 1e3),
            "M_y": ("M_y", r"kN$\cdot$m", 1e3)}
_TENSIONES = {"sigma_N": (r"\sigma_N", "MPa", 1e6), "sigma_max": (r"\sigma_{max}", "MPa", 1e6),
              "tau_T": (r"\tau_T", "MPa", 1e6), "sigma_vm": (r"\sigma_{VM}", "MPa", 1e6)}

_COLOR = "#d65f28"
# secuencial de un solo tono, sin el extremo mas claro (se perderia sobre fondo blanco)
_CMAP = ListedColormap(plt.get_cmap("Oranges")(np.linspace(0.3, 1.0, 256)))


def _q(q_locales, e):
    return None if q_locales is None else q_locales[e]


def _a_lo_largo(res, e, q_locales):
    # limpia el ruido de redondeo (~1e-16) para que no aparezca en los ejes
    d = esfuerzos_a_lo_largo(res, e, _q(q_locales, e))
    for clave in list(_FUERZAS) + list(_TENSIONES):
        d[clave][np.abs(d[clave]) < 1e-9] = 0.0
    return d


def _xyz_m(xyz):
    return xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, float)


# ------------------------------------------------------------------
# 2D: un elemento, esfuerzos vs posicion z local
# ------------------------------------------------------------------
def _graficar_elemento(res, e, q_locales, conectividad, tabla, forma, titulo):
    d = _a_lo_largo(res, e, q_locales)
    z = d["z"]
    fig, axes = plt.subplots(*forma, figsize=(10, 2.6*forma[0]), sharex=True)
    for ax, (clave, (simb, unidad, div)) in zip(axes.flat, tabla.items()):
        v = d[clave] / div
        ax.fill_between(z, v, color=_COLOR, alpha=0.2, linewidth=0)
        ax.plot(z, v, color=_COLOR, linewidth=2)
        ax.axhline(0, color="0.4", linewidth=0.8)
        i = int(np.argmax(np.abs(v)))
        if abs(v[i]) > 0:
            ax.plot(z[i], v[i], "o", color=_COLOR, markersize=6)
            ax.annotate(f"{v[i]:.3g}", (z[i], v[i]), textcoords="offset points",
                        xytext=(0, 6 if v[i] >= 0 else -12), ha="center", fontsize=8)
        ax.set_title(f"${simb}$ [{unidad}]", fontsize=10)
        ax.grid(alpha=0.3)
        ax.margins(y=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("z local [m]")
    n1, n2 = conectividad[e]
    fig.suptitle(f"{titulo} - elemento {e} (nodos {n1}-{n2}, L = {res['largos'][e]:.2f} m)")
    fig.tight_layout()
    return fig


def graficar_fuerzas_elemento(res, e, q_locales, conectividad):
    return _graficar_elemento(res, e, q_locales, conectividad, _FUERZAS, (3, 2), "Fuerzas internas")


def graficar_tensiones_elemento(res, e, q_locales, conectividad):
    return _graficar_elemento(res, e, q_locales, conectividad, _TENSIONES, (2, 2), "Tensiones")


# ------------------------------------------------------------------
# 3D: diagrama sobre la estructura
# ------------------------------------------------------------------
def _dibujar_barras(ax, xyz_m, conectividad, **kw):
    for n1, n2 in conectividad:
        ax.plot(*np.array([xyz_m[n1], xyz_m[n2]]).T, **kw)


def graficar_fuerzas_3d(xyz, conectividad, res, q_locales=None,
                        componentes=("N", "V_x", "V_y", "T", "M_x", "M_y"),
                        titulo="Fuerzas internas"):
    """Cada diagrama se dibuja perpendicular a la barra, en la misma direccion local
    que exportar_latex: M_y, V_x, N y T hacia +x_L; M_x y V_y hacia +y_L."""
    xyz_m = _xyz_m(xyz)
    tam = np.ptp(xyz_m, axis=0).max()
    datos = [_a_lo_largo(res, e, q_locales) for e in range(len(conectividad))]

    n_col = min(3, len(componentes))
    n_fil = int(np.ceil(len(componentes) / n_col))
    fig = plt.figure(figsize=(5.2*n_col, 4.8*n_fil))
    for k, comp in enumerate(componentes):
        ax = fig.add_subplot(n_fil, n_col, k + 1, projection="3d")
        eje = _DIAG[comp][0]
        simb, unidad, div = _FUERZAS[comp]
        v_max = max(np.abs(d[comp]).max() for d in datos)
        s = 0.15*tam/v_max if v_max > 1e-9 else 0.0

        _dibujar_barras(ax, xyz_m, conectividad, color="black", linewidth=1.5)
        puntos = [xyz_m]
        for e, (n1, n2) in enumerate(conectividad):
            d = datos[e]
            v = d[comp]
            base = xyz_m[n1] + np.outer(d["z"]/res["largos"][e], xyz_m[n2] - xyz_m[n1])
            curva = base + np.outer(v*s, res["rot"][e][eje])
            ax.add_collection3d(Poly3DCollection([np.vstack([base, curva[::-1]])],
                                                 facecolor=_COLOR, alpha=0.25, edgecolor="none"))
            ax.plot(*curva.T, color=_COLOR, linewidth=1.5)
            puntos.append(curva)
            # rotulo solo en el maximo |valor| de cada elemento (los extremos saturan la vista)
            i = int(np.argmax(np.abs(v)))
            if abs(v[i]) > 0:
                ax.text(*curva[i], f"{v[i]/div:.3g}", fontsize=6)
        ax.set_title(f"${simb}$ [{unidad}]", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        _igualar_ejes_3d(ax, np.vstack(puntos))
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig


def graficar_tensiones_3d(xyz, conectividad, res, q_locales=None,
                          componentes=("sigma_N", "sigma_max", "tau_T", "sigma_vm"),
                          titulo="Tensiones"):
    """Cada barra coloreada a lo largo segun la tension (|valor| en MPa)."""
    xyz_m = _xyz_m(xyz)
    datos = [_a_lo_largo(res, e, q_locales) for e in range(len(conectividad))]

    n_col = min(2, len(componentes))
    n_fil = int(np.ceil(len(componentes) / n_col))
    fig = plt.figure(figsize=(6*n_col, 5*n_fil))
    for k, comp in enumerate(componentes):
        ax = fig.add_subplot(n_fil, n_col, k + 1, projection="3d")
        simb, unidad, div = _TENSIONES[comp]
        segs, vals = [], []
        for e, (n1, n2) in enumerate(conectividad):
            d = datos[e]
            pts = xyz_m[n1] + np.outer(d["z"]/res["largos"][e], xyz_m[n2] - xyz_m[n1])
            v = np.abs(d[comp]) / div
            segs += [pts[i:i+2] for i in range(len(pts) - 1)]
            vals += list((v[:-1] + v[1:]) / 2)
        vals = np.array(vals)
        norma = colors.Normalize(0, vals.max() if vals.max() > 0 else 1.0)
        ax.add_collection3d(Line3DCollection(segs, cmap=_CMAP, norm=norma, array=vals, linewidth=4))
        for i, q in enumerate(xyz_m):
            ax.text(*q, str(i), fontsize=7)
        fig.colorbar(cm.ScalarMappable(norm=norma, cmap=_CMAP), ax=ax, shrink=0.6, pad=0.1,
                     label=f"$|{simb}|$ [{unidad}]")
        ax.set_title(f"$|{simb}|$ (máx {vals.max():.2f} {unidad})", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        _igualar_ejes_3d(ax, xyz_m)
    fig.suptitle(titulo)
    fig.tight_layout()
    return fig
