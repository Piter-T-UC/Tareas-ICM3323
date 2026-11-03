import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D

from unidades import ureg
from etapa1_malla.elementos import ELEMENTOS, tipo_elemento
from etapa1_malla.cargas import leer_cargas_dist, puntos_lado, q_lineal


COLORES = {"T3": "#cfe2f3", "T6": "#d9ead3", "Q4": "#fff2cc", "Q8": "#f4cccc", "Q9": "#d9d2e9"}


## Simbolo de apoyo con la punta en p, abriendose hacia d (vector unitario hacia afuera)
## rodillo=True: agrega dos ruedas (restringe solo la direccion d)
def _dibujar_apoyo(ax, p, d, h, rodillo):
    n = np.array([-d[1], d[0]])
    base = p + d*h
    ax.add_patch(Polygon([p, base + 0.6*h*n, base - 0.6*h*n], closed=True,
                         fc="#4a3aa7", ec="#4a3aa7", alpha=0.35, zorder=4))
    if rodillo:
        r = 0.15*h
        for s in (-0.3, 0.3):
            ax.add_patch(Circle(base + s*h*n + d*r, r, fc="white", ec="#4a3aa7", zorder=4))
        base = base + 2*r*d
    ax.plot(*np.array([base + 0.9*h*n, base - 0.9*h*n]).T, color="#4a3aa7", lw=1.5, zorder=4)


## Malla con el numero de cada nodo, sin apoyos ni cargas (para ubicar nodos al definir el problema)
## zoom = (xmin, xmax, ymin, ymax) agrega un segundo panel con esa zona ampliada,
## para las partes donde la malla es tan fina que los numeros se pisan
def graficar_nodos(xy, conectividad, titulo="Numeracion de nodos", zoom=None):
    xy_m = np.asarray(getattr(xy, "magnitude", xy), dtype=float)
    unidad = f"{xy.units:~}" if hasattr(xy, "units") else "m"
    n_paneles = 1 if zoom is None else 2
    fig, axs = plt.subplots(1, n_paneles, figsize=(10*n_paneles, 10), layout="constrained", squeeze=False)

    for ax in axs[0]:
        for nodos in conectividad:
            info = ELEMENTOS[tipo_elemento(nodos)]
            pts = xy_m[np.array(nodos)[info["contorno"]]]
            ax.add_patch(Polygon(pts, closed=True, fc="#f3f0fa", ec="0.45", lw=0.8, zorder=1))
        ax.plot(xy_m[:, 0], xy_m[:, 1], "o", color="k", ms=2.5, zorder=5)
        for n, p in enumerate(xy_m):
            ax.annotate(str(n), p, xytext=(2, 2), textcoords="offset points", fontsize=7,
                        color="#1a5fb4", zorder=6)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.margins(0.04)
        ax.set_xlabel(f"x [{unidad}]")
        ax.set_ylabel(f"y [{unidad}]")

    axs[0, 0].set_title(titulo)
    if zoom is not None:
        axs[0, 1].set_xlim(zoom[0], zoom[1])
        axs[0, 1].set_ylim(zoom[2], zoom[3])
        axs[0, 1].set_title("Zoom")
    return fig, axs[0]


## Malla con numeracion de nodos y elementos, apoyos y fuerzas nodales
## conectividad = una fila de nodos por elemento (el tipo sale del numero de nodos)
## apoyos    = [[nodo, rx, ry], ...]       (1 = restringido)
## fuerzas   = [(nodo, Fx, Fy), ...]       (pint, ejes globales)
## cargas_dist: formato de cargas.leer_cargas_dist (se rotulan en MPa: q / espesor)
def graficar_malla(xy, conectividad, apoyos=None, fuerzas=None, cargas_dist=None, espesor=None,
                   titulo="Malla", numeracion=True):
    xy_m = np.asarray(getattr(xy, "magnitude", xy), dtype=float)
    centro = xy_m.mean(axis=0)
    tam = np.ptp(xy_m, axis=0).max()       # tamano de la malla (para escalar simbolos)
    dx, dy = np.ptp(xy_m, axis=0)
    fig, ax = plt.subplots(figsize=(10, np.clip(10*dy/max(dx, 1e-12) + 1.5, 3, 9)), layout="constrained")

    # elementos: poligono por el contorno (incluye nodos medios) + numero y tipo
    for e, nodos in enumerate(conectividad):
        tipo = tipo_elemento(nodos)
        info = ELEMENTOS[tipo]
        pts = xy_m[np.array(nodos)[info["contorno"]]]
        ax.add_patch(Polygon(pts, closed=True, fc=COLORES[tipo], ec="k", lw=1.0, zorder=1))
        if numeracion:
            c = xy_m[list(nodos)[:info["esquinas"]]].mean(axis=0)
            ax.annotate(f"{e} ({tipo})", c, xytext=(0, -5), textcoords="offset points", ha="center",
                        va="top", fontsize=7, color="0.3", zorder=2)

    # nodos
    ax.plot(xy_m[:, 0], xy_m[:, 1], "o", color="k", ms=3, zorder=5)
    if numeracion:
        for n, p in enumerate(xy_m):
            ax.annotate(str(n), p, xytext=(3, 3), textcoords="offset points", fontsize=7,
                        color="#2a78d6", zorder=6)

    # apoyos: el simbolo se dibuja del lado de afuera de la malla
    h = 0.05*tam
    for nodo, rx, ry in (apoyos if apoyos is not None else []):
        p = xy_m[nodo]
        lado_x = np.array([1.0 if p[0] > centro[0] else -1.0, 0.0])
        lado_y = np.array([0.0, 1.0 if p[1] > centro[1] else -1.0])
        if rx and ry:
            _dibujar_apoyo(ax, p, lado_y, h, rodillo=False)
        elif ry:
            _dibujar_apoyo(ax, p, lado_y, h, rodillo=True)
        elif rx:
            _dibujar_apoyo(ax, p, lado_x, h, rodillo=True)

    # fuerzas: flecha que llega al nodo (empuja) o que sale de el si apunta hacia afuera de la
    # malla (tira), largo entre 60% y 100% de h_F segun |F|
    if fuerzas:
        F = np.array([[f.to(ureg.N).magnitude for f in c] for _, *c in fuerzas])
        F_max = max(np.linalg.norm(F, axis=1).max(), 1e-12)
        h_F = 0.15*tam
        for (nodo, *_), Fv in zip(fuerzas, F):
            mag = np.linalg.norm(Fv)
            if mag < 1e-9*F_max:
                continue
            punta = xy_m[nodo]
            largo = Fv/mag*h_F*(0.6 + 0.4*mag/F_max)
            tira = (punta - centro) @ Fv > 0
            cola = punta - largo
            texto, signo = cola, 1
            if tira:
                cola, punta = punta, punta + largo
                texto, signo = punta, -1
            ax.update_datalim([cola, punta])
            ax.annotate("", punta, cola, zorder=7,
                        arrowprops=dict(arrowstyle="-|>", color="#eda100", lw=2, mutation_scale=15))
            ax.text(*texto, f"{mag/1e3:.3g} kN", color="#b37800", fontsize=8, zorder=7,
                    ha="right" if signo*Fv[0] > 0 else "left", va="top" if signo*Fv[1] > 0 else "bottom")

    # cargas distribuidas: flechas que llegan al lado y envolvente lineal (largo maximo h_q)
    if cargas_dist:
        cargas = leer_cargas_dist(conectividad, cargas_dist, espesor)
        s = np.linspace(0, 1, 41)
        q_max = max(max(np.linalg.norm(q_lineal(q1, q2, s), axis=1).max()
                        for _, nl, q1, q2 in cargas), 1e-12)
        h_q = 0.10*tam
        for _, nl, q1, q2 in cargas:
            L = np.linalg.norm(np.diff(puntos_lado(xy_m, nl, s)[0], axis=0), axis=1).sum()
            s_fl = np.linspace(0, 1, max(5, int(round(12*L/tam)) + 1))
            for sv, x_s, q_s in zip(s_fl, puntos_lado(xy_m, nl, s_fl)[0], q_lineal(q1, q2, s_fl)):
                if np.linalg.norm(q_s) > 1e-3*q_max:
                    ax.annotate("", x_s, x_s - q_s/q_max*h_q, zorder=7,
                                arrowprops=dict(arrowstyle="-|>", color="#eda100", lw=1, mutation_scale=9))
            colas = puntos_lado(xy_m, nl, s)[0] - q_lineal(q1, q2, s)/q_max*h_q
            ax.plot(*colas.T, color="#eda100", lw=1.2, zorder=7)
            t_m = espesor.to(ureg.meter).magnitude
            a, b = np.linalg.norm(q1)/t_m/1e6, np.linalg.norm(q2)/t_m/1e6
            texto = f"{a:.3g} MPa" if np.isclose(a, b) else f"{a:.3g}–{b:.3g} MPa"
            medio = colas[len(colas)//2]
            afuera = medio - puntos_lado(xy_m, nl, 0.5)[0][0]
            ax.annotate(texto, medio, xytext=3*afuera/max(np.linalg.norm(afuera), 1e-12),
                        textcoords="offset points", color="#b37800", fontsize=8, zorder=7,
                        ha="left" if afuera[0] > 1e-9 else "right" if afuera[0] < -1e-9 else "center",
                        va="bottom" if afuera[1] > 1e-9 else "top" if afuera[1] < -1e-9 else "center")

    tipos = sorted({tipo_elemento(nodos) for nodos in conectividad})
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=10, mfc=COLORES[t], mec="k", label=t)
                       for t in tipos], loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.12)
    unidad = f"{xy.units:~}" if hasattr(xy, "units") else "m"
    ax.set_xlabel(f"x [{unidad}]")
    ax.set_ylabel(f"y [{unidad}]")
    ax.set_title(titulo)
    return fig, ax
