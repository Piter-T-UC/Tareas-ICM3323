import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D

from placa_agujero import resolver_placa
from etapa1_malla.elementos import ELEMENTOS, tipo_elemento
from etapa5_graficar.graficar_tensiones import triangular


## Figuras del informe T3: MEF (malla 1 = problema_2, malla 2 = problema_3) vs Kirsch, placa L/a = 20
##   comparacion_borde_agujero.png : sigma_theta(theta) en r = a
##   comparacion_eje_y.png         : sxx por el eje x = 0, de B (0, a) a D (0, L)
##   deformada_mef.png             : deformada amplificada y mapas de ux, uy (malla 2)
##   convergencia_Kt.png           : Kt = sxx(B)/S vs numero de GDL
## Se guardan en T3/figuras (la ruta que usa el .tex)

aqui = os.path.dirname(os.path.abspath(__file__))
carpeta = os.path.join(aqui, "..", "figuras")
os.makedirs(carpeta, exist_ok=True)

# Datos, iguales a problema_2.py y problema_3.py
# (las tensiones no dependen de E ni del espesor: solo hay apoyos de simetria)
E_GPa = 200
nu = 0.3
estado = "tension_plana"
a_m, L_m = 0.010, 0.200         # L/a = 20
S_Pa = 100e6
from unidades import ureg
datos = (S_Pa, E_GPa*ureg.GPa, nu, 10*ureg.mm, estado)

# Mallas del informe: (n_rad, n_ang) por bloque, misma razon radial
razon_rad = 15
mallas = {"Malla 1 (problema 2)": (10, 4), "Malla 2 (problema 3)": (20, 8)}
# Convergencia: la misma familia refinada (la malla 1 y la malla 2 son el 2do y 3er nivel)
niveles = [(5, 2), (10, 4), (20, 8), (40, 16), (80, 32)]
# Referencia: placa chica L = 5a (razon 3), para ver cuanto pesa el tamano finito de la placa
L_chica, razon_chica = 0.050, 3
niveles_chica = [(3, 2), (6, 4), (12, 8), (24, 16), (48, 32)]

# Estilo: Kirsch en tinta oscura (referencia), cada malla con su color y marcador
KIRSCH, C1, C2, C_TRAC, C_CHICA = "#2b2b2b", "#2a78d6", "#eb6834", "#e87ba4", "0.6"
estilo = {"Malla 1 (problema 2)": dict(color=C1, marker="o"),
          "Malla 2 (problema 3)": dict(color=C2, marker="s")}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "0.9", "grid.linewidth": 0.6})


## Kirsch, placa infinita
def kirsch_sigma_theta_borde(th):
    return S_Pa*(1 - 2*np.cos(2*th))


def kirsch_sxx_eje_y(y):
    q = a_m/y
    return S_Pa*(1 + q**2/2 + 3*q**4/2)


## sigma_theta en los nodos de r = a, ordenados por theta
def sigma_theta_borde(sol):
    x, y = sol.xy_m.T
    i = np.where(np.isclose(np.hypot(x, y), a_m, rtol=1e-6))[0]
    th = np.arctan2(y[i], x[i])
    i, th = i[np.argsort(th)], np.sort(th)
    sxx, syy, txy = sol.sigma_nodos[i].T
    s, c = np.sin(th), np.cos(th)
    return th, sxx*s**2 + syy*c**2 - 2*txy*s*c


## sxx en los nodos de x = 0, ordenados por y
def sxx_eje_y(sol):
    x, y = sol.xy_m.T
    i = np.where(np.isclose(x, 0))[0]
    i = i[np.argsort(y[i])]
    return y[i], sol.sigma_nodos[i, 0]


sols = {nombre: resolver_placa(a_m, L_m, *datos, *m, razon_rad, "uniforme") for nombre, m in mallas.items()}
for nombre, s in sols.items():
    print(f"{nombre}: {s.n_elem} elementos, {len(s.xy_m)} nodos, {s.n_gdl} GDL, "
          f"Kt = {s.sigma_nodos[s.B, 0]/S_Pa:.4f}")


# --- 1. sigma_theta en el borde del agujero ----------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
th = np.linspace(0, np.pi/2, 300)
ax.plot(np.degrees(th), kirsch_sigma_theta_borde(th)/S_Pa, color=KIRSCH, lw=2, label="Kirsch (placa infinita)")
for nombre, s in sols.items():
    t, st = sigma_theta_borde(s)
    ax.plot(np.degrees(t), st/S_Pa, ls="--", lw=1, ms=6, mec="white", mew=0.8, **estilo[nombre],
            label=f"MEF, {nombre.lower()}")
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlim(0, 90)
ax.set_xticks(np.arange(0, 91, 15))
ax.set_xlabel(r"$\theta$ [grados]")
ax.set_ylabel(r"$\sigma_\theta \,/\, S$")
ax.set_title(r"$\sigma_\theta$ en el borde del agujero ($r = a$), $L = 20a$")
ax.legend(frameon=False)
fig.savefig(os.path.join(carpeta, "comparacion_borde_agujero.png"), dpi=200)


# --- 2. sxx por el eje y ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
y = np.geomspace(a_m, L_m, 400)
ax.plot(y/a_m, kirsch_sxx_eje_y(y)/S_Pa, color=KIRSCH, lw=2, label="Kirsch (placa infinita)")
for nombre, s in sols.items():
    yy, sx = sxx_eje_y(s)
    ax.plot(yy/a_m, sx/S_Pa, ls="--", lw=1, ms=5, mec="white", mew=0.8, **estilo[nombre],
            label=f"MEF, {nombre.lower()}")
ax.axhline(1, color="0.5", lw=0.8)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 20])
ax.set_xticklabels(["1", "2", "3", "5", "10", "20"])
ax.set_xlim(1, L_m/a_m)
ax.set_xlabel(r"$y / a$  (sobre $x = 0$, escala log)")
ax.set_ylabel(r"$\sigma_{xx} \,/\, S$")
ax.set_title(r"$\sigma_{xx}$ a lo largo del eje $y$, de B $(0, a)$ a D $(0, L)$")
ax.legend(frameon=False)
fig.savefig(os.path.join(carpeta, "comparacion_eje_y.png"), dpi=200)


# --- Deformada amplificada y mapas de ux, uy (malla 2) -------------------------------------------
s2 = sols["Malla 2 (problema 3)"]
xy_mm, U_um = s2.xy_m*1e3, s2.U_nodos_m*1e6
tri, v = triangular(xy_mm, s2.conectividad, U_um)
escala = 0.10*np.ptp(xy_mm, axis=0).max()/(np.linalg.norm(U_um, axis=1).max()*1e-3)

fig, axs = plt.subplot_mosaic([["def", "ux"], ["def", "uy"]], figsize=(10, 6.2), layout="constrained",
                              width_ratios=[1.35, 1])
ax = axs["def"]
for nodos in s2.conectividad:
    c = np.array(nodos)[ELEMENTOS[tipo_elemento(nodos)]["contorno"]]
    ax.add_patch(Polygon(xy_mm[c], closed=True, fc="none", ec="0.75", lw=0.5))
    ax.add_patch(Polygon((xy_mm + escala*U_um*1e-3)[c], closed=True, fc="none", ec=C1, lw=0.9))
ax.legend(handles=[Line2D([], [], color="0.75", lw=1, label="Original"),
                   Line2D([], [], color=C1, lw=2, label=f"Deformada (x{escala:.0f})")],
          loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=2, frameon=False)
ax.set_title(f"Deformada amplificada, malla 2 ({s2.n_elem} elementos Q9)")
for k, (clave, nombre, cmap) in enumerate([("ux", "u_x", "Blues"), ("uy", "u_y", "Blues_r")]):
    ax = axs[clave]
    cf = ax.tricontourf(tri, v[:, k], 20, cmap=cmap)
    for nodos in s2.conectividad:
        c = np.array(nodos)[ELEMENTOS[tipo_elemento(nodos)]["contorno"]]
        ax.add_patch(Polygon(xy_mm[c], closed=True, fc="none", ec="k", lw=0.25, alpha=0.5))
    fig.colorbar(cf, ax=ax, label=rf"${nombre}$ [$\mu$m]")
    i = np.argmax(np.abs(U_um[:, k]))
    ax.set_title(rf"${nombre}$: max $|{nombre}|$ = {abs(U_um[i, k]):.2f} $\mu$m en ({xy_mm[i, 0]:.0f}, {xy_mm[i, 1]:.0f}) mm",
                 fontsize=9)
for ax in axs.values():
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.02)
    ax.grid(False)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
fig.savefig(os.path.join(carpeta, "deformada_mef.png"), dpi=200)


# --- 3. Convergencia de Kt ---------------------------------------------------------------------
def barrido(L_i, niveles_i, razon_i, carga):
    gdl, kt = [], []
    for n_rad, n_ang in niveles_i:
        t0 = time.time()
        s = resolver_placa(a_m, L_i, *datos, n_rad, n_ang, razon_i, carga)
        gdl.append(s.n_gdl)
        kt.append(s.sigma_nodos[s.B, 0]/S_Pa)
        print(f"  L/a = {L_i/a_m:>4.0f}  {carga:<9}{n_rad:>4}x{n_ang:<3}{gdl[-1]:>8} GDL"
              f"   Kt = {kt[-1]:.4f}   ({time.time() - t0:.1f} s)")
    return gdl, kt


print("\nConvergencia de Kt:")
gdl, kt_u = barrido(L_m, niveles, razon_rad, "uniforme")
_, kt_e = barrido(L_m, niveles, razon_rad, "exacta")
gdl_c, kt_c = barrido(L_chica, niveles_chica, razon_chica, "uniforme")

fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
ax.plot(gdl_c, kt_c, color=C_CHICA, marker="o", lw=1.5, ls="--", ms=5, mec="white", mew=0.8,
        label=r"Referencia: $S$ uniforme, placa chica $L = 5a$")
ax.plot(gdl, kt_u, color=C1, marker="o", lw=2, ms=6, mec="white", mew=0.8,
        label=r"MEF, $S$ uniforme en $x = L$ ($L = 20a$)")
ax.plot(gdl, kt_e, color=C_TRAC, marker="^", lw=2, ms=6, mec="white", mew=0.8,
        label=r"MEF, tracciones de Kirsch en los bordes ($L = 20a$)")
ax.axhline(3, color=KIRSCH, lw=1.2, ls=":", label=r"Kirsch, $K_t = 3$")
for nombre, s in sols.items():
    ax.plot(s.n_gdl, s.sigma_nodos[s.B, 0]/S_Pa, ls="none", ms=9, mfc="none", mew=1.5,
            color=estilo[nombre]["color"], marker=estilo[nombre]["marker"], label=nombre)
ax.annotate(f"{kt_u[-1]:.3f}", (gdl[-1], kt_u[-1]), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9, color="0.3")
ax.annotate(f"{kt_c[-1]:.3f}", (gdl_c[-1], kt_c[-1]), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9, color="0.3")
ax.set_xscale("log")
ax.set_xlabel("Numero de GDL")
ax.set_ylabel(r"$K_t = \sigma_{xx}(B) \,/\, S$")
ax.set_title(r"Convergencia de $K_t$ al refinar la malla")
ax.legend(frameon=False, fontsize=8)
fig.savefig(os.path.join(carpeta, "convergencia_Kt.png"), dpi=200)

print(f"\nFiguras guardadas en {os.path.abspath(carpeta)}")
