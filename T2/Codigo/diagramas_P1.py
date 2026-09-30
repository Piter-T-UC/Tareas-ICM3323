import os
import numpy as np
import matplotlib.pyplot as plt

# Diagramas de corte, momento y torsor del Problema 1 (leer despues de correr problema_1.py)
# convencion de la parte a: V = -dM/dz, M = P (a - z) en 0 <= z <= a

carpeta_datos = "datos_P1"
carpeta_fig = "diagramas_P1"
n_mef = 2                           # malla del caso mostrado (nodos 0-1-2)
_COLOR = "#d65f28"

an = np.genfromtxt(os.path.join(carpeta_datos, "analitica.dat"), names=True)
mef = np.genfromtxt(os.path.join(carpeta_datos, f"mef_n{n_mef}_esfuerzos.dat"), names=True)

# (clave, titulo, unidad) ; datos en N y N*m -> kN y kN*m
paneles = [("V", "Corte $V$", "kN"), ("M", "Momento flector $M$", r"kN$\cdot$m"),
           ("T", "Torsor $T$", r"kN$\cdot$m")]

fig, axes = plt.subplots(len(paneles), 1, figsize=(8, 7.5), sharex=True)
for ax, (c, titulo, unidad) in zip(axes, paneles):
    z = an["z"] / 1e3
    v = an[c] / 1e3
    if c == "V":                    # salto en z = a: dibujar vertical en vez de interpolar
        z = np.array([0, 0.5, 0.5, 1.0])
        v = np.array([1.0, 1.0, 0.0, 0.0])
    ax.fill_between(z, v, color=_COLOR, alpha=0.2, linewidth=0)
    ax.plot(z, v, color=_COLOR, linewidth=2, label="Analítica")
    if c in mef.dtype.names:
        ax.plot(mef["z"]/1e3, mef[c]/1e3, "o", color="black", markersize=5,
                label=f"MEF ({n_mef} elementos)")
    ax.axhline(0, color="0.4", linewidth=0.8)
    i = int(np.argmax(np.abs(v)))
    ax.annotate(f"{v[i]:.3g} {unidad}", (z[i], v[i]), textcoords="offset points",
                xytext=(6, 6), fontsize=9)
    ax.set_title(f"{titulo} [{unidad}]", fontsize=10)
    ax.set_ylabel(unidad)
    ax.grid(alpha=0.3)
    ax.margins(y=0.25)
axes[0].legend(fontsize=8, loc="upper right")
axes[-1].set_xlabel("z [m]  (z = 0 empotramiento, z = L extremo libre)")
fig.suptitle("Problema 1 - Diagramas de esfuerzos internos\n"
             "P = 1 kN en z = L/2,  M = 2 kN·m (torsor) en z = L")
fig.tight_layout()

os.makedirs(carpeta_fig, exist_ok=True)
ruta = os.path.join(carpeta_fig, "diagramas_V_M_T.png")
fig.savefig(ruta, dpi=200)
print(f"figura guardada en {ruta}")
