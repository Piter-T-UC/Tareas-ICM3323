"""Estudio de convergencia de la placa con agujero (L/a = 20, S uniforme en x = L).
Se refina la malla de problema_3.py (Q9) y se compara con Kirsch."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unidades import ureg
from etapa1_malla.mallador import bloque, unir, arco, nodos_en, apoyos_en
from etapa1_malla.cargas import lados_desde_nodos
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_global import gen_matriz_global
from etapa3_matrices.gen_f_distribuida import gen_f_distribuida
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_reducido import Sist_red
from etapa4_resolver.u_completa import U_completa
from etapa4_resolver.tensiones import calc_tensiones

MPa = ureg.MPa
E, nu, t = 200 * ureg.GPa, 0.3, 10 * ureg.mm
a, L, S = 10 * ureg.mm, 200 * ureg.mm, 100 * MPa
a_m, L_m = a.to(ureg.m).magnitude, L.to(ureg.m).magnitude
razon_rad = 15


def resolver(n_rad, n_ang):
    p45 = a_m/np.sqrt(2)*np.array([1, 1])
    b1 = bloque([(a_m, 0), (L_m, 0), (L_m, L_m), p45], n_rad, n_ang, "Q9",
                lados={3: arco((0, 0), a_m, 0, np.pi/4)}, razon=(razon_rad, 1))
    b2 = bloque([p45, (L_m, L_m), (0, L_m), (0, a_m)], n_rad, n_ang, "Q9",
                lados={3: arco((0, 0), a_m, np.pi/4, np.pi/2)}, razon=(razon_rad, 1))
    xy_m, con = unir(b1, b2)
    xy = (xy_m * ureg.m).to(ureg.mm)
    apoyos = np.array(apoyos_en(xy, lambda x, y: np.isclose(y, 0), 0, 1)
                      + apoyos_en(xy, lambda x, y: np.isclose(x, 0), 1, 0), dtype=int)
    trac = [(n, S, 0*MPa) for n in nodos_en(xy, lambda x, y: np.isclose(x, L_m))]
    cd = lados_desde_nodos(con, trac)
    D = matriz_D(E, nu, "tension_plana")
    K = gen_matriz_global(xy, con, D, t)
    F = gen_f_distribuida(xy, con, cd, t)
    fijos = Restriciones(apoyos)
    U_red, K_red, F_red = Sist_red(K, F, fijos)
    U = U_completa(fijos, U_red)
    sig, _ = calc_tensiones(xy, con, U, D)
    nodo = lambda px, py: nodos_en(xy, lambda x, y: np.isclose(x, px) and np.isclose(y, py))[0]
    nA, nB, nC = nodo(a_m, 0), nodo(0, a_m), nodo(L_m, 0)
    Sp = S.to(ureg.Pa).magnitude
    Un = U.reshape(len(xy), 2)
    return dict(gdl=2*len(xy), nel=len(con), Kt=sig[nB, 0]/Sp, syyA=sig[nA, 1]/Sp,
                uxC=Un[nC, 0]*1e3)  # mm


kirsch = dict(Kt=3.0, syyA=-1.0, uxC=0.10066)
mallas = [(5, 2), (10, 4), (20, 8), (30, 12), (40, 16)]
res = []
print(f"{'malla':>8} {'elem':>6} {'GDL':>7} {'Kt':>8} {'err Kt %':>9} {'syyA/S':>8} {'err %':>7} {'uxC mm':>9} {'err %':>7}")
for nr, na in mallas:
    r = resolver(nr, na); res.append(r)
    print(f"{nr:>4}x{na:<3} {r['nel']:>6} {r['gdl']:>7} {r['Kt']:>8.4f} {100*(r['Kt']/3-1):>9.2f} "
          f"{r['syyA']:>8.4f} {100*(abs(r['syyA'])-1):>7.2f} {r['uxC']:>9.5f} {100*(r['uxC']/0.10066-1):>7.2f}")

g = [r["gdl"] for r in res]
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogx(g, [100*abs(r["Kt"]/3-1) for r in res], "o-", label=r"$K_t=\sigma_{xx}(B)/S$")
ax.semilogx(g, [100*abs(abs(r["syyA"])-1) for r in res], "s-", label=r"$|\sigma_{yy}(A)|/S$")
ax.semilogx(g, [100*abs(r["uxC"]/0.10066-1) for r in res], "^-", label=r"$u_x(C)$")
ax.set_xlabel("GDL"); ax.set_ylabel("error respecto de Kirsch [%]")
ax.grid(True, which="both", alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig("../figuras/convergencia_error.png", dpi=200)
