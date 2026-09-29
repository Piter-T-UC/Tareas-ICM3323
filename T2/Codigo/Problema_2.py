import os

import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
from gen_matriz_global import gen_matriz_global
from gen_f_global import gen_f_global
from gen_f_distribuida import gen_f_distribuida
from restricciones import Restriciones
from sistema_reducido import Sist_red
from u_completa import U_completa
from graficar_reticulado import graficar_reticulado
from esfuerzos import calc_esfuerzos, calc_reacciones, esfuerzos_a_lo_largo
from exportar_latex import informe_latex
from diagramas import (graficar_fuerzas_elemento, graficar_tensiones_elemento,
                       graficar_fuerzas_3d, graficar_tensiones_3d)


M_Elasticidad = 206 * ureg.GPa
nu = 0.3
G_Cortante = M_Elasticidad / (2*(1+nu))

#Variables
diametro=30*ureg.mm
espesor=3*ureg.mm   
radio = diametro/2
Area = np.pi * (radio**2 - (radio-espesor)**2)
I_x = np.pi * radio**4 / 4-np.pi * (radio-espesor)**4 / 4
I_y = I_x
J_p = 2 * I_x

Largo=4
Ancho=2
Altura=2

xyz = np.array([
    [0,0,0],#base
    [Ancho,0,0],
    [Ancho,Largo,0],
    [0,Largo,0], #3

    [0,0,Altura],#techo
    [Ancho,0,Altura],
    [Ancho,Largo,Altura],
    [0,Largo,Altura], #7
 
    [Ancho/2,0,Altura],#medios #8
    [Ancho/2,Largo,Altura],

]) * ureg.meter

conectividad = np.array([
    [0, 8], #arriba
    [1, 5],
    [2, 6],
    [3, 9],#techo
    [4,8],
    [8,5],
    [5,6],

    [6,9],
    [9,7],
    [7,4],
], dtype=int)

# da lo mismo la orientacion todos son de seccion constante
vec_ref = np.array([[0.0, 1.0, 1.0]] * len(conectividad))

apoyos = np.array([
    [0, 1, 1, 1, 1, 1, 1],   # nodo 0: empotrado (6 restricciones)
    [1, 1, 1, 1, 1, 1, 1],   # nodo 1: empotrado
    [2, 1, 1, 1, 1, 1, 1],   # nodo 2: empotrado
    [3, 1, 1, 1, 1, 1, 1],   # nodo 2: empotrado
], dtype=int)

densidad = 7850*ureg.kg/ureg.m**3
g = 9.81*ureg.m/ureg.s**2
peso_lineal = (densidad*g*Area).to(ureg.N/ureg.m)     # peso propio por metro de tubo

# sin cargas nodales: (Nodo, F x,y,z ; M x,y,z)
fuerzas = []

# peso propio: carga uniforme en -z global sobre todos los elementos
# (n1, n2, [fx,fy,fz], [mx,my,mz]) -> constante y en ejes globales
cargas_dist = [(n1, n2, np.array([0.0, 0.0, -1.0])*peso_lineal, np.zeros(3)*ureg.N)
               for n1, n2 in conectividad]


K_global = gen_matriz_global(xyz, conectividad, Area, I_x, I_y, J_p, vec_ref,
                                M_Elasticidad, G_Cortante)

F_dist, f_eq_locales, q_locales = gen_f_distribuida(xyz, conectividad, vec_ref, cargas_dist)
F_global = gen_f_global(xyz, fuerzas) + F_dist
dofs_fijos = Restriciones(apoyos)
U_red, K_red, F_red = Sist_red(K_global, F_global, dofs_fijos)
U = U_completa(dofs_fijos, U_red)

print("dofs_fijos:", dofs_fijos)
print("rank(K_reducida):", np.linalg.matrix_rank(K_red), "de", K_red.shape[0])
print("\nDesplazamientos y giros por nodo:")
U_nodos = U.reshape(len(xyz), 6)
traslaciones = (U_nodos[:, :3] * ureg.meter).to(ureg.mm)
rotaciones = (U_nodos[:, 3:] * ureg.radian).to(ureg.degree)
for i in range(len(xyz)):
    ux, uy, uz = traslaciones[i]
    tx, ty, tz = rotaciones[i]
    print(f"  nodo {i}: ux={ux: .4f}  uy={uy: .4f}  uz={uz: .4f}  |  "
            f"thx={tx: .4f}  thy={ty: .4f}  thz={tz: .4f}")

fig, axes = graficar_reticulado(xyz, conectividad, U=U, apoyos=apoyos,
                                    escala=1, titulo="paradero de Bus")
plt.savefig("Paradero de Bus.png", dpi=150)
print("\nguardado")

# reacciones y equilibrio: sum Rz debe igualar el peso total
R_apoyos = calc_reacciones(K_global, U, F_global, dofs_fijos)
peso_total = peso_lineal.magnitude * sum(np.linalg.norm((xyz[b] - xyz[a]).to(ureg.meter).magnitude)
                                         for a, b in conectividad)
print(f"\npeso lineal = {peso_lineal:.3f}   peso total = {peso_total:.2f} N")
print("Reacciones [N, N*m]:")
for n in apoyos[:, 0]:
    print(f"  nodo {n}: " + "  ".join(f"{v: .3f}" for v in R_apoyos[n]))
print("sum R (x, y, z) =", np.round(R_apoyos[:, :3].sum(axis=0), 4), "N")

# esfuerzos internos (restando las fuerzas de empotramiento de la carga distribuida)
res = calc_esfuerzos(xyz, conectividad, U, Area, I_x, I_y, J_p, vec_ref,
                     M_Elasticidad, G_Cortante, c_x=radio, c_y=radio, r_t=radio,
                     f_eq_locales=f_eq_locales)

# chequeo: el equilibrio a lo largo de la barra debe llegar a los valores del nodo final
err = max(abs(esfuerzos_a_lo_largo(res, e, q_locales[e])[c][-1] - res[c][e, 1])
          for e in range(len(conectividad)) for c in ("V_x", "V_y", "N", "M_x", "M_y", "T"))
print(f"error de cierre de los diagramas: {err:.2e}")

# diagramas 2D por elemento (se guardan y se cierran: serian 20 ventanas)
carpeta = "diagramas_P2"
os.makedirs(carpeta, exist_ok=True)
for e in range(len(conectividad)):
    fig = graficar_fuerzas_elemento(res, e, q_locales, conectividad)
    fig.savefig(os.path.join(carpeta, f"elem_{e:02d}_fuerzas.png"), dpi=150)
    plt.close(fig)
    fig = graficar_tensiones_elemento(res, e, q_locales, conectividad)
    fig.savefig(os.path.join(carpeta, f"elem_{e:02d}_tensiones.png"), dpi=150)
    plt.close(fig)

# diagramas 3D sobre la estructura
fig = graficar_fuerzas_3d(xyz, conectividad, res, q_locales, titulo="Paradero - fuerzas internas")
fig.savefig(os.path.join(carpeta, "fuerzas_3d.png"), dpi=150)
fig = graficar_tensiones_3d(xyz, conectividad, res, q_locales, titulo="Paradero - tensiones")
fig.savefig(os.path.join(carpeta, "tensiones_3d.png"), dpi=150)
print(f"diagramas guardados en {carpeta}/")
plt.show()