import os

import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
from gen_matriz_global import gen_matriz_global
from gen_f_global import gen_f_global
from cargas_nodales import cargas_nodales_equivalentes
from restricciones import Restriciones
from sistema_reducido import Sist_red
from u_completa import U_completa
from graficar_reticulado import graficar_reticulado, graficar_deflexion_vertical
from esfuerzos import calc_esfuerzos, calc_reacciones, esfuerzos_a_lo_largo
from exportar_latex import informe_latex
from diagramas import (graficar_fuerzas_barra, graficar_tensiones_barra,
                       graficar_fuerzas_3d, graficar_tensiones_3d)
from refinar_malla import refinar


M_Elasticidad = 206 * ureg.GPa
G_Cortante = 80 * ureg.GPa       # dato del enunciado

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
z_at = 1.4     # altura donde el atiesador se une a la columna inclinada

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

    # atiesadores: punto de cada columna inclinada a z_at -> esquina trasera del techo
    [Ancho/2*z_at/Altura,0,z_at],#10 sobre columna 0-8
    [Ancho/2*z_at/Altura,Largo,z_at],#11 sobre columna 3-9


]) * ureg.meter

conectividad = np.array([
    [0, 10], #columna inclinada (tramo inferior)
    [1, 5],
    [2, 6],
    [3, 11],#columna inclinada (tramo inferior)
    #techo
    [4,8],
    [8,5],
    [5,6],
    [6,9],
    [9,7],
    [7,4],

    [10, 8],#columna inclinada (tramo superior)
    [11, 9],
    [10, 4],#atiesadores laterales (caras cortas, no tapan el frente)
    [11, 7],

], dtype=int)

# vec_ref = Z global: x_L = Z x z_L queda horizontal y el peso propio cae solo en el
# plano local y_L-z_L (V_y, M_x, N). En barras verticales Z es paralelo -> se usa Y.
d = (xyz[conectividad[:, 1]] - xyz[conectividad[:, 0]]).magnitude
vertical = np.hypot(d[:, 0], d[:, 1]) < 1e-9
vec_ref = np.where(vertical[:, None], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])

apoyos = np.array([
    [0, 1, 1, 1, 1, 1, 1],   # nodo 0: empotrado (6 restricciones)
    [1, 1, 1, 1, 1, 1, 1],   # nodo 1: empotrado
    [2, 1, 1, 1, 1, 1, 1],   # nodo 2: empotrado
    [3, 1, 1, 1, 1, 1, 1],   # nodo 3: empotrado
], dtype=int)


densidad = 7800*ureg.kg/ureg.m**3
g = 9.80*ureg.m/ureg.s**2
peso_lineal = (densidad*g*Area).to(ureg.N/ureg.m)     # peso propio por metro de tubo
# refinamiento: {indice de barra: n sub-elementos}; los nodos nuevos van al final (0-11 no cambian)
divisiones = {6: 10,
               9: 10}      # vigas de 4 m del techo
xyz, conectividad, vec_ref, props, _, barras = refinar(
    xyz, conectividad, vec_ref, divisiones,
    dict(Area=Area, I_x=I_x, I_y=I_y, J_p=J_p))
Area, I_x, I_y, J_p = (props[k] for k in ("Area", "I_x", "I_y", "J_p"))

K_global = gen_matriz_global(xyz, conectividad, Area, I_x, I_y, J_p, vec_ref,
                                M_Elasticidad, G_Cortante)

# peso propio como cargas nodales equivalentes: por elemento qL/2 en cada nodo (vertical)
# y momentos +-(L/12)(d x q); se aplican igual que cargas puntuales
fuerzas = cargas_nodales_equivalentes(xyz, conectividad, peso_lineal)
F_global = gen_f_global(xyz, fuerzas)
print("Cargas nodales equivalentes del peso propio (nodos 4-11) [N, N*m]:")
for nodo, *c in fuerzas:
    if 4 <= nodo <= 11:
        print(f"  nodo {nodo:2d}: " + "  ".join(f"{v.magnitude: 8.4f}" for v in c))
print(f"  suma Fz (todos los nodos) = {F_global[2::6].sum():.2f} N")
dofs_fijos = Restriciones(apoyos)
U_red, K_red, F_red = Sist_red(K_global, F_global, dofs_fijos)
U = U_completa(dofs_fijos, U_red)

print("dofs_fijos:", dofs_fijos)
print("rank(K_reducida):", np.linalg.matrix_rank(K_red), "de", K_red.shape[0])
print("\nDesplazamientos y giros por nodo:")
#aqui obetengo por nodo
U_nodos = U.reshape(len(xyz), 6)
#para cada nodo traslaciones y rotaciones
traslaciones = (U_nodos[:, :3] * ureg.meter).to(ureg.mm)
rotaciones = (U_nodos[:, 3:] * ureg.radian).to(ureg.degree)
for i in range(len(xyz)):
    ux, uy, uz = traslaciones[i]
    tx, ty, tz = rotaciones[i]
    print(f"  nodo {i}: ux={ux: .4f}  uy={uy: .4f}  uz={uz: .4f}  |  "
            f"thx={tx: .4f}  thy={ty: .4f}  thz={tz: .4f}")

fig, axes = graficar_reticulado(xyz, conectividad, U=U, apoyos=apoyos,
                                    escala=5, titulo="paradero de Bus")
plt.savefig("Paradero de Bus.png", dpi=150)
fig, _ = graficar_deflexion_vertical(xyz, conectividad, U, apoyos=apoyos, escala=5,
                                     titulo="Paradero de Bus")
fig.savefig("Paradero de Bus - deflexion vertical.png", dpi=150)
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

# esfuerzos internos: f = k u (todas las cargas estan en los nodos)
res = calc_esfuerzos(xyz, conectividad, U, Area, I_x, I_y, J_p, vec_ref,
                     M_Elasticidad, G_Cortante, c_x=radio, c_y=radio, r_t=radio)

# chequeo: el equilibrio a lo largo de la barra debe llegar a los valores del nodo final
err = max(abs(esfuerzos_a_lo_largo(res, e)[c][-1] - res[c][e, 1])
          for e in range(len(conectividad)) for c in ("V_x", "V_y", "N", "M_x", "M_y", "T"))
print(f"error de cierre de los diagramas: {err:.2e}")

# verificacion de las restricciones de diseno
S_y = 300*ureg.MPa
delta_max = 10*ureg.mm
desp = np.abs(traslaciones.magnitude[:, 2])     # deflexion vertical |u_z|
n_d = int(np.argmax(desp))
print(f"\ndeflexion vertical maxima = {desp[n_d]:.2f} mm en nodo {n_d} {xyz[n_d].magnitude} m"
      f"  (limite {delta_max:~P}) -> {'CUMPLE' if desp[n_d] < delta_max.magnitude else 'NO CUMPLE'}")
print("tension de von Mises maxima por barra:")
svm_barras = []
for i, elems in enumerate(barras):
    svm_barras.append(max(esfuerzos_a_lo_largo(res, e)["sigma_vm"].max() for e in elems)/1e6)
    a, b = conectividad[elems[0]][0], conectividad[elems[-1]][1]
    print(f"  barra {i:2d} (nodos {a}-{b}): {svm_barras[-1]:6.2f} MPa")
i_s = int(np.argmax(svm_barras))
print(f"sigma_vm max = {svm_barras[i_s]:.2f} MPa en barra {i_s}  ->  FS = S_y/sigma = "
      f"{S_y.magnitude/svm_barras[i_s]:.1f}")

# fuerzas internas en N y N*m solo para este problema (diagramas.py queda en kN y kN*m)
import diagramas
for clave, (simb, unidad, _) in diagramas._FUERZAS.items():
    diagramas._FUERZAS[clave] = (simb, unidad[1:], 1)

# diagramas 2D por barra original, juntando sus sub-elementos (se guardan y se cierran)
carpeta = "diagramas_P2"
os.makedirs(carpeta, exist_ok=True)
for i, elems in enumerate(barras):
    fig = graficar_fuerzas_barra(res, elems, conectividad)
    fig.savefig(os.path.join(carpeta, f"barra_{i:02d}_fuerzas.png"), dpi=150)
    plt.close(fig)
    fig = graficar_tensiones_barra(res, elems, conectividad)
    fig.savefig(os.path.join(carpeta, f"barra_{i:02d}_tensiones.png"), dpi=150)
    plt.close(fig)

# diagramas 3D sobre la estructura
fig = graficar_fuerzas_3d(xyz, conectividad, res, titulo="Paradero - fuerzas internas")
fig.savefig(os.path.join(carpeta, "fuerzas_3d.png"), dpi=150)
fig = graficar_tensiones_3d(xyz, conectividad, res, titulo="Paradero - tensiones")
fig.savefig(os.path.join(carpeta, "tensiones_3d.png"), dpi=150)
print(f"diagramas guardados en {carpeta}/")
plt.show()