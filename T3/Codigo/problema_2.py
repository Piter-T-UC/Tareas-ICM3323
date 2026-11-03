import os

import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
# 1. malla
from etapa1_malla.leer_malla import leer_nodos, leer_conectividad, leer_apoyos
from etapa1_malla.elementos import validar_malla
from etapa1_malla.cargas import leer_cargas_dist, lados_desde_nodos
# 3. matrices
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_global import gen_matriz_global
from etapa3_matrices.gen_f_global import gen_f_global, fuerzas_desde_vector
from etapa3_matrices.gen_f_distribuida import gen_f_distribuida
# 4. resolver
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_reducido import Sist_red
from etapa4_resolver.u_completa import U_completa
from etapa4_resolver.tensiones import calc_tensiones, von_mises
# 5. graficar
from etapa5_graficar.graficar_malla import graficar_malla, graficar_nodos
from etapa5_graficar.graficar_tensiones import graficar_tensiones

MPa = ureg.MPa


titulo = "Placa con agujero (Kirsch) - 1/4 por simetria"

# Material y seccion
E = 200 * ureg.GPa
nu = 0.3
espesor = 10 * ureg.mm
estado = "tension_plana" # o "deformacion_plana"

# Geometria
a = 10 * ureg.mm                
L = 50 * ureg.mm               

# Carga
S = 100 * ureg.MPa              # traccion constante en el borde x = L


# --- Malla Q9: 48 elementos, 6 radiales x 4 angulares en cada mitad de 45 grados -------------
carpeta_mallas = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mallas")
xy = leer_nodos(os.path.join(carpeta_mallas, "problema_2_nodos.txt")) * ureg.mm
conectividad = leer_conectividad(os.path.join(carpeta_mallas, "problema_2_elementos.txt"))

# Apoyos por simetria (mallas/problema_2_apoyos.txt): "nodo  rx  ry" por linea
apoyos = leer_apoyos(os.path.join(carpeta_mallas, "problema_2_apoyos.txt"))

# Fuerzas nodales: (nodo, Fx, Fy) en ejes globales
fuerzas = []

# Malla con el numero de cada nodo; el zoom [mm] es para la zona fina alrededor del agujero
fig, ax = graficar_nodos(xy, conectividad, titulo=f"{titulo}, Q9 - numeracion de nodos", zoom=(0, 20, 0, 20))
fig.savefig("malla_nodos_problema_2.png", dpi=200)

# Solo el borde x = L cargado con S constante: (nodo, tx, ty) en las esquinas de sus lados
tracciones_x_L = [(n, S, 0*MPa) for n in (108, 110, 112, 114, 116)]
cargas_dist = lados_desde_nodos(conectividad, tracciones_x_L)


validar_malla(xy, conectividad)
leer_cargas_dist(conectividad, cargas_dist, espesor)
print(f"{titulo} (Q9)\n{len(xy)} nodos, {len(conectividad)} elementos, {2*len(xy)} GDL")
print(f"a = {a}, L = {L}, S = {S}")
print(f"E = {E}, nu = {nu}, t = {espesor}, estado: {estado}")

D = matriz_D(E, nu, estado)
print(f"D [GPa] =\n{np.round(D/1e9, 3)}")

# Sistema K U = F (floats SI: N/m, N, m)
K_global = gen_matriz_global(xy, conectividad, D, espesor)
F_dist = gen_f_distribuida(xy, conectividad, cargas_dist, espesor)
F_global = gen_f_global(xy, fuerzas) + F_dist
print(f"K_global {K_global.shape}, suma F = [{F_global[0::2].sum():.1f}, {F_global[1::2].sum():.1f}] N")

dofs_fijos = Restriciones(apoyos)
U_red, K_red, F_red = Sist_red(K_global, F_global, dofs_fijos)
U = U_completa(dofs_fijos, U_red)
U_nodos = (U.reshape(len(xy), 2) * ureg.meter).to(ureg.mm)     # [ux, uy] por nodo

print("\nDesplazamientos en puntos de control [mm]:")
for nombre, n in [("A (a, 0)", 0), ("B (0, a)", 124), ("C (L, 0)", 108), ("D (0, L)", 220)]:
    ux, uy = U_nodos[n].magnitude
    print(f"  {nombre}: nodo {n:3d}  ux = {ux: .6f}  uy = {uy: .6f}")

# Tensiones en los nodos (promedio de los elementos vecinos) [Pa]
sigma_nodos, sigma_elem = calc_tensiones(xy, conectividad, U, D)
print("\nTensiones en puntos de control [MPa]:")
for nombre, n in [("A (a, 0)", 0), ("B (0, a)", 124), ("C (L, 0)", 108), ("D (0, L)", 220)]:
    sxx, syy, txy = sigma_nodos[n]/1e6
    print(f"  {nombre}: nodo {n:3d}  sxx = {sxx: 9.3f}  syy = {syy: 9.3f}  txy = {txy: 9.3f}"
          f"  VM = {von_mises(sigma_nodos[n])/1e6: 9.3f}")
print(f"  Kt = sxx(B) / S = {sigma_nodos[124, 0]/S.to(ureg.Pa).magnitude:.3f}   (Kirsch placa infinita: 3)")

# Cuanto se lleva cada nodo de la carga distribuida
print("\nFuerzas nodales equivalentes de la carga en x = L [kN]:")
for n, fx, fy in fuerzas_desde_vector(F_dist):
    print(f"  nodo {n:3d}  Fx = {fx.to(ureg.kN).magnitude: .4f}  Fy = {fy.to(ureg.kN).magnitude: .4f}")
print(f"  suma Fx = {F_dist[0::2].sum()/1e3:.4f} kN   (S t L = {(S*espesor*L).to(ureg.kN).magnitude:.4f} kN)")

# Graficos: malla con la carga distribuida, cuanto se lleva cada nodo, tensiones
fig, ax = graficar_malla(xy, conectividad, apoyos, fuerzas, cargas_dist, espesor, titulo=f"{titulo}, Q9",
                         numeracion=False)
fig.savefig("malla_problema_2.png", dpi=150)

fig, ax = graficar_malla(xy, conectividad, apoyos, fuerzas_desde_vector(F_dist), numeracion=False,
                         titulo="Fuerzas nodales equivalentes de S en x = L")
fig.savefig("fuerzas_nodales_problema_2.png", dpi=150)

fig = graficar_tensiones(xy, conectividad, sigma_nodos, titulo=f"{titulo}, Q9 - S = {S:~} en x = L")
fig.savefig("tensiones_problema_2.png", dpi=150)
plt.show()
