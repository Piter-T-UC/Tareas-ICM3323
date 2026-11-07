import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
# 1. malla
from etapa1_malla.mallador import bloque, unir, arco, nodos_en, apoyos_en
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
from etapa5_graficar.graficar_malla import graficar_malla
from etapa5_graficar.graficar_tensiones import graficar_tensiones

MPa = ureg.MPa


titulo = "Placa con agujero (Kirsch) - 1/4 por simetria, L/a = 20, malla 2"

# Material y seccion
E = 200 * ureg.GPa
nu = 0.3
espesor = 10 * ureg.mm
estado = "tension_plana" # o "deformacion_plana"

# Geometria
a = 10 * ureg.mm
L = 200 * ureg.mm               # L/a = 20: borde cargado lejos del agujero (cerca de placa infinita)

# Carga
S = 100 * ureg.MPa              # traccion constante en el borde x = L


# --- Malla 2: Q9, la malla 1 (problema 2) con el doble de divisiones en cada direccion ----------
# Dos bloques separados por la diagonal y = x, cada uno con n_rad x n_ang elementos:
#   bloque 1: arco r = a de 0 a 45 grados  ->  borde x = L
#   bloque 2: arco r = a de 45 a 90 grados ->  borde y = L
# razon_rad = largo del ultimo elemento radial / el primero. Con L/a = 20 se usa una razon grande
# para que los elementos junto al agujero queden chicos (~1-2 mm) y los del borde exterior grandes
tipo = "Q9"
n_rad, n_ang = 20, 8
razon_rad = 15

a_m, L_m = a.to(ureg.m).magnitude, L.to(ureg.m).magnitude
p45 = a_m/np.sqrt(2)*np.array([1, 1])
bloque_1 = bloque([(a_m, 0), (L_m, 0), (L_m, L_m), p45], n_rad, n_ang, tipo,
                  lados={3: arco((0, 0), a_m, 0, np.pi/4)}, razon=(razon_rad, 1))
bloque_2 = bloque([p45, (L_m, L_m), (0, L_m), (0, a_m)], n_rad, n_ang, tipo,
                  lados={3: arco((0, 0), a_m, np.pi/4, np.pi/2)}, razon=(razon_rad, 1))
xy_m, conectividad = unir(bloque_1, bloque_2)
xy = (xy_m * ureg.m).to(ureg.mm)

# Seleccion de nodos por coordenadas (x, y en metros)
en_x0 = lambda x, y: np.isclose(x, 0)
en_y0 = lambda x, y: np.isclose(y, 0)
en_xL = lambda x, y: np.isclose(x, L_m)
nodo_en = lambda px, py: nodos_en(xy, lambda x, y: np.isclose(x, px) and np.isclose(y, py))[0]

# Apoyos por simetria: y = 0 -> uy = 0, x = 0 -> ux = 0
apoyos = np.array(apoyos_en(xy, en_y0, 0, 1) + apoyos_en(xy, en_x0, 1, 0), dtype=int)

# Fuerzas nodales: (nodo, Fx, Fy) en ejes globales
fuerzas = []

# Solo el borde x = L cargado con S constante: (nodo, tx, ty) en todos sus nodos
tracciones_x_L = [(n, S, 0*MPa) for n in nodos_en(xy, en_xL)]
cargas_dist = lados_desde_nodos(conectividad, tracciones_x_L)

# Puntos de control
control = [("A (a, 0)", nodo_en(a_m, 0)), ("B (0, a)", nodo_en(0, a_m)),
           ("C (L, 0)", nodo_en(L_m, 0)), ("D (0, L)", nodo_en(0, L_m))]


validar_malla(xy, conectividad)
leer_cargas_dist(conectividad, cargas_dist, espesor)
print(f"{titulo} ({tipo})\n{len(xy)} nodos, {len(conectividad)} elementos, {2*len(xy)} GDL")
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
for nombre, n in control:
    ux, uy = U_nodos[n].magnitude
    print(f"  {nombre}: nodo {n:3d}  ux = {ux: .6f}  uy = {uy: .6f}")

# Tensiones en los nodos (promedio de los elementos vecinos) [Pa]
sigma_nodos, sigma_elem = calc_tensiones(xy, conectividad, U, D)
print("\nTensiones en puntos de control [MPa]:")
for nombre, n in control:
    sxx, syy, txy = sigma_nodos[n]/1e6
    print(f"  {nombre}: nodo {n:3d}  sxx = {sxx: 9.3f}  syy = {syy: 9.3f}  txy = {txy: 9.3f}"
          f"  VM = {von_mises(sigma_nodos[n])/1e6: 9.3f}")
n_B = control[1][1]
print(f"  Kt = sxx(B) / S = {sigma_nodos[n_B, 0]/S.to(ureg.Pa).magnitude:.3f}   (Kirsch placa infinita: 3)")

# Cuanto se lleva cada nodo de la carga distribuida
print("\nFuerzas nodales equivalentes de la carga en x = L [kN]:")
for n, fx, fy in fuerzas_desde_vector(F_dist):
    print(f"  nodo {n:3d}  Fx = {fx.to(ureg.kN).magnitude: .4f}  Fy = {fy.to(ureg.kN).magnitude: .4f}")
print(f"  suma Fx = {F_dist[0::2].sum()/1e3:.4f} kN   (S t L = {(S*espesor*L).to(ureg.kN).magnitude:.4f} kN)")

# Graficos: malla con la carga distribuida, cuanto se lleva cada nodo, tensiones
fig, ax = graficar_malla(xy, conectividad, apoyos, fuerzas, cargas_dist, espesor, titulo=f"{titulo}, {tipo}",
                         numeracion=False)
fig.savefig("malla_problema_3.png", dpi=150)

fig, ax = graficar_malla(xy, conectividad, apoyos, fuerzas_desde_vector(F_dist), numeracion=False,
                         titulo="Fuerzas nodales equivalentes de S en x = L")
fig.savefig("fuerzas_nodales_problema_3.png", dpi=150)

# Tensiones: zona del agujero (0 a 5a), donde estan los gradientes; lejos el campo es uniforme
fig = graficar_tensiones(xy, conectividad, sigma_nodos, titulo=f"{titulo}, {tipo} - S = {S:~} en x = L",
                         zoom=(0, 5*a.magnitude, 0, 5*a.magnitude))
fig.savefig("tensiones_problema_3.png", dpi=150)
plt.show()
