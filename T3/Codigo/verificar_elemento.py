import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
# 2. interpolacion
from etapa2_interpolacion.funciones_forma import funciones_forma, NODOS_NATURALES
# 3. matrices
from etapa3_matrices.matriz_B import jacobiano, matriz_B
from etapa3_matrices.matriz_D import matriz_D
# 5. graficar
from etapa5_graficar.graficar_elemento import (graficar_funciones_forma, graficar_mapeo,
                                               graficar_deformacion, graficar_matrices)

np.set_printoptions(precision=4, suppress=True, linewidth=160)


# Elemento a revisar: T3, T6, Q4, Q8 o Q9
tipo = "Q8"
# Punto (xi, eta) donde se imprimen las matrices (en triangulos: xi, eta >= 0 y xi + eta <= 1)
punto = (0.25, 0.25) if tipo[0] == "T" else (0.3, -0.4)

# Material
E = 200 * ureg.GPa
nu = 0.3
estado = "tension_plana"

# Elemento fisico [m]: los nodos naturales pasados por un mapeo suave (lados curvos en T6/Q8/Q9).
# Se puede reemplazar por coordenadas a mano: xy_e = np.array([[x0, y0], [x1, y1], ...])
distorsion = lambda xi, eta: np.c_[2 + 1.2*xi + 0.3*eta + 0.15*xi*eta + 0.2*eta**2,
                                   1 + 0.2*xi + 0.9*eta - 0.15*xi**2]
xy_e = distorsion(*NODOS_NATURALES[tipo].T)

# Desplazamientos conocidos: (nombre, u(x, y) -> (ux, uy), eps exacta [exx, eyy, gxy])
th = 0.1
casos = [
    ("Estiramiento en x", lambda x, y: (0.1*x, 0*y), [0.1, 0, 0]),
    ("Estiramiento en y", lambda x, y: (0*x, 0.1*y), [0, 0.1, 0]),
    ("Corte simple", lambda x, y: (0.1*y, 0*x), [0, 0, 0.1]),
    ("Rotacion rigida", lambda x, y: (-th*y, th*x), [0, 0, 0]),
]


# --- Matrices en el punto (consola) -----------------------------------------------------------
N, dN = funciones_forma(tipo, *punto)
J, detJ, dN_dx = jacobiano(dN, xy_e)
B, _, _ = matriz_B(tipo, xy_e, *punto)
D = matriz_D(E, nu, estado)

print(f"Elemento {tipo}, punto (xi, eta) = {punto} -> (x, y) = {N @ xy_e}")
print(f"\nN = {N}\n  suma N = {N.sum():.6f}   (debe ser 1)")
print(f"\ndN (fila 0: dN/dxi, fila 1: dN/deta) =\n{dN}\n  suma por fila = {dN.sum(axis=1)}   (debe ser 0)")
print(f"\nJ = dN @ xy_e =\n{J}\n  det J = {detJ:.6g}")
print(f"\ndN_dx = J^-1 dN (fila 0: dN/dx, fila 1: dN/dy) =\n{dN_dx}")
print(f"\nB ({B.shape[0]}x{B.shape[1]}) =\n{B}")
print(f"\nD [GPa] =\n{D/1e9}")
print("\nCasos (eps = B u_e en el punto):")
for nombre, u, eps in casos:
    u_e = np.column_stack(u(xy_e[:, 0], xy_e[:, 1])).ravel()
    print(f"  {nombre:20s} B u = {np.round(B @ u_e, 12) + 0.0}   exacta = {np.array(eps, dtype=float)}")


# --- Figuras ----------------------------------------------------------------------------------
graficar_funciones_forma(tipo)
graficar_mapeo(tipo, xy_e, punto)
graficar_deformacion(tipo, xy_e, casos, punto)
graficar_matrices(tipo, xy_e, punto, D)
plt.show()
