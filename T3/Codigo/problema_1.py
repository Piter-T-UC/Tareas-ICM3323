import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
# 1. malla
from etapa1_malla.elementos import validar_malla
from etapa1_malla.cargas import leer_cargas_dist, lados_desde_nodos
# 3. matrices
from etapa3_matrices.matriz_D import matriz_D
# 5. graficar
from etapa5_graficar.graficar_malla import graficar_malla

kN, MPa = ureg.kN, ureg.MPa


titulo = "Malla de prueba (todos los tipos de elemento)"

# Material y seccion
E = 200 * ureg.GPa
nu = 0.3
espesor = 10 * ureg.mm
estado = "tension_plana"        # o "deformacion_plana"

# Nodos
xy = np.array([
    [0, 0], [1, 0], [2, 0], [3, 0], [4, 0],     # 0-4   borde inferior
    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1],     # 5-9   borde superior
    [2.5, 0],    # 10 medios del Q8
    [3, 0.5],    # 11 (compartido con el T6 de arriba)
    [2.5, 1],    # 12
    [2, 0.5],    # 13 ojo: queda colgando (los T3 vecinos son lineales)
    [3.5, 0],    # 14 medios de los T6
    [4, 0.5],    # 15 (compartido con el Q9)
    [3.5, 1],    # 16
    [3.5, 0.5],  # 17 diagonal 3-9
    [5, 0], [5, 1],   # 18-19 esquinas del Q9
    [4.5, 0],    # 20 medios del Q9
    [5, 0.5],    # 21
    [4.5, 1],    # 22
    [4.5, 0.5],  # 23 centro del Q9
]) * ureg.meter

# Conectividad: una fila de nodos por elemento -> antihorario, esquinas primero, luego medios,
# luego centro. El tipo sale del numero de nodos: 3 T3, 6 T6, 4 Q4, 8 Q8, 9 Q9
# (si todos son del mismo tipo puede ser np.array(..., dtype=int); aqui se mezclan -> lista)
conectividad = [
    [0, 1, 6, 5],                       # 0  Q4
    [1, 2, 7],                          # 1  T3
    [1, 7, 6],                          # 2  T3
    [2, 3, 8, 7, 10, 11, 12, 13],       # 3  Q8
    [3, 4, 9, 14, 15, 17],              # 4  T6
    [3, 9, 8, 17, 16, 11],              # 5  T6
    [4, 18, 19, 9, 20, 21, 22, 15, 23], # 6  Q9
]

# Apoyos: [nodo, restringe_x, restringe_y]
apoyos = np.array([
    [0, 1, 1],    # fijo
    [5, 1, 0],    # rodillo (solo x)
], dtype=int)

# Fuerzas nodales: (nodo, Fx, Fy) en ejes globales
fuerzas = [
    (21, 0*kN, -20*kN),
    (19, 5*kN, 0*kN),
]

# Cargas distribuidas (tracciones) como presion; el codigo multiplica por el espesor
# a) por lado, como en T2: (n1, n2, tx, ty) constante o (n1, n2, tx1, ty1, tx2, ty2) lineal de n1 a n2
#    n1, n2 = esquinas del lado (en lados cuadraticos el nodo medio se busca solo)
cargas_dist = [
    (7, 6,  0*MPa, -1*MPa),                        # T3, constante
    (8, 7,  0*MPa, -1*MPa,  0*MPa, -2*MPa),        # Q8, lineal (medio 12)
]
# b) por nodo: (nodo, tx, ty); se cargan los lados de borde con sus dos esquinas en la lista
tracciones = [
    (9, 0*MPa, -2*MPa),
    (8, 0*MPa, -1*MPa),
]
cargas_dist += lados_desde_nodos(conectividad, tracciones)     # T6, lineal (medio 16)


validar_malla(xy, conectividad)
leer_cargas_dist(conectividad, cargas_dist, espesor)     # revisa que cada carga este sobre un lado
print(f"{titulo}\n{len(xy)} nodos, {len(conectividad)} elementos, {2*len(xy)} GDL")
print(f"E = {E}, nu = {nu}, t = {espesor}, estado: {estado}")

D = matriz_D(E, nu, estado)
print(f"D [GPa] =\n{np.round(D/1e9, 3)}")

fig, ax = graficar_malla(xy, conectividad, apoyos, fuerzas, cargas_dist, espesor, titulo=titulo)
fig.savefig("malla_problema_1.png", dpi=150)
plt.show()
