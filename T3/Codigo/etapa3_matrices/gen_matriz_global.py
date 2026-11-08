import numpy as np

from unidades import ureg
from etapa1_malla.elementos import tipo_elemento
from etapa3_matrices.matriz_B import matriz_B
from etapa3_matrices.puntos_gauss import puntos_gauss


## GDL globales de un elemento: [2*n0, 2*n0 + 1, 2*n1, 2*n1 + 1, ...] (ux, uy por nodo)
def dofs_elemento(nodos):
    return [2*n + k for n in nodos for k in range(2)]


## Matriz de rigidez del elemento (2n x 2n)
##   K_e = int_A B^T D B t dA
## Se pasa al elemento de referencia (xi, eta) con dA = det(J) dxi deta y la integral se
## calcula con cuadratura de Gauss: suma sobre los puntos de Gauss de (peso * integrando)
##   K_e = sum_i  w_i * B_i^T D B_i * t * det(J_i)
## xy_e: (n, 2) coordenadas de los nodos [m], D: (3, 3) [Pa], t: espesor [m]
def k_elemento(xy_e, D, t):
    tipo = tipo_elemento(xy_e)
    n_nodos = len(xy_e)
    K_e = np.zeros((2*n_nodos, 2*n_nodos))
    pts, pesos = puntos_gauss(tipo)

    for i in range(len(pesos)):
        xi, eta = pts[i]
        w = pesos[i]
        # B y det(J) evaluados en el punto de Gauss i
        B, detJ, _ = matriz_B(tipo, xy_e, xi, eta)

        # Integrando en ese punto: B^T D B t det(J)
        integrando = B.T @ D @ B * t * detJ

        # Aporte del punto i a la integral
        K_e += w * integrando

    return K_e


## Ensambla K_global (2 n_nodos x 2 n_nodos) en floats SI (N/m)
## xy: (n_nodos, 2) pint, conectividad: una fila de nodos por elemento,
## D: matriz_D (Pa), espesor: pint
def gen_matriz_global(xy, conectividad, D, espesor):
    xy_m = xy.to(ureg.meter).magnitude
    t = espesor.to(ureg.meter).magnitude
    n_lib = 2*len(xy_m)
    K_global = np.zeros((n_lib, n_lib))
    for nodos in conectividad:
        K_e = k_elemento(xy_m[list(nodos)], D, t)
        dofs = dofs_elemento(nodos)
        K_global[np.ix_(dofs, dofs)] += K_e
    return K_global
