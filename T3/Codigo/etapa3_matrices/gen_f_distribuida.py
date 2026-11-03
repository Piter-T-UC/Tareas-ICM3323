import numpy as np

from unidades import ureg
from etapa1_malla.cargas import leer_cargas_dist, puntos_lado, q_lineal
from etapa3_matrices.puntos_gauss import gauss_01


## Funciones de forma de un lado de 2 o 3 nodos en el punto s de [0, 1]: vector (n_nodos,)
## mismo orden que puntos_lado: [extremo, extremo] o [extremo, medio, extremo]
def _N_lado(n_nodos, s):
    if n_nodos == 2:
        return np.array([1 - s, s])
    return np.array([(1 - s)*(1 - 2*s), 4*s*(1 - s), s*(2*s - 1)])


## Fuerzas nodales equivalentes de un lado (n_nodos_lado, 2) [N]: una fila [fx, fy] por nodo
##   f_i = int_L N_i q dL
## Se pasa al lado de referencia (s en [0, 1]) con dL = |dx/ds| ds y la integral se calcula
## con cuadratura de Gauss: suma sobre los puntos de Gauss de (peso * integrando)
##   f_i = sum_p  w_p * N_i(s_p) * q(s_p) * |dx/ds|_p
## q1, q2: [qx, qy] en los extremos del lado [N/m] (ya multiplicados por el espesor)
## n_gauss = 4 es exacto en lados rectos (N grado 2 * q grado 1 = grado 3); los curvos dan aprox.
def f_eq_lado(xy_m, nodos_lado, q1, q2, n_gauss=4):
    n_nodos = len(nodos_lado)
    f_lado = np.zeros((n_nodos, 2))

    # Puntos de Gauss del lado de referencia: pts (m,) con s en [0, 1] y pesos (m,)
    pts, pesos = gauss_01(n_gauss)

    for i in range(len(pesos)):
        s = pts[i]
        w = pesos[i]

        # N, q y |dx/ds| evaluados en el punto de Gauss i
        N = _N_lado(n_nodos, s)                     # (n_nodos,)
        q = q_lineal(q1, q2, s)[0]                  # [qx, qy]
        _, dx = puntos_lado(xy_m, nodos_lado, s)
        dL_ds = np.linalg.norm(dx[0])               # |dx/ds| [m]

        # Integrando en ese punto: N_i * [qx, qy] * |dx/ds|, una fila por nodo
        integrando = np.outer(N, q) * dL_ds

        # Aporte del punto i a la integral
        f_lado += w * integrando

    return f_lado


## Vector global (2 n_nodos) [N] de las cargas distribuidas (formato de cargas.leer_cargas_dist)
def gen_f_distribuida(xy, conectividad, cargas_dist, espesor):
    xy_m = xy.to(ureg.meter).magnitude
    F_dist = np.zeros(2*len(xy_m))
    for _, nodos_lado, q1, q2 in leer_cargas_dist(conectividad, cargas_dist, espesor):
        f = f_eq_lado(xy_m, nodos_lado, q1, q2)
        for nodo, (fx, fy) in zip(nodos_lado, f):
            F_dist[2*nodo] += fx
            F_dist[2*nodo + 1] += fy
    return F_dist
