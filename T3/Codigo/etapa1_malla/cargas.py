import numpy as np

from unidades import ureg
from etapa1_malla.elementos import buscar_lado, lados_borde


## Posicion x(s) y tangente dx/ds sobre un lado de 2 o 3 nodos, s en [0, 1]
## (Lagrange lineal o cuadratico: sirve tambien para lados curvos)
def puntos_lado(xy_m, nodos_lado, s):
    s = np.atleast_1d(s)[:, None]
    p = xy_m[list(nodos_lado)]
    if len(p) == 2:
        return (1 - s)*p[0] + s*p[1], np.broadcast_to(p[1] - p[0], (len(s), 2))
    x = (1 - s)*(1 - 2*s)*p[0] + 4*s*(1 - s)*p[1] + s*(2*s - 1)*p[2]
    dx = (4*s - 3)*p[0] + (4 - 8*s)*p[1] + (4*s - 1)*p[2]
    return x, dx


## Pasa a N/m una traccion: si viene como presion (MPa, kPa, ...) se multiplica por el espesor,
## si ya viene por metro de lado (kN/m, ...) se usa tal cual
def _a_N_m(t, espesor):
    if t.check("[pressure]"):
        t = t*espesor
    return t.to(ureg.newton/ureg.meter).magnitude


## Pasa cargas_dist a floats SI y busca el lado de cada carga en la malla.
## Formato de entrada, como en T2:
##   (n1, n2, tx, ty)                  constante en el lado
##   (n1, n2, tx1, ty1, tx2, ty2)      lineal de n1 a n2
##   n1, n2: nodos extremos (esquinas) del lado; si es cuadratico el nodo medio se busca solo
##   t en ejes globales, como presion (MPa, kPa, ...: se multiplica por el espesor)
##   o por metro de lado (kN/m, ...)
## Devuelve [(e, nodos_lado, q1, q2), ...] con q en N/m y nodos_lado de n1 a n2
def leer_cargas_dist(conectividad, cargas_dist, espesor):
    salida = []
    for carga in cargas_dist:
        n1, n2, *valores = carga
        if len(valores) == 2:                        # constante
            valores = valores * 2
        if len(valores) != 4:
            raise ValueError(f"carga {carga}: el formato es (n1, n2, tx, ty) o (n1, n2, tx1, ty1, tx2, ty2)")
        q = [_a_N_m(v, espesor) for v in valores]
        e, nodos_lado = buscar_lado(conectividad, n1, n2)
        salida.append((e, nodos_lado, np.array(q[:2], dtype=float), np.array(q[2:], dtype=float)))
    return salida


## Cargas por nodo -> formato de cargas_dist (por lado)
## tracciones = [(nodo, tx, ty), ...]: valor de la traccion en cada nodo del borde cargado.
## Se carga cada lado de borde que tenga sus dos esquinas en la lista (lineal entre ellas;
## los nodos medios se pueden poner pero no se usan)
## Ojo: en una esquina donde la traccion cambia de un borde a otro, hacer una lista por borde
def lados_desde_nodos(conectividad, tracciones):
    t = {nodo: (tx, ty) for nodo, tx, ty in tracciones}
    salida = []
    for lado in lados_borde(conectividad):
        n1, n2 = lado[0], lado[-1]
        if n1 in t and n2 in t:
            salida.append((n1, n2, *t[n1], *t[n2]))
    if not salida:
        raise ValueError(f"ningun lado de borde tiene sus dos esquinas en {sorted(t)}")
    return salida


## q = [qx, qy] en los puntos s del lado (n_pts, 2): lineal de q1 (s = 0) a q2 (s = 1)
def q_lineal(q1, q2, s):
    s = np.atleast_1d(s)[:, None]
    return (1 - s)*q1 + s*q2
