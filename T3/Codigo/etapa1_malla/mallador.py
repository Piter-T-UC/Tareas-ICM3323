import numpy as np

from etapa1_malla.elementos import ELEMENTOS, lados, tipo_elemento
from etapa1_malla.cargas import puntos_lado


## Mallador por bloques. Toda la geometria va en floats SI (metros); el problema le pone
## las unidades despues (xy * ureg.meter).
## Una region se arma con bloques de 4 lados (rectos o curvos): cada bloque se malla
## estructurado con interpolacion transfinita (Coons) y despues se juntan con unir().


## Lado recto de p0 a p1 como funcion f(s) -> (n_pts, 2), s en [0, 1]
def recta(p0, p1):
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    return lambda s: p0 + np.atleast_1d(s)[:, None]*(p1 - p0)


## Arco de circunferencia de th0 a th1 [rad] como funcion f(s), s en [0, 1]
def arco(centro, r, th0, th1):
    centro = np.asarray(centro, dtype=float)
    def f(s):
        th = th0 + np.atleast_1d(s)*(th1 - th0)
        return centro + r*np.column_stack([np.cos(th), np.sin(th)])
    return f


## Parametros en [0, 1] de los nodos en una direccion: n elementos, graduacion geometrica
## razon = tamano del ultimo elemento / el primero; p = 2 agrega los nodos medios
def _parametros(n, razon, p):
    q = razon**(1/(n - 1)) if n > 1 else 1.0
    t = np.r_[0, np.cumsum(q**np.arange(n))]
    t = t / t[-1]
    if p == 2:
        t = np.r_[np.c_[t[:-1], (t[:-1] + t[1:])/2].ravel(), 1.0]
    return t


## Conectividad como matriz (n_elem, n_nodos) si todos los elementos tienen los mismos nodos;
## si se mezclan tipos queda como lista de listas (cada fila con su largo)
def _como_matriz(filas):
    filas = [[int(n) for n in nodos] for nodos in filas]
    if len({len(nodos) for nodos in filas}) == 1:
        return np.array(filas, dtype=int)
    return filas


## Deja solo los nodos usados por algun elemento y renumera
def _compactar(xy, conectividad):
    usados = sorted({n for nodos in conectividad for n in nodos})
    nuevo = {n: k for k, n in enumerate(usados)}
    return xy[usados], _como_matriz([[nuevo[n] for n in nodos] for nodos in conectividad])


## Malla estructurada de un bloque de 4 lados
## esquinas: 4 puntos en sentido antihorario c0, c1, c2, c3
## nx: elementos en la direccion c0 -> c1 (xi), ny: elementos en la direccion c0 -> c3 (eta)
## lados: dict opcional {k: f(s)} para lados curvos (los que falten son rectos), recorridos asi:
##   0: c0 -> c1    1: c1 -> c2    2: c3 -> c2    3: c0 -> c3
## razon = (rx, ry): ultimo elemento / primero en cada direccion (1 = uniforme)
## Devuelve xy (n, 2) y conectividad (n_elem, n_nodos): una fila de nodos por elemento
def bloque(esquinas, nx, ny, tipo="Q4", lados=None, razon=(1, 1)):
    if tipo not in ELEMENTOS:
        raise ValueError(f"tipo '{tipo}' no existe (opciones: {list(ELEMENTOS)})")
    c = np.asarray(esquinas, dtype=float)
    f = {0: recta(c[0], c[1]), 1: recta(c[1], c[2]), 2: recta(c[3], c[2]), 3: recta(c[0], c[3])}
    f.update(lados or {})

    p = 1 if ELEMENTOS[tipo]["n_nodos"] == ELEMENTOS[tipo]["esquinas"] else 2
    xi, eta = _parametros(nx, razon[0], p), _parametros(ny, razon[1], p)
    X, Y = np.meshgrid(xi, eta, indexing="ij")
    x, y = X.ravel(), Y.ravel()
    u, v = x[:, None], y[:, None]
    xy = ((1 - v)*f[0](x) + v*f[2](x) + (1 - u)*f[3](y) + u*f[1](y)
          - (1 - u)*(1 - v)*c[0] - u*(1 - v)*c[1] - u*v*c[2] - (1 - u)*v*c[3])
    id_ = np.arange(len(xy)).reshape(X.shape)

    conectividad = []
    for i in range(0, p*nx, p):
        for j in range(0, p*ny, p):
            n = lambda di, dj: int(id_[i + di, j + dj])
            # esquinas de la celda (antihorario) y, si es cuadratica, medios y centro
            e = [n(0, 0), n(p, 0), n(p, p), n(0, p)]
            if tipo == "Q4":
                conectividad.append(e)
            elif tipo == "T3":
                conectividad += [[e[0], e[1], e[2]], [e[0], e[2], e[3]]]
            else:
                m = [n(1, 0), n(2, 1), n(1, 2), n(0, 1)]
                centro = n(1, 1)
                if tipo == "Q8":
                    conectividad.append(e + m)
                elif tipo == "Q9":
                    conectividad.append(e + m + [centro])
                else:                  # T6: el centro de la celda es el medio de la diagonal 0-2
                    conectividad += [[e[0], e[1], e[2], m[0], m[1], centro],
                                     [e[0], e[2], e[3], centro, m[2], m[3]]]
    return _compactar(xy, conectividad)


## Junta varias mallas (xy, conectividad) fundiendo los nodos con la misma coordenada.
## Las caras compartidas deben calzar (mismas divisiones y graduacion a ambos lados).
## Numera los nodos en el orden en que aparecen.
def unir(*mallas, tol=1e-9):
    xy = np.vstack([m[0] for m in mallas])
    offsets = np.cumsum([0] + [len(m[0]) for m in mallas])
    conectividad = [[n + off for n in nodos]
                    for (_, conec), off in zip(mallas, offsets) for nodos in conec]

    paso = tol*max(np.ptp(xy, axis=0).max(), 1e-300)
    claves = np.round(xy/paso).astype(np.int64)
    _, primero, inversa = np.unique(claves, axis=0, return_index=True, return_inverse=True)
    orden = np.argsort(primero)
    rango = np.empty_like(orden)
    rango[orden] = np.arange(len(orden))
    nuevo = rango[inversa.ravel()]
    return xy[primero[orden]], _como_matriz([[nuevo[n] for n in nodos] for nodos in conectividad])


## ---------------------------------------------------------------------------------------------
## Seleccion por condicion geometrica en(x, y) -> bool, con x, y en metros.
## Ej: en = lambda x, y: np.isclose(y, 0)
## ---------------------------------------------------------------------------------------------

def _sin_unidades(xy):
    return np.asarray(xy.to("meter").magnitude if hasattr(xy, "magnitude") else xy, dtype=float)


## Nodos que cumplen en(x, y)
def nodos_en(xy, en):
    xy_m = _sin_unidades(xy)
    return [n for n, (x, y) in enumerate(xy_m) if en(x, y)]


## Lados de borde (de un solo elemento) con todos sus nodos cumpliendo en(x, y),
## como listas de nodos globales en sentido antihorario respecto a su elemento
def lados_en(xy, conectividad, en):
    xy_m = _sin_unidades(xy)
    todos = [[int(nodos[k]) for k in lado] for nodos in conectividad
             for lado in lados(tipo_elemento(nodos))]
    cuenta = {}
    for lado in todos:
        cuenta[frozenset(lado)] = cuenta.get(frozenset(lado), 0) + 1
    return [lado for lado in todos
            if cuenta[frozenset(lado)] == 1 and all(en(*xy_m[n]) for n in lado)]


## Filas [nodo, rx, ry] para el arreglo apoyos (1 = restringido)
def apoyos_en(xy, en, rx, ry):
    return [[n, rx, ry] for n in nodos_en(xy, en)]


## Entradas de cargas_dist (n1, n2, qx1, qy1, qx2, qy2) en los lados de borde que cumplen
## en(x, y). q (por metro de lado, ya multiplicado por el espesor) puede ser:
##   constante pint [qx, qy]                  -> q1 = q2 = q
##   funcion q(x, y, nx, ny) -> pint [qx, qy] -> evaluada en los extremos n1 y n2 del lado con
##                                              su normal exterior (lineal entre ellos)
def cargas_en(xy, conectividad, en, q):
    xy_m = _sin_unidades(xy)
    salida = []
    for lado in lados_en(xy_m, conectividad, en):
        if not callable(q):
            q1 = q2 = q
        else:
            pts, dx = puntos_lado(xy_m, lado, [0.0, 1.0])
            n = np.column_stack([dx[:, 1], -dx[:, 0]]) / np.linalg.norm(dx, axis=1, keepdims=True)
            q1, q2 = q(*pts[0], *n[0]), q(*pts[1], *n[1])
        salida.append((lado[0], lado[-1], q1[0], q1[1], q2[0], q2[1]))
    return salida
