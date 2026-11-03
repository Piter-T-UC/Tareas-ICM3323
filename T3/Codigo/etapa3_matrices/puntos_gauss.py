import numpy as np


## Gauss-Legendre 1D en [-1, 1] con n puntos: (puntos, pesos)
## Con n puntos la regla integra exacto cualquier polinomio de grado <= 2n - 1.
## Los pesos de cada regla suman 2 (el largo de [-1, 1]).
def gauss_1d(n):
    if n == 1:      # exacta hasta grado 1
        puntos = [0.0]
        pesos  = [2.0]
    elif n == 2:    # exacta hasta grado 3
        a = 1/np.sqrt(3)                        # 0.577350
        puntos = [-a, a]
        pesos  = [1.0, 1.0]
    elif n == 3:    # exacta hasta grado 5
        a = np.sqrt(3/5)                        # 0.774597
        puntos = [-a, 0.0, a]
        pesos  = [5/9, 8/9, 5/9]
    elif n == 4:    # exacta hasta grado 7
        a = np.sqrt(3/7 - 2/7*np.sqrt(6/5))     # 0.339981
        b = np.sqrt(3/7 + 2/7*np.sqrt(6/5))     # 0.861136
        wa = (18 + np.sqrt(30))/36              # 0.652145
        wb = (18 - np.sqrt(30))/36              # 0.347855
        puntos = [-b, -a, a, b]
        pesos  = [wb, wa, wa, wb]
    else:
        raise ValueError(f"gauss_1d: n = {n} no esta definido (opciones: 1, 2, 3, 4)")
    return np.array(puntos), np.array(pesos)


## Gauss-Legendre 1D en [0, 1] (para integrar sobre los lados, s en [0, 1])
## Cambio de variable s = (r + 1)/2, ds = dr/2: los puntos se trasladan y los pesos se dividen por 2
def gauss_01(n):
    r, w = gauss_1d(n)
    s = (r + 1)/2
    return s, w/2


## Puntos (m, 2) y pesos (m,) de integracion en el elemento de referencia
## Cada fila de puntos es [xi, eta] y pesos[i] es el peso del punto i:
##   int f dxi deta  =  sum_i  pesos[i] * f(puntos[i])
def puntos_gauss(tipo):
    # Triangulos (xi = L2, eta = L3, area de referencia 1/2): los pesos suman 1/2
    if tipo == "T3":                # 1 punto en el centroide (B es constante)
        puntos = [[1/3, 1/3]]
        pesos  = [1/2]
        return np.array(puntos), np.array(pesos)
    if tipo == "T6":                # 3 puntos, exacta hasta grado 2
        puntos = [[1/6, 1/6],
                  [2/3, 1/6],
                  [1/6, 2/3]]
        pesos  = [1/6, 1/6, 1/6]
        return np.array(puntos), np.array(pesos)

    # Cuadrilateros: regla n x n en [-1, 1] x [-1, 1] (los pesos suman 4)
    # Q4 -> 2 x 2 = 4 puntos, Q8 y Q9 -> 3 x 3 = 9 puntos
    if tipo == "Q4":
        n = 2
    elif tipo in ("Q8", "Q9"):
        n = 3
    else:
        raise ValueError(f"puntos_gauss: tipo '{tipo}' no existe (opciones: T3, T6, Q4, Q8, Q9)")

    # Se toman todas las combinaciones de un punto 1D en xi con un punto 1D en eta.
    # El peso de cada combinacion es el producto de los dos pesos 1D.
    # Orden: i = n*a + b  (para Q9: i = 0 es (-0.77, -0.77), i = 1 es (-0.77, 0), ...)
    s, w = gauss_1d(n)
    puntos = []
    pesos = []
    for a in range(n):              # punto en xi
        for b in range(n):          # punto en eta
            puntos.append([s[a], s[b]])
            pesos.append(w[a]*w[b])
    return np.array(puntos), np.array(pesos)
