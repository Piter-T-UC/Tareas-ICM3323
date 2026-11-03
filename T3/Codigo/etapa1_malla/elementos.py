import numpy as np


## Catalogo de elementos 2D.0
ELEMENTOS = {
    "T3": dict(n_nodos=3, esquinas=3, contorno=[0, 1, 2]),
    "T6": dict(n_nodos=6, esquinas=3, contorno=[0, 3, 1, 4, 2, 5]),
    "Q4": dict(n_nodos=4, esquinas=4, contorno=[0, 1, 2, 3]),
    "Q8": dict(n_nodos=8, esquinas=4, contorno=[0, 4, 1, 5, 2, 6, 3, 7]),
    "Q9": dict(n_nodos=9, esquinas=4, contorno=[0, 4, 1, 5, 2, 6, 3, 7]),
}

## Cada tipo tiene un numero de nodos distinto: el largo de la fila de conectividad dice el tipo
TIPO_POR_NODOS = {info["n_nodos"]: tipo for tipo, info in ELEMENTOS.items()}
#me dice si tengo errores en la enumeracion de los nodos
def tipo_elemento(nodos):
    if len(nodos) not in TIPO_POR_NODOS:
        raise ValueError(f"elemento con {len(nodos)} nodos {list(nodos)}: no corresponde a ningun tipo "
                         f"(3 T3, 6 T6, 4 Q4, 8 Q8, 9 Q9)")
    return TIPO_POR_NODOS[len(nodos)]

 
## Lados del elemento como listas de nodos locales en sentido antihorario:
## lineales [esquina, esquina], cuadraticos [esquina, medio, esquina]
def lados(tipo):
    info = ELEMENTOS[tipo]
    c = info["contorno"]
    if info["n_nodos"] == info["esquinas"]:
        return [[c[k], c[(k + 1) % len(c)]] for k in range(len(c))]
    return [[c[k], c[k + 1], c[(k + 2) % len(c)]] for k in range(0, len(c), 2)]


## Busca el lado con extremos n1 y n2 (esquinas). Devuelve el elemento e y los nodos del lado
## de n1 a n2: [n1, n2] si es lineal, [n1, n_medio, n2] si es cuadratico (el medio lo pone solo)
def buscar_lado(conectividad, n1, n2):
    for e, nodos in enumerate(conectividad):
        for lado in lados(tipo_elemento(nodos)):
            globales = [int(nodos[k]) for k in lado]
            if (globales[0], globales[-1]) == (n1, n2):
                return e, globales
            if (globales[-1], globales[0]) == (n1, n2):
                return e, globales[::-1]
    raise ValueError(f"los nodos {n1} y {n2} no son los extremos de un lado de ningun elemento")


## Lados de borde (pertenecen a un solo elemento) como listas de nodos globales,
## sentido antihorario respecto a su elemento
def lados_borde(conectividad):
    todos = [[int(nodos[k]) for k in lado] for nodos in conectividad
             for lado in lados(tipo_elemento(nodos))]
    cuenta = {}
    for lado in todos:
        cuenta[frozenset(lado)] = cuenta.get(frozenset(lado), 0) + 1
    return [lado for lado in todos if cuenta[frozenset(lado)] == 1]


## Area con signo del poligono (formula del cordon); > 0 si es antihorario
def area_signo(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5*(x @ np.roll(y, -1) - y @ np.roll(x, -1))


## Revisa numero de nodos, indices validos y orientacion antihoraria de cada elemento
## conectividad: una fila de nodos por elemento (np.array si todos son del mismo tipo,
## lista de listas si se mezclan tipos)
def validar_malla(xy, conectividad):
    xy_m = np.asarray(getattr(xy, "magnitude", xy), dtype=float)
    n_nodos = len(xy_m)
    for e, nodos in enumerate(conectividad):
        try:
            tipo = tipo_elemento(nodos)
        except ValueError as err:
            raise ValueError(f"elemento {e}: {err}") from None
        info = ELEMENTOS[tipo]
        if min(nodos) < 0 or max(nodos) >= n_nodos:
            raise ValueError(f"elemento {e} ({tipo}): nodo fuera de rango en {list(nodos)} (hay {n_nodos} nodos)")
        if len(set(nodos)) != len(nodos):
            raise ValueError(f"elemento {e} ({tipo}): nodos repetidos en {list(nodos)}")
        area = area_signo(xy_m[list(nodos)[:info["esquinas"]]])
        if area <= 0:
            raise ValueError(f"elemento {e} ({tipo}): esquinas en sentido horario o area nula "
                             f"(A = {area:.3g}); ordenar {list(nodos)} antihorario")
