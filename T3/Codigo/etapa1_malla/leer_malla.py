import numpy as np


## Lineas con datos de un archivo de texto: [(numero_de_linea, [columnas]), ...]
## Ignora las lineas vacias y todo lo que va despues de '#'
def _lineas_con_datos(archivo):
    filas = []
    with open(archivo, encoding="utf-8") as f:
        for k, linea in enumerate(f, 1):
            datos = linea.split("#")[0].split()
            if datos:
                filas.append((k, datos))
    return filas


## Filas de un archivo con una linea por nodo o elemento. La primera columna es el numero
## (0, 1, 2, ... en orden) y el resto son los datos de esa fila.
def _leer_filas(archivo):
    filas = _lineas_con_datos(archivo)
    for i, (k, datos) in enumerate(filas):
        if int(datos[0]) != i:
            raise ValueError(f"{archivo}, linea {k}: se esperaba el numero {i} y dice {datos[0]} "
                             f"(los numeros van en orden desde 0)")
    return [(k, datos[1:]) for k, datos in filas]


## Nodos: una linea "nodo  x  y" por nodo -> (n_nodos, 2) floats, en las unidades del archivo
def leer_nodos(archivo):
    filas = _leer_filas(archivo)
    for k, datos in filas:
        if len(datos) != 2:
            raise ValueError(f"{archivo}, linea {k}: un nodo lleva 'nodo x y' y tiene {len(datos) + 1} columnas")
    return np.array([datos for _, datos in filas], dtype=float)


## Conectividad: una linea "elem  n0  n1  ..." por elemento, nodos en sentido antihorario.
## El tipo sale del numero de nodos (3 T3, 6 T6, 4 Q4, 8 Q8, 9 Q9).
## Devuelve np.array (n_elem, n_nodos) si todos tienen los mismos nodos; si no, lista de listas
def leer_conectividad(archivo):
    filas = [[int(n) for n in datos] for _, datos in _leer_filas(archivo)]
    if len({len(nodos) for nodos in filas}) == 1:
        return np.array(filas, dtype=int)
    return filas


## Apoyos: una linea "nodo  rx  ry" por nodo apoyado (1 = restringido, 0 = libre)
## -> (n_apoyos, 3) int, el mismo arreglo apoyos de siempre
def leer_apoyos(archivo):
    filas = _lineas_con_datos(archivo)
    for k, datos in filas:
        if len(datos) != 3 or any(r not in ("0", "1") for r in datos[1:]):
            raise ValueError(f"{archivo}, linea {k}: un apoyo lleva 'nodo rx ry' con rx, ry = 0 o 1")
    return np.array([datos for _, datos in filas], dtype=int)
