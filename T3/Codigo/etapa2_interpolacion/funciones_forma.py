import numpy as np


## Coordenadas naturales (xi, eta) de los nodos locales, misma numeracion que elementos.py
## (antihorario, esquinas -> medios -> centro).
## Triangulos: xi = L2, eta = L3, L1 = 1 - xi - eta (coordenadas de area), nodos en (0,0) (1,0) (0,1)
## Cuadrilateros: cuadrado [-1, 1] x [-1, 1]
NODOS_NATURALES = {
    "T3": np.array([[0, 0], [1, 0], [0, 1]], dtype=float),
    "T6": np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]]),
    "Q4": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float),
    "Q8": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0]], dtype=float),
    "Q9": np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1], [-1, 0], [0, 0]],
                   dtype=float),
}


## Lagrange cuadratico 1D con nodos en -1, 0, 1 evaluado en s: (l, dl/ds) para el nodo s_i
def _lagrange_1d(s_i, s):
    if s_i < 0:
        return s*(s - 1)/2, s - 0.5
    if s_i > 0:
        return s*(s + 1)/2, s + 0.5
    return 1 - s**2, -2*s


## Funciones de forma en (xi, eta)
## Devuelve N (n_nodos,) y dN (2, n_nodos): fila 0 = dN/dxi, fila 1 = dN/deta
def funciones_forma(tipo, xi, eta):
    if tipo in ("T3", "T6"):
        L = np.array([1 - xi - eta, xi, eta])
        dL = np.array([[-1, 1, 0],          # dL/dxi
                       [-1, 0, 1]])         # dL/deta
        if tipo == "T3":
            return L, dL.astype(float)
        L1, L2, L3 = L
        N = np.array([L1*(2*L1 - 1), L2*(2*L2 - 1), L3*(2*L3 - 1), 4*L1*L2, 4*L2*L3, 4*L3*L1])
        dN_dL = np.array([[4*L1 - 1, 0,        0],          # filas: nodo, columnas: d/dL1, d/dL2, d/dL3
                          [0,        4*L2 - 1, 0],
                          [0,        0,        4*L3 - 1],
                          [4*L2,     4*L1,     0],
                          [0,        4*L3,     4*L2],
                          [4*L3,     0,        4*L1]])
        return N, dL @ dN_dL.T

    nat = NODOS_NATURALES[tipo]
    xi_i, eta_i = nat[:, 0], nat[:, 1]
    if tipo == "Q4":
        N = (1 + xi_i*xi)*(1 + eta_i*eta)/4
        dN = np.array([xi_i*(1 + eta_i*eta)/4,
                       eta_i*(1 + xi_i*xi)/4])
        return N, dN

    if tipo == "Q8":
        N, dN = np.zeros(8), np.zeros((2, 8))
        for k, (a, b) in enumerate(nat):
            if a != 0 and b != 0:       # esquina
                N[k] = (1 + a*xi)*(1 + b*eta)*(a*xi + b*eta - 1)/4
                dN[:, k] = [a*(1 + b*eta)*(2*a*xi + b*eta)/4,
                            b*(1 + a*xi)*(a*xi + 2*b*eta)/4]
            elif a == 0:                # medio de lado horizontal (eta = b)
                N[k] = (1 - xi**2)*(1 + b*eta)/2
                dN[:, k] = [-xi*(1 + b*eta), b*(1 - xi**2)/2]
            else:                       # medio de lado vertical (xi = a)
                N[k] = (1 + a*xi)*(1 - eta**2)/2
                dN[:, k] = [a*(1 - eta**2)/2, -(1 + a*xi)*eta]
        return N, dN

    if tipo == "Q9":                    # producto de Lagrange cuadraticos en xi y eta
        N, dN = np.zeros(9), np.zeros((2, 9))
        for k, (a, b) in enumerate(nat):
            lx, dlx = _lagrange_1d(a, xi)
            ly, dly = _lagrange_1d(b, eta)
            N[k] = lx*ly
            dN[:, k] = [dlx*ly, lx*dly]
        return N, dN

    raise ValueError(f"tipo '{tipo}' no existe (opciones: {list(NODOS_NATURALES)})")
