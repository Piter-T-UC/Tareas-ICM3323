import numpy as np

from etapa2_interpolacion.funciones_forma import funciones_forma


## Jacobiano de la transformacion (xi, eta) -> (x, y) de un elemento isoparametrico
## dN: (2, n) derivadas naturales (funciones_forma), xy_e: (n, 2) coordenadas de los nodos [m]
##   J = [[dx/dxi,  dy/dxi ],
##        [dx/deta, dy/deta]]  = dN @ xy_e
## Devuelve J, det(J) y dN_dx (2, n): fila 0 = dN/dx, fila 1 = dN/dy  (dN_dx = J^-1 dN)
def jacobiano(dN, xy_e):
    J = dN @ xy_e
    detJ = np.linalg.det(J)
    if detJ <= 0:
        raise ValueError(f"det(J) = {detJ:.3g} <= 0: elemento invertido o muy distorsionado "
                         f"(revisar orden antihorario y nodos medios)")
    return J, detJ, np.linalg.solve(J, dN)


## Matriz B (3, 2n) en (xi, eta): eps = B u_e, con eps = [exx, eyy, gxy] y
## u_e = [u0x, u0y, u1x, u1y, ...] (2 GDL por nodo, mismo orden que los nodos del elemento)
## xy_e: (n, 2) coordenadas de los nodos del elemento [m]
## Devuelve B, det(J) y N (para integrar con Gauss: dA = det(J) dxi deta)
def matriz_B(tipo, xy_e, xi, eta):
    N, dN = funciones_forma(tipo, xi, eta)
    _, detJ, dN_dx = jacobiano(dN, np.asarray(xy_e, dtype=float))
    B = np.zeros((3, 2*len(N)))
    B[0, 0::2] = dN_dx[0]       # exx = du/dx
    B[1, 1::2] = dN_dx[1]       # eyy = dv/dy
    B[2, 0::2] = dN_dx[1]       # gxy = du/dy + dv/dx
    B[2, 1::2] = dN_dx[0]
    return B, detJ, N
