import numpy as np

from unidades import ureg


## Peso propio como cargas nodales equivalentes (formato de gen_f_global).
## Por elemento, con d = x2 - x1, L = |d| y q = (0, 0, -peso_lineal):
##   fuerzas:  F1 = F2 = q L / 2                 (verticales)
##   momentos: M1 = (L/12) (d x q),  M2 = -M1    (modulo q_perp L^2 / 12, eje horizontal
##                                                 perpendicular a la barra)
## Se suman por nodo; en nodos interiores de una barra partida en tramos iguales los
## momentos de los dos elementos vecinos se cancelan y solo queda la fuerza.
def cargas_nodales_equivalentes(xyz, conectividad, peso_lineal):
    xyz_m = xyz.to(ureg.meter).magnitude
    q = np.array([0.0, 0.0, -peso_lineal.to(ureg.N/ureg.m).magnitude])
    cargas = np.zeros((len(xyz_m), 6))              # [Fx, Fy, Fz, Mx, My, Mz] por nodo (N, N*m)
    for n1, n2 in conectividad:
        d = xyz_m[n2] - xyz_m[n1]
        L = np.linalg.norm(d)
        F = q*L/2
        M = L/12*np.cross(d, q)
        cargas[n1] += np.concatenate([F, M])
        cargas[n2] += np.concatenate([F, -M])

    N, Nm = ureg.N, ureg.N*ureg.meter
    return [(i, *(c[:3]*N), *(c[3:]*Nm)) for i, c in enumerate(cargas) if np.any(c)]
