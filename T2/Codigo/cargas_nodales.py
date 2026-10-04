import numpy as np

from unidades import ureg


##Aqui paso una fuerza uniforme a todos los elementos, podria hacer simplemente sea una f quese vaya
##sumando para asi poder hacerlo para no todos los elementos pero aprovechando que solo trabajo con peso aprovecho, de todos modos es mas o menos lo mismo si solo quiero aplicarla en algunos nodos
-
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
