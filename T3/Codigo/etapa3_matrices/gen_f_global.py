import numpy as np

from unidades import ureg


## Vector de fuerzas nodales (2 n_nodos) en N
## f_nodales = [(nodo, Fx, Fy), ...] (pint, ejes globales)
def gen_f_global(xy, f_nodales):
    dof_por_nodo = 2
    n_lib = len(xy) * dof_por_nodo
    F_global = np.zeros(n_lib)
    for entrada in f_nodales:
        nodo = entrada[0]
        cargas = entrada[1:]              # (Fx, Fy)
        for k, f in enumerate(cargas):
            F_global[dof_por_nodo*nodo + k] += f.to_base_units().magnitude
    return F_global


## Al reves: vector global F (2 n_nodos) [N] -> [(nodo, Fx, Fy), ...] pint, solo nodos cargados
## (sirve para graficar cuanto se lleva cada nodo de una carga distribuida)
def fuerzas_desde_vector(F, tol=1e-9):
    F_n = F.reshape(-1, 2)
    tope = max(np.abs(F_n).max(), 1e-300)
    return [(n, fx*ureg.N, fy*ureg.N) for n, (fx, fy) in enumerate(F_n)
            if np.hypot(fx, fy) > tol*tope]
