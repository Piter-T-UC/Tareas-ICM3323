import numpy as np



def gen_f_global(xyz, f_nodales):
    dof_por_nodo = 6
    n_lib = len(xyz) * dof_por_nodo
    F_global = np.zeros(n_lib)            
    for entrada in f_nodales:
        nodo = entrada[0]
        cargas = entrada[1:]              # (Fx,Fy,Fz,Mx,My,Mz)
        for k, f in enumerate(cargas):
            F_global[dof_por_nodo*nodo + k] += f.to_base_units().magnitude
    return F_global
