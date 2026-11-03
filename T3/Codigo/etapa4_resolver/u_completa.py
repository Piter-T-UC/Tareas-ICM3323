import numpy as np


def U_completa(dofs_fijos, U_red):
    n_lib = len(U_red) + len(dofs_fijos)
    gl_libres = [d for d in range(n_lib) if d not in dofs_fijos]
    U = np.zeros(n_lib)
    U[gl_libres] = U_red
    return U
