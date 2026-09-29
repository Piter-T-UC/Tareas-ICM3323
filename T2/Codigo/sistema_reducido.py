import numpy as np

def Sist_red(K_global, F_global, dofs_fijos):
    K_reducida = np.delete(np.delete(K_global, dofs_fijos, axis=0), dofs_fijos, axis=1)
    F_reducida = np.delete(F_global, dofs_fijos, axis=0)
    U_red = np.linalg.solve(K_reducida, F_reducida)
    return U_red, K_reducida, F_reducida
