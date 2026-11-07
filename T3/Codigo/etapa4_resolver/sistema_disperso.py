import numpy as np
from scipy.sparse.linalg import spsolve


## Resuelve K U = F con K dispersa y devuelve U completo (0 en los dofs_fijos)
def resolver_disperso(K_global, F_global, dofs_fijos):
    libres = np.setdiff1d(np.arange(K_global.shape[0]), dofs_fijos)
    U = np.zeros(K_global.shape[0])
    U[libres] = spsolve(K_global[libres][:, libres], F_global[libres])
    return U
