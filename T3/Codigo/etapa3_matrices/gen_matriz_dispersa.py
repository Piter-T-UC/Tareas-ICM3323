import numpy as np
from scipy.sparse import coo_matrix

from unidades import ureg
from etapa3_matrices.gen_matriz_global import k_elemento, dofs_elemento


## Igual que gen_matriz_global pero con K dispersa (CSC), para mallas de decenas de miles de GDL
def gen_matriz_dispersa(xy, conectividad, D, espesor):
    xy_m = xy.to(ureg.meter).magnitude
    t = espesor.to(ureg.meter).magnitude
    filas, columnas, valores = [], [], []
    for nodos in conectividad:
        K_e = k_elemento(xy_m[list(nodos)], D, t)
        dofs = np.array(dofs_elemento(nodos))
        filas.append(np.repeat(dofs, len(dofs)))
        columnas.append(np.tile(dofs, len(dofs)))
        valores.append(K_e.ravel())
    n_lib = 2*len(xy_m)
    # coo_matrix suma las entradas repetidas: eso es el ensamble
    ij = (np.concatenate(filas), np.concatenate(columnas))
    return coo_matrix((np.concatenate(valores), ij), shape=(n_lib, n_lib)).tocsc()
