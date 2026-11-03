import numpy as np

from unidades import ureg
from etapa1_malla.elementos import tipo_elemento
from etapa2_interpolacion.funciones_forma import NODOS_NATURALES
from etapa3_matrices.matriz_B import matriz_B
from etapa3_matrices.gen_matriz_global import dofs_elemento


## Tensiones sigma = D B u_e evaluadas en los nodos de cada elemento
##   sigma = [sxx, syy, txy] [Pa], B en las coordenadas naturales de cada nodo
## Un nodo compartido recibe un valor de cada elemento vecino -> se promedian
## (dentro de un elemento el campo es continuo, entre elementos salta un poco)
## xy: pint, U: vector global (2 n_nodos) [m], D: matriz_D [Pa]
## Devuelve sigma_nodos (n_nodos, 3) promediado y sigma_elem [(n_nodos_e, 3), ...] sin promediar
def calc_tensiones(xy, conectividad, U, D):
    xy_m = xy.to(ureg.meter).magnitude
    suma = np.zeros((len(xy_m), 3))
    cuenta = np.zeros(len(xy_m))
    sigma_elem = []
    for nodos in conectividad:
        tipo = tipo_elemento(nodos)
        xy_e = xy_m[list(nodos)]
        u_e = U[dofs_elemento(nodos)]
        s_e = np.array([D @ matriz_B(tipo, xy_e, xi, eta)[0] @ u_e for xi, eta in NODOS_NATURALES[tipo]])
        sigma_elem.append(s_e)
        suma[list(nodos)] += s_e
        cuenta[list(nodos)] += 1
    return suma/cuenta[:, None], sigma_elem


## Von Mises en tension plana [Pa] a partir de sigma (..., 3) = [sxx, syy, txy]
def von_mises(sigma):
    sxx, syy, txy = sigma[..., 0], sigma[..., 1], sigma[..., 2]
    return np.sqrt(sxx**2 - sxx*syy + syy**2 + 3*txy**2)
