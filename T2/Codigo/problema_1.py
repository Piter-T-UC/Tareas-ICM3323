import os
import numpy as np

from unidades import ureg
from gen_matriz_global import gen_matriz_global
from gen_f_global import gen_f_global
from restricciones import Restriciones
from sistema_reducido import Sist_red
from u_completa import U_completa
from esfuerzos import calc_esfuerzos, calc_reacciones
from exportar_latex import ecuacion_latex




M_Elasticidad = 210 * ureg.GPa
G_Cortante = 80 * ureg.GPa  # dato del enunciado

I_0 = 2e6 * ureg.mm**4# empotramiento (x = 0)
I_1 = 1e6 * ureg.mm**4  # extremo libre (x = L)
Area = 1000 * ureg.mm**2  # no influye: no hay carga axial

L = 1 * ureg.meter
P = 1.0 * ureg.kN
a_P = 0.5 * ureg.meter
M = 2.0 * ureg.kN * ureg.meter
kNm = ureg.kN * ureg.meter


mallas = [2, 4, 8, 16]
carpeta_datos = "datos_P1"


## Arma y resuelve el voladizo con n_elem elementos de viga de dos nodos
def resolver(n_elem):
    xs = np.linspace(0, L.magnitude, n_elem + 1)
    xyz = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]) * ureg.meter

    conectividad = np.array([[i, i+1] for i in range(n_elem)], dtype=int)
    vec_ref = np.array([[0.0, 0.0, 1.0]] * n_elem)   # barra en X, referencia Z global

    # inercia en cada nodo y matriz (n_elem, 2) con [I_inicio, I_fin] por elemento
    I_nodos = (I_0 + (I_1 - I_0) * xs / L.magnitude).to(ureg.mm**4).magnitude
    I_elem = np.column_stack([I_nodos[:-1], I_nodos[1:]]) * ureg.mm**4
    I_x = I_elem                 
    I_y = I_elem                 
    J_p = 2*I_x                  

    apoyos = np.array([
        [0, 1, 1, 1, 1, 1, 1],   # nodo 0: empotrado
    ], dtype=int)

    nodo_P = int(np.argmin(np.abs(xs - a_P.magnitude)))   # Nodo mas cercano a la mitad
    assert np.isclose(xs[nodo_P], a_P.magnitude), "n_elem debe dejar un nodo en x = a_P"
    nodo_M = n_elem                   # nodo donde se aplica el torsor (extremo libre)
    fuerzas = [
        (nodo_P, 0.0*ureg.kN, 0.0*ureg.kN, -P, 0.0*kNm, 0.0*kNm, 0.0*kNm),
        (nodo_M, 0.0*ureg.kN, 0.0*ureg.kN, 0*ureg.kN, M, 0.0*kNm, 0.0*kNm)
    ]

    K_global = gen_matriz_global(xyz, conectividad, Area, I_x, I_y, J_p, vec_ref,
                                 M_Elasticidad, G_Cortante)
    F_global = gen_f_global(xyz, fuerzas)
    dofs_fijos = Restriciones(apoyos)
    U_red, K_red, F_red = Sist_red(K_global, F_global, dofs_fijos)
    U = U_completa(dofs_fijos, U_red)

    res = calc_esfuerzos(xyz, conectividad, U, Area, I_x, I_y, J_p, vec_ref,
                         M_Elasticidad, G_Cortante)
    R_apoyos = calc_reacciones(K_global, U, F_global, dofs_fijos)
    dofs_libres = np.setdiff1d(np.arange(6*len(xyz)), dofs_fijos)
    return xs, U.reshape(len(xyz), 6), res, R_apoyos, (K_red, F_red, dofs_libres)


