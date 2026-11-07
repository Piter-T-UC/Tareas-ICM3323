from types import SimpleNamespace

import numpy as np

from unidades import ureg
from solucion_kirsch import kirsch_tracciones
from etapa1_malla.mallador import bloque, unir, arco, nodos_en, apoyos_en
from etapa1_malla.cargas import lados_desde_nodos
from etapa3_matrices.matriz_D import matriz_D
from etapa3_matrices.gen_matriz_dispersa import gen_matriz_dispersa
from etapa3_matrices.gen_f_distribuida import gen_f_distribuida
from etapa4_resolver.restricciones import Restriciones
from etapa4_resolver.sistema_disperso import resolver_disperso
from etapa4_resolver.tensiones import calc_tensiones


## 1/4 de placa cuadrada de lado 2L con agujero de radio a, en dos bloques separados por y = x
##   bloque 1: arco de 0 a 45 grados  ->  borde x = L
##   bloque 2: arco de 45 a 90 grados ->  borde y = L
## n_rad, n_ang: elementos radiales y angulares por bloque
## razon: largo del ultimo elemento radial / el primero. Con 6 x 4 y razon 3 es la malla del problema 2
def malla_placa(a_m, L_m, n_rad, n_ang, razon, tipo="Q9"):
    p45 = a_m/np.sqrt(2)*np.array([1, 1])
    bloque_1 = bloque([(a_m, 0), (L_m, 0), (L_m, L_m), p45], n_rad, n_ang, tipo,
                      lados={3: arco((0, 0), a_m, 0, np.pi/4)}, razon=(razon, 1))
    bloque_2 = bloque([p45, (L_m, L_m), (0, L_m), (0, a_m)], n_rad, n_ang, tipo,
                      lados={3: arco((0, 0), a_m, np.pi/4, np.pi/2)}, razon=(razon, 1))
    xy_m, conectividad = unir(bloque_1, bloque_2)
    return (xy_m*ureg.m).to(ureg.mm), conectividad


## Cargas distribuidas de la placa, floats SI en a_m, L_m, S_Pa
##   carga = "uniforme": S constante en x = L, borde y = L libre (el problema real)
##   carga = "exacta":   tracciones de Kirsch en x = L e y = L (como si la placa siguiera infinita)
def cargas_placa(xy, conectividad, a_m, L_m, S_Pa, carga):
    en_xL = nodos_en(xy, lambda x, y: np.isclose(x, L_m))
    en_yL = nodos_en(xy, lambda x, y: np.isclose(y, L_m))
    Pa = ureg.Pa
    if carga == "uniforme":
        return lados_desde_nodos(conectividad, [(n, S_Pa*Pa, 0*Pa) for n in en_xL])
    if carga != "exacta":
        raise ValueError(f"carga '{carga}' no existe (opciones: uniforme, exacta)")
    xy_m = xy.to(ureg.m).magnitude
    # una lista por borde: en la esquina (L, L) la normal cambia
    t_x = [(n, *(kirsch_tracciones(*xy_m[n], 1, 0, S_Pa, a_m)*Pa)) for n in en_xL]
    t_y = [(n, *(kirsch_tracciones(*xy_m[n], 0, 1, S_Pa, a_m)*Pa)) for n in en_yL]
    return lados_desde_nodos(conectividad, t_x) + lados_desde_nodos(conectividad, t_y)


## Resuelve la placa y devuelve todo en floats SI, mas los nodos de control A (a, 0), B (0, a),
## C (L, 0) y D (0, L). E y espesor van con unidades; a_m, L_m, S_Pa son floats
def resolver_placa(a_m, L_m, S_Pa, E, nu, espesor, estado, n_rad, n_ang, razon, carga):
    xy, conectividad = malla_placa(a_m, L_m, n_rad, n_ang, razon)
    apoyos = np.array(apoyos_en(xy, lambda x, y: np.isclose(y, 0), 0, 1)
                      + apoyos_en(xy, lambda x, y: np.isclose(x, 0), 1, 0), dtype=int)
    D = matriz_D(E, nu, estado)
    K = gen_matriz_dispersa(xy, conectividad, D, espesor)
    F = gen_f_distribuida(xy, conectividad, cargas_placa(xy, conectividad, a_m, L_m, S_Pa, carga), espesor)
    U = resolver_disperso(K, F, Restriciones(apoyos))
    sigma_nodos, _ = calc_tensiones(xy, conectividad, U, D)

    xy_m = xy.to(ureg.m).magnitude
    nodo_en = lambda px, py: nodos_en(xy, lambda x, y: np.isclose(x, px) and np.isclose(y, py))[0]
    return SimpleNamespace(xy_m=xy_m, conectividad=conectividad, U_nodos_m=U.reshape(-1, 2),
                           sigma_nodos=sigma_nodos, n_gdl=len(U), n_elem=len(conectividad),
                           A=nodo_en(a_m, 0), B=nodo_en(0, a_m), C=nodo_en(L_m, 0), D=nodo_en(0, L_m))
