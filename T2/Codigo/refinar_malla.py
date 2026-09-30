import numpy as np

from gen_f_distribuida import _buscar_elemento


## Divide barras en sub-elementos iguales.
## divisiones = {indice_barra: n}; las que no aparecen quedan con n = 1.
## props = {"Area": ..., "I_x": ...}: escalar o matriz (n_elem, 2) [inicio, fin] (Quantity);
## se devuelven como (n_nuevo, 2) interpoladas linealmente en los nodos de cada sub-elemento.
## Los nodos intermedios se agregan al final: la numeracion original (apoyos, fuerzas) no cambia.
## barras[i] = lista ordenada de sub-elementos (sentido a -> b) de la barra original i.
def refinar(xyz, conectividad, vec_ref, divisiones, props, cargas_dist=()):
    n_elem = len(conectividad)
    unidad = xyz.units
    xyz_m = list(xyz.magnitude)
    props2 = {k: np.broadcast_to(v.magnitude, (n_elem, 2)) for k, v in props.items()}

    conect_nueva, vec_nuevo, barras = [], [], []
    props_nuevas = {k: [] for k in props}
    for i, (a, b) in enumerate(conectividad):
        n = divisiones.get(i, 1)
        nodos = [a]
        for k in range(1, n):
            xyz_m.append(xyz_m[a] + k/n*(xyz_m[b] - xyz_m[a]))
            nodos.append(len(xyz_m) - 1)
        nodos.append(b)

        barras.append(list(range(len(conect_nueva), len(conect_nueva) + n)))
        xi = np.linspace(0, 1, n + 1)
        for k in range(n):
            conect_nueva.append([nodos[k], nodos[k+1]])
            vec_nuevo.append(vec_ref[i])
            for c, v in props2.items():
                v1, v2 = v[i]
                props_nuevas[c].append([(1 - xi[k])*v1 + xi[k]*v2, (1 - xi[k+1])*v1 + xi[k+1]*v2])

    # cargas distribuidas: una entrada lineal por sub-elemento, interpolada en sus extremos
    cargas_nuevas = []
    for carga in cargas_dist:
        n1, n2, *valores = carga
        extra = []
        if isinstance(valores[-1], str):
            extra = [valores[-1]]
            valores = valores[:-1]
        if len(valores) == 2:                        # constante
            valores = valores * 2
        F1, M1, F2, M2 = valores
        i, invertido = _buscar_elemento(conectividad, n1, n2)
        if invertido:
            F1, M1, F2, M2 = F2, M2, F1, M1
        n = len(barras[i])
        xi = np.linspace(0, 1, n + 1)
        for k, e in enumerate(barras[i]):
            p, q = xi[k], xi[k+1]
            cargas_nuevas.append((*conect_nueva[e], (1 - p)*F1 + p*F2, (1 - p)*M1 + p*M2,
                                  (1 - q)*F1 + q*F2, (1 - q)*M1 + q*M2, *extra))

    return (np.array(xyz_m) * unidad, np.array(conect_nueva, dtype=int), np.array(vec_nuevo),
            {c: np.array(v) * props[c].units for c, v in props_nuevas.items()},
            cargas_nuevas, barras)
