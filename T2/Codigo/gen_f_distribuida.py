import numpy as np

from gen_matriz_global import geometria_elemento


_XI = 0.5 + 0.5*np.array([-np.sqrt(3/5), 0.0, np.sqrt(3/5)])
_W = 0.5*np.array([5/9, 8/9, 5/9])


## Vector de cargas equivalentes local (12), mismo orden y convencion que k_local_viga:
## [U1,V1,W1,Thx1,Thy1,Thz1, U2,V2,W2,Thx2,Thy2,Thz2]
## q1, q2 = [fx,fy,fz,mx,my,mz] locales (floats SI) en el inicio y fin del elemento
def f_eq_local(L, q1, q2):
    f = np.zeros(12)
    for xi, w in zip(_XI, _W):
        fx, fy, fz, mx, my, mz = (1 - xi)*q1 + xi*q2
        # Hermite (flexion) y sus derivadas d/dxi
        H = np.array([1 - 3*xi**2 + 2*xi**3, L*xi*(1 - xi)**2, 3*xi**2 - 2*xi**3, L*(xi**3 - xi**2)])
        dH = np.array([-6*xi + 6*xi**2, L*(1 - 4*xi + 3*xi**2), 6*xi - 6*xi**2, L*(3*xi**2 - 2*xi)])
        Nl = np.array([1 - xi, xi])                  # lineales (axial y torsion)

        fe = np.zeros(12)
        # plano x-z: U = H.[U1,Thy1,U2,Thy2], Thy = dU/dz
        fe[[0, 4, 6, 10]] += fx*H + my*dH/L
        # plano y-z: V = H.[V1,-Thx1,V2,-Thx2], Thx = -dV/dz
        s = np.array([1, -1, 1, -1])
        fe[[1, 3, 7, 9]] += s*(fy*H - mx*dH/L)
        # axial y torsion
        fe[[2, 8]] += fz*Nl
        fe[[5, 11]] += mz*Nl

        f += w*L*fe
    return f


## Busca el elemento (n1,n2) en la conectividad; invertido=True si esta guardado como (n2,n1)
def _buscar_elemento(conectividad, n1, n2):
    for e, (a, b) in enumerate(conectividad):
        if (a, b) == (n1, n2):
            return e, False
        if (a, b) == (n2, n1):
            return e, True
    raise ValueError(f"No existe un elemento entre los nodos {n1} y {n2}")


def _si(q):
    return np.asarray(q.to_base_units().magnitude, dtype=float)


## Devuelve F_dist (vector global, floats SI), f_eq_locales (n_elementos, 12),
## que se usa en calc_esfuerzos para restar las fuerzas de empotramiento, y
## q_locales (n_elementos, 2, 6): carga local [fx,fy,fz,mx,my,mz] al inicio y fin
## de cada elemento, para reconstruir los esfuerzos a lo largo de la barra
def gen_f_distribuida(xyz, conectividad, vec_ref, cargas_dist):
    dof_por_nodo = 6
    F_dist = np.zeros(len(xyz) * dof_por_nodo)
    f_eq_locales = np.zeros((len(conectividad), 12))
    q_locales = np.zeros((len(conectividad), 2, 6))

    for carga in cargas_dist:
        n1, n2, *valores = carga
        local = isinstance(valores[-1], str) and valores[-1].lower() == "local"
        if local:
            valores = valores[:-1]
        if len(valores) == 2:                        # constante
            valores = valores * 2
        q1 = np.concatenate([_si(valores[0]), _si(valores[1])])
        q2 = np.concatenate([_si(valores[2]), _si(valores[3])])

        e, invertido = _buscar_elemento(conectividad, n1, n2)
        if invertido:
            q1, q2 = q2, q1
        a, b = conectividad[e]
        L, R, T_v = geometria_elemento(xyz, a, b, vec_ref[e])

        if not local:                                # global -> local (fuerza y momento por separado)
            q1 = np.concatenate([R @ q1[:3], R @ q1[3:]])
            q2 = np.concatenate([R @ q2[:3], R @ q2[3:]])

        f_loc = f_eq_local(L, q1, q2)
        f_eq_locales[e] += f_loc
        q_locales[e, 0] += q1
        q_locales[e, 1] += q2

        dofs = [dof_por_nodo*a + k for k in range(dof_por_nodo)] + \
               [dof_por_nodo*b + k for k in range(dof_por_nodo)]
        F_dist[dofs] += T_v.T @ f_loc

    return F_dist, f_eq_locales, q_locales
