import numpy as np

from unidades import ureg
from gen_matriz_global import propiedades_si, geometria_elemento, k_local_viga


#aqui tenemos el calculo de los esfuerzos respecto a U
def calc_esfuerzos(xyz, conectividad, U, Area, I_x, I_y, J_p, vec_ref,
                   M_Elasticidad=206*ureg.GPa, G_Cortante=206*ureg.GPa/(2*(1+0.3)),
                   c_x=None, c_y=None, r_t=None):
    num_elementos = len(conectividad)
    E = M_Elasticidad.to(ureg.pascal).magnitude
    G = G_Cortante.to(ureg.pascal).magnitude
    A, Ix, Iy, Jp = propiedades_si(num_elementos, Area, I_x, I_y, J_p)

    def _dist(d):
        if d is None:
            return np.full((num_elementos, 2), np.nan)
        return np.broadcast_to(d.to(ureg.meter).magnitude, (num_elementos, 2))
    cx, cy, rt = _dist(c_x), _dist(c_y), _dist(r_t)

    largos = np.zeros(num_elementos)
    rot = np.zeros((num_elementos, 3, 3))            # filas = ejes locales x_L, y_L, z_L
    u_local = np.zeros((num_elementos, 12))
    internas = np.zeros((num_elementos, 2, 6))       # [V_x, V_y, N, M_x, M_y, T]

    for e in range(num_elementos):
        n1, n2 = conectividad[e]
        L, R, T_v = geometria_elemento(xyz, n1, n2, vec_ref[e])
        dofs = [6*n1 + k for k in range(6)] + [6*n2 + k for k in range(6)]
        u_L = T_v @ U[dofs]                           # global -> local
        f_L = k_local_viga(E, G, A[e], Ix[e], Iy[e], Jp[e], L) @ u_L   # cargas solo en los nodos

        largos[e] = L
        rot[e] = R
        u_local[e] = u_L
        internas[e, 0] = -f_L[:6]
        internas[e, 1] = f_L[6:]

    # limpia el ruido de redondeo (1e-15 relativo) para que no aparezca en diagramas
    internas[np.abs(internas) < 1e-10*np.abs(internas).max(initial=0)] = 0.0
    Vx, Vy, N, Mx, My, T = np.moveaxis(internas, 2, 0)   # cada uno (n_elementos, 2)

    # deformaciones generalizadas de la seccion
    eps = N / (E*A)                    # deformacion axial
    kap_x = Mx / (E*Ix)                # curvaturas
    kap_y = My / (E*Iy)
    tw = T / (G*Jp)                    # giro por unidad de largo

    # tensiones: sigma en la fibra mas alejada (suma de valores absolutos:
    # exacto en secciones con esquinas, conservador en secciones circulares)
    sigma_N = N / A
    sigma_max = np.abs(sigma_N) + np.abs(Mx)*cy/Ix + np.abs(My)*cx/Iy
    tau_T = np.abs(T)*rt/Jp
    sigma_vm = np.sqrt(sigma_max**2 + 3*tau_T**2)

    return {
        "E": E, "G": G, "A": A, "I_x": Ix, "I_y": Iy, "J_p": Jp,
        "largos": largos, "rot": rot, "u_local": u_local,
        "V_x": Vx, "V_y": Vy, "N": N, "M_x": Mx, "M_y": My, "T": T,
        "eps": eps, "kappa_x": kap_x, "kappa_y": kap_y, "giro_unitario": tw,
        "sigma_N": sigma_N, "sigma_max": sigma_max, "tau_T": tau_T, "sigma_vm": sigma_vm,
        "c_x": cx, "c_y": cy, "r_t": rt,
    }


## Esfuerzos a lo largo del elemento e (z local de 0 a L), por equilibrio del tramo [0, z]
## partiendo de los esfuerzos en el nodo inicial. Todas las cargas estan en los nodos,
## asi que dentro del elemento V, N y T son constantes y M es lineal.
def esfuerzos_a_lo_largo(res, e, n_pts=41):
    L = res["largos"][e]
    z = np.linspace(0, L, n_pts)
    xi = z / L
    uno = np.ones_like(z)

    Vx0, Vy0, N0, Mx0, My0, T0 = (res[c][e, 0] for c in ("V_x", "V_y", "N", "M_x", "M_y", "T"))
    Vx, Vy, N, T = Vx0*uno, Vy0*uno, N0*uno, T0*uno
    Mx = Mx0 + z*Vy0
    My = My0 - z*Vx0

    # propiedades de seccion interpoladas linealmente (igual que en la rigidez)
    def _lin(clave):
        a = res[clave][e]
        return (1 - xi)*a[0] + xi*a[1]
    A, Ix, Iy, Jp = _lin("A"), _lin("I_x"), _lin("I_y"), _lin("J_p")
    cx, cy, rt = _lin("c_x"), _lin("c_y"), _lin("r_t")

    sigma_N = N / A
    sigma_max = np.abs(sigma_N) + np.abs(Mx)*cy/Ix + np.abs(My)*cx/Iy
    tau_T = np.abs(T)*rt/Jp
    sigma_vm = np.sqrt(sigma_max**2 + 3*tau_T**2)

    return {"z": z, "V_x": Vx, "V_y": Vy, "N": N, "M_x": Mx, "M_y": My, "T": T,
            "sigma_N": sigma_N, "sigma_max": sigma_max, "tau_T": tau_T, "sigma_vm": sigma_vm}


# Reacciones: R = K U - F (solo tiene sentido en los GDL restringidos)
def calc_reacciones(K_global, U, F_global, dofs_fijos):
    R = K_global @ U - F_global
    R_fijos = np.zeros_like(R)
    R_fijos[dofs_fijos] = R[dofs_fijos]
    return R_fijos.reshape(-1, 6)       # (n_nodos, 6): [Rx, Ry, Rz, Mx, My, Mz]
