import numpy as np

from unidades import ureg
from matriz_rotacion_viga import matriz_rotacion_viga


## Rigidez a flexion (plano x-z) con I lineal: I(xi) = I1*(1-xi) + I2*xi
def _k_flex(E, I1, I2, L):
    a = I1 + I2
    b = 2*I1 + I2
    c = I1 + 2*I2
    return (E/L**3) * np.array([[ 6*a,    2*L*b,          -6*a,    2*L*c         ],
                                [ 2*L*b,  L**2*(3*I1+I2), -2*L*b,  L**2*a        ],
                                [-6*a,   -2*L*b,           6*a,   -2*L*c         ],
                                [ 2*L*c,  L**2*a,         -2*L*c,  L**2*(I1+3*I2)]])



## Como tengo seccion variable esto me ayuda a hacer lista (escalar -> se replica; matriz [I_inicio, I_fin] por elemento -> se respeta)
def propiedades_si(num_elementos, Area, I_x, I_y, J_p):
    A = np.broadcast_to(Area.to(ureg.meter**2).magnitude, (num_elementos, 2))
    Ix = np.broadcast_to(I_x.to(ureg.meter**4).magnitude, (num_elementos, 2))
    Iy = np.broadcast_to(I_y.to(ureg.meter**4).magnitude, (num_elementos, 2))
    Jp = np.broadcast_to(J_p.to(ureg.meter**4).magnitude, (num_elementos, 2))
    return A, Ix, Iy, Jp


## Largo [m], matriz de rotacion R (3x3) y T_v = blockdiag(R,R,R,R) de un elemento
def geometria_elemento(xyz, n1, n2, vec_ref_e):
    delta = (xyz[n2] - xyz[n1]).to(ureg.meter).magnitude
    L = np.linalg.norm(delta)
    R = matriz_rotacion_viga(delta / L, vec_ref_e)   # eje local z_L a lo largo de la barra
    T_v = np.zeros((12, 12))
    ##Aqui lo que hago es colocar las matrices de rotacion R en la diagonal
    for b in range(4):
        T_v[3*b:3*b+3, 3*b:3*b+3] = R
    return L, R, T_v


## Matriz de rigidez local (12x12), orden por nodo:
## [U1,V1,W1,Thx1,Thy1,Thz1, U2,V2,W2,Thx2,Thy2,Thz2]
## A_e, Ix_e, Iy_e, Jp_e = [valor_inicio, valor_fin] del elemento (floats SI)
def k_local_viga(E, G, A_e, Ix_e, Iy_e, Jp_e, L):
    K_local = np.zeros((12, 12))
    D = np.diag([1, -1, 1, -1])# cambio de signo de giros para el plano y-z

    def _sumar_bloque(idx, bloque):
        K_local[np.ix_(idx, idx)] += bloque

    ##Ahora sumo las rigidez por bloque, cada uno va relacionado a su propio nodo
    idx_ax = [2, 8]  # W1, W2
    A_med = (A_e[0] + A_e[1]) / 2 # int B^T E A(x) B dx con A lineal y N lineales
    _sumar_bloque(idx_ax, (E*A_med/L) * np.array([[1, -1],
                                                [-1, 1]]))

    idx_tor = [5, 11]                      # Thz1, Thz2
    J_med = (Jp_e[0] + Jp_e[1]) / 2        # exacto para J lineal con N lineales
    _sumar_bloque(idx_tor, (G*J_med/L) * np.array([[1, -1],
                                                     [-1, 1]]))

    idx_xz = [0, 4, 6, 10] # U1, Thy1, U2, Thy2 (flexion plano x-z)
    _sumar_bloque(idx_xz, _k_flex(E, Iy_e[0], Iy_e[1], L))

    idx_yz = [1, 3, 7, 9]                  # V1, Thx1, V2, Thx2 (flexion plano y-z)
    _sumar_bloque(idx_yz, D @ _k_flex(E, Ix_e[0], Ix_e[1], L) @ D) #pero con signos cambiados
    return K_local

## Es basicamente lo mismo que para barras pero mas elementos y teniendo en cuenta la descomposición
def gen_matriz_global(xyz, conectividad, Area, I_x, I_y, J_p, vec_ref,
                       M_Elasticidad=206*ureg.GPa, G_Cortante=206*ureg.GPa/(2*(1+0.3))):
    num_nodos = len(xyz)
    num_elementos = len(conectividad)
    dof_por_nodo = 6                      # [U, V, W, Theta_x, Theta_y, Theta_z]
    n_lib = num_nodos * dof_por_nodo

    # K_global mezcla unidades distintas por bloque (N/m, N, N*m): ya no es
    # representable como una sola Quantity de pint, se trabaja en floats SI.
    K_global = np.zeros((n_lib, n_lib))
    E = M_Elasticidad.to(ureg.pascal).magnitude
    G = G_Cortante.to(ureg.pascal).magnitude
    A, Ix, Iy, Jp = propiedades_si(num_elementos, Area, I_x, I_y, J_p)

    for e in range(num_elementos):
        n1, n2 = conectividad[e]
        L, _, T_v = geometria_elemento(xyz, n1, n2, vec_ref[e])
        K_local = k_local_viga(E, G, A[e], Ix[e], Iy[e], Jp[e], L)

        # aqui simplemente la roto con la multiplicacion de matrices
        ke_global = T_v.T @ K_local @ T_v

        dofs = [dof_por_nodo*n1 + k for k in range(dof_por_nodo)] + \
               [dof_por_nodo*n2 + k for k in range(dof_por_nodo)]
        K_global[np.ix_(dofs, dofs)] += ke_global

    return K_global
