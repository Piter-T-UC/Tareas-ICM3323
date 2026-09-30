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
G_Cortante = 80 * ureg.GPa       # dato del enunciado

I_0 = 2e6 * ureg.mm**4       # empotramiento (x = 0)
I_1 = 1e6 * ureg.mm**4       # extremo libre (x = L)
Area = 1000 * ureg.mm**2     # no influye: no hay carga axial

L = 1 * ureg.meter
P = 1.0 * ureg.kN
a_P = 0.5 * ureg.meter
M = 2.0 * ureg.kN * ureg.meter
kNm = ureg.kN * ureg.meter

# discretizaciones: siempre pares para que haya un nodo en x = L/2 (donde va P)
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
    I_x = I_elem                 # flexion en plano X-Z (carga P en Z)
    I_y = I_elem                 # flexion en plano X-Y (sin carga en ese plano)
    J_p = 2*I_x                  # seccion circular: J = 2I, tambien lineal

    apoyos = np.array([
        [0, 1, 1, 1, 1, 1, 1],   # nodo 0: empotrado
    ], dtype=int)

    nodo_P = int(np.argmin(np.abs(xs - a_P.magnitude)))   # nodo donde se aplica P
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


# ------------------------------------------------------------------
# solucion analitica (parte a) en mm, N: mismas expresiones de P1.tex
# v positivo hacia abajo (sentido de P), theta = v', phi = giro de torsion
# ------------------------------------------------------------------
E = M_Elasticidad.to(ureg.MPa).magnitude          # N/mm^2
G = G_Cortante.to(ureg.MPa).magnitude
Lmm = L.to(ureg.mm).magnitude
amm = a_P.to(ureg.mm).magnitude
Pn = P.to(ureg.N).magnitude
Tn = M.to(ureg.N*ureg.mm).magnitude
I0 = I_0.to(ureg.mm**4).magnitude
I1 = I_1.to(ureg.mm**4).magnitude
z0 = I0 * Lmm / I1                                 # I(z) = I1/L (z0 - z) -> z0 = 2000 mm


def analitica(z):
    z = np.asarray(z, dtype=float)
    # tramo 1 (z <= a): E v'' = P (a - z) / I(z)
    k = Pn * Lmm / (E * I1)                        # P / (E I1/L)
    th1 = lambda s: k * (s + (z0 - amm) * np.log(1 - s/z0))
    v1 = lambda s: k * (s**2/2 - (z0 - amm)*s - (z0 - amm)*(z0 - s)*np.log(1 - s/z0))
    th_a, v_a = th1(amm), v1(amm)
    zc = np.minimum(z, amm)
    theta = np.where(z <= amm, th1(zc), th_a)
    v = np.where(z <= amm, v1(zc), v_a + th_a*(z - amm))   # pasado a: solido rigido
    phi = Tn * Lmm / (2 * G * I1) * np.log(z0 / (z0 - z))  # J = 2 I
    M_f = np.where(z < amm, Pn*(amm - z), 0.0) / 1e3       # N*m
    V = np.where(z < amm, Pn, 0.0)                         # N
    T = np.full_like(z, Tn / 1e3)                          # N*m
    return v, theta, phi, M_f, V, T


## Interpolacion de Hermite de la flecha dentro de cada elemento (misma base que la rigidez)
def campos_mef(xs, U_nodos, n_pts=21):
    zmm = xs * 1e3
    v_n = -U_nodos[:, 2] * 1e3          # uz [m] -> v hacia abajo [mm]
    th_n = U_nodos[:, 4]                # theta_y = -duz/dx = dv/dx
    phi_n = U_nodos[:, 3]               # giro de torsion
    filas = []
    for e in range(len(xs) - 1):
        h = zmm[e+1] - zmm[e]
        xi = np.linspace(0, 1, n_pts)
        N = [1 - 3*xi**2 + 2*xi**3, h*(xi - 2*xi**2 + xi**3), 3*xi**2 - 2*xi**3, h*(-xi**2 + xi**3)]
        dN = [(-6*xi + 6*xi**2)/h, 1 - 4*xi + 3*xi**2, (6*xi - 6*xi**2)/h, -2*xi + 3*xi**2]
        q = [v_n[e], th_n[e], v_n[e+1], th_n[e+1]]
        v = sum(Ni*qi for Ni, qi in zip(N, q))
        th = sum(dNi*qi for dNi, qi in zip(dN, q))
        phi = (1 - xi)*phi_n[e] + xi*phi_n[e+1]
        filas.append(np.column_stack([zmm[e] + xi*h, v, th, phi]))
    nodos = np.column_stack([zmm, v_n, th_n, phi_n])
    return np.vstack(filas), nodos


## M y V por elemento: M lineal, V constante (no hay carga distribuida)
## se duplican los puntos de borde para que V se dibuje escalonado
def esfuerzos_mef(xs, res):
    zmm = xs * 1e3
    filas = []
    for e in range(len(xs) - 1):
        M_i, M_j = res["M_x"][e]                     # N*m
        V_e = -(M_j - M_i) / (xs[e+1] - xs[e]) + 0.0 # V = -dM/dz, convencion de la parte a
        filas.append([zmm[e], M_i, V_e])
        filas.append([zmm[e+1], M_j, V_e])
    return np.array(filas)


def guardar(nombre, datos, columnas):
    np.savetxt(os.path.join(carpeta_datos, nombre), datos, header=" ".join(columnas),
               comments="", fmt="%.8e")


os.makedirs(carpeta_datos, exist_ok=True)

z_an = np.linspace(0, Lmm, 401)
guardar("analitica.dat", np.column_stack([z_an, *analitica(z_an)]),
        ["z", "v", "theta", "phi", "M", "V", "T"])

v_L, th_L, phi_L = (c[0] for c in analitica([Lmm])[:3])
v_a, th_a, phi_a = (c[0] for c in analitica([amm])[:3])
print(f"Analitica: v(L/2) = {v_a:.4f} mm   v(L) = {v_L:.4f} mm   theta(L) = {th_L:.4e} rad"
      f"   phi(L) = {phi_L:.4e} rad")

convergencia = []
print("\n  n    v(L/2) [mm]   v(L) [mm]   theta(L) [rad]   phi(L/2) [rad]   phi(L) [rad]"
      "   M(0) [N*m]   V(0) [N]")
for n in mallas:
    xs, U_nodos, res, R_apoyos, sist_red = resolver(n)
    campos, nodos = campos_mef(xs, U_nodos)
    esf = esfuerzos_mef(xs, res)
    guardar(f"mef_n{n}_campos.dat", campos, ["z", "v", "theta", "phi"])
    guardar(f"mef_n{n}_nodos.dat", nodos, ["z", "v", "theta", "phi"])
    guardar(f"mef_n{n}_esfuerzos.dat", esf, ["z", "M", "V"])

    i_a = n // 2
    print(f"  {n:<3d}  {nodos[i_a, 1]:11.5f}  {nodos[-1, 1]:10.5f}  {nodos[-1, 2]:15.5e}"
          f"  {nodos[i_a, 3]:15.5e}  {nodos[-1, 3]:13.5e}  {esf[0, 1]:11.2f}  {esf[0, 2]:9.2f}")
    convergencia.append([n, abs(nodos[-1, 1] - v_L) / v_L, abs(nodos[-1, 2] - th_L) / th_L,
                         abs(nodos[-1, 3] - phi_L) / phi_L])

    if n == mallas[0]:
        print(f"       Reacciones: Rz = {R_apoyos[0, 2]/1e3:.3f} kN   My = {R_apoyos[0, 4]/1e3:.3f} kN*m"
              f"   Mx = {R_apoyos[0, 3]/1e3:.3f} kN*m"
              f"   (teorico: {P:.1f}, P*a = {(P*a_P).to(kNm):.1f}, M = {M:.1f})")
        # los GDL se desacoplan: solo flexion en X-Z (u_z, theta_y) y torsion (theta_x) tienen carga
        K_red, F_red, dofs_libres = sist_red
        idx = np.nonzero(np.isin(dofs_libres % 6, (2, 3, 4)))[0]
        print("\n       Sistema reducido (u_z, theta_x, theta_y) (SI):")
        print(ecuacion_latex(K_red[np.ix_(idx, idx)], F_red[idx], dofs_libres[idx]))
        print()

guardar("convergencia.dat", np.array(convergencia), ["n", "err_v", "err_theta", "err_phi"])
print("\nError relativo en z = L:")
for n, ev, et, ep in convergencia:
    print(f"  n = {int(n):<3d}  v: {ev:.3e}   theta: {et:.3e}   phi: {ep:.3e}")
print(f"\ndatos guardados en {carpeta_datos}/")
