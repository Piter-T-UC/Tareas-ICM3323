import numpy as np
import matplotlib.pyplot as plt

from unidades import ureg
from gen_matriz_global import gen_matriz_global
from gen_f_global import gen_f_global
from restricciones import Restriciones
from sistema_reducido import Sist_red
from u_completa import U_completa
from graficar_reticulado import graficar_reticulado
from esfuerzos import calc_esfuerzos, calc_reacciones




M_Elasticidad = 206 * ureg.GPa
nu = 0.3
G_Cortante = M_Elasticidad / (2*(1+nu))

I_0 = 2e6 * ureg.mm**4       # empotramiento (x = 0)
I_1 = 1e6 * ureg.mm**4       # extremo libre (x = L)
Area = 1000 * ureg.mm**2     # no influye: no hay carga axial

L = 1 * ureg.meter
n_elem = 2
xs = np.linspace(0, L.magnitude, n_elem + 1)
xyz = np.column_stack([xs, np.zeros_like(xs), np.zeros_like(xs)]) * ureg.meter

conectividad = np.array([[i, i+1] for i in range(n_elem)], dtype=int)
vec_ref = np.array([[0.0, 0.0, 1.0]] * n_elem)   # barra en X, referencia Z global

# inercia en cada nodo y matriz (n_elem, 2) con [I_inicio, I_fin] por elemento
I_nodos = (I_0 + (I_1 - I_0) * xs / L.magnitude).to(ureg.mm**4).magnitude
I_elem = np.column_stack([I_nodos[:-1], I_nodos[1:]]) * ureg.mm**4
I_x = I_elem                 # flexion en plano X-Z (carga P en Z)
I_y = I_elem                 # flexion en plano X-Y (sin carga en ese plano)
J_p = 2*I_x               # rigidez torsional (momento torsor Mx), tambien lineal

apoyos = np.array([
    [0, 1, 1, 1, 1, 1, 1],   # nodo 0: empotrado
], dtype=int)

P = 10.0 * ureg.kN
M = 2.0 * ureg.kN * ureg.meter#momento torsor (eje X = eje de la viga)
i = int(np.argmin(np.abs(xs - 0.5))) #mitad de barra

nodo_M = 1                       # nodo donde se aplica el torsor
kNm = ureg.kN * ureg.meter
fuerzas = [
    (n_elem, 0.0*ureg.kN, 0.0*ureg.kN, -P, 0.0*kNm, 0.0*kNm, 0.0*kNm),   # P en el extremo libre
    (nodo_M, 0.0*ureg.kN, 0.0*ureg.kN, 0*ureg.kN, M, 0.0*kNm, 0.0*kNm),  # torsor Mx
]

K_global = gen_matriz_global(xyz, conectividad, Area, I_x, I_y, J_p, vec_ref,
                                M_Elasticidad, G_Cortante)
F_global = gen_f_global(xyz, fuerzas)
dofs_fijos = Restriciones(apoyos)
U_red, K_red, F_red = Sist_red(K_global, F_global, dofs_fijos)
U = U_completa(dofs_fijos, U_red)

U_nodos = U.reshape(len(xyz), 6)




print("Desplazamientos por nodo:")
for i in range(len(xyz)):
    uz = (U_nodos[i, 2] * ureg.meter).to(ureg.mm)
    thx = (U_nodos[i, 3] * ureg.radian).to(ureg.degree)
    thy = (U_nodos[i, 4] * ureg.radian).to(ureg.degree)
    print(f"  nodo {i} (x={xs[i]:.2f} m): uz={uz:.4f}  thx={thx:.4f}  thy={thy:.4f}")

# solucion analitica con integracion
E = M_Elasticidad.to(ureg.pascal).magnitude
G = G_Cortante.to(ureg.pascal).magnitude
Lm, Pn, Mn = L.magnitude, P.to(ureg.N).magnitude, M.to(ureg.N*ureg.meter).magnitude
x = np.linspace(0, Lm, 20001)
I_fun = (I_0 + (I_1 - I_0) * x / Lm).to(ureg.meter**4).magnitude
# flexion (trabajo virtual): delta = int M m / (E I(x)) dx, carga P en el extremo
uz_teo = np.trapezoid(Pn*(Lm - x)**2 / (E*I_fun), x)
thy_teo = np.trapezoid(Pn*(Lm - x) / (E*I_fun), x)
# torsion: T(x) = M para x < x_M y 0 despues -> theta_x = int_0^x T / (G J(s)) ds
# (constante desde x_M hasta el extremo libre), con J = 2 I
x_M = xs[nodo_M]
thx_teo = np.trapezoid(np.where(x <= x_M, Mn, 0.0) / (G*2*I_fun), x)

print("\n                        MEF            analitica")
print(f"  |uz|  extremo [mm]  {abs(U_nodos[-1, 2])*1e3:12.4f}   {uz_teo*1e3:12.4f}")
print(f"  |thy| extremo [deg] {np.degrees(abs(U_nodos[-1, 4])):12.4f}   {np.degrees(thy_teo):12.4f}")
print(f"  |thx| nodo {nodo_M} [deg]  {np.degrees(abs(U_nodos[nodo_M, 3])):12.4f}   {np.degrees(thx_teo):12.4f}")
print(f"  |thx| extremo [deg] {np.degrees(abs(U_nodos[-1, 3])):12.4f}   {np.degrees(thx_teo):12.4f}")

R_apoyos = calc_reacciones(K_global, U, F_global, dofs_fijos)
print(f"\nReacciones empotramiento: Rz = {R_apoyos[0, 2]/1e3:.3f} kN   My = {R_apoyos[0, 4]/1e3:.3f} kN*m"
        f"   Mx = {R_apoyos[0, 3]/1e3:.3f} kN*m")
print(f"  (teorico: Rz = {P:.1f}, |My| = PL = {(P*L).to(kNm):.1f}, |Mx| = M = {M:.1f})")

res = calc_esfuerzos(xyz, conectividad, U, Area, I_x, I_y, J_p, vec_ref,
                        M_Elasticidad, G_Cortante)
print("\nMomento flector M_x y torsor T por elemento [kN*m] (inicio, fin):")
for e in range(n_elem):
    print(f"  elem {e}: M_x = {res['M_x'][e, 0]/1e3: .3f} {res['M_x'][e, 1]/1e3: .3f}"
            f"   T = {res['T'][e, 0]/1e3: .3f} {res['T'][e, 1]/1e3: .3f}")

graficar_reticulado(xyz, conectividad, U=U, apoyos=apoyos,
                    escala=20, titulo="Problema 1 - voladizo con inercia variable")

plt.savefig("problema_1.png", dpi=150)      # antes de show: al cerrar la ventana la figura se borra
print("\nproblema_1.png guardado")
plt.show()
