import pint
import numpy as np

ureg = pint.UnitRegistry()


M_Elasticidad=206*ureg.GPa
A_0=1600*ureg.mm**2
L=10*ureg.m
n=5

nodos = np.linspace(0,L,n)
def area(x,l):
    return A_0*np.exp(-x/l)

matriz_rigidez = np.zeros((n,n))*ureg.newton/ureg.meter
conectividad = np.zeros((n-1,2),dtype=int)
for i in range(n-1):
    conectividad[i,0] = i
    conectividad[i,1] = i+1


for i in range(n-1):
    N1=conectividad[i,0]
    x1 = nodos[N1]
    N2=conectividad[i,1]
    x2 = nodos[N2]
    L_e = x2 - x1 #Esto es el largop de cada uno
    A_1 = area((x1)/2,L)# aqui es donde saco la Area
    A_2 = area((x2)/2,L)#probe con este metodo me dio mayor error
    A_e= (A_1 + A_2)/2
    #A_e = area((x1+x2)/2,L)# aqui es donde saco la Area
    
    k_e = (M_Elasticidad*A_e/L_e)
    matriz_rigidez[N1,N1] += k_e
    matriz_rigidez[N1,N2] -= k_e
    matriz_rigidez[N2,N1] -= k_e
    matriz_rigidez[N2,N2] += k_e


matriz_reducida = matriz_rigidez[1:,1:]# saco la primera fila y primera columna que son u_1=0 y Reaccion desconocida
Fuerza = np.zeros(n)*ureg.newton
Fuerza[n-1] = 3*ureg.kN # Aqui va la fuerza aplicada en el nodoultimo
#print(f"{Fuerza[n-1]}")
Fuerza_reducida = Fuerza[1:]# saco la primera fila y primera columna que son u_1=0 y Reaccion desconocida

K_mag = matriz_reducida.magnitude
F_mag = Fuerza_reducida.magnitude
U_red = np.linalg.solve(matriz_reducida,Fuerza_reducida)#esto me lo mostro la IA de como se resuelve
#print(f"{U:.01e}")
U=np.zeros(n)*ureg.meter
U[1:] = U_red
print(f"{matriz_rigidez@U:.03f}")#esto me lo mostro la IA como hacer esa multiplicacion
print(f"{U_red.to(ureg.mm)[-1]-0.1564*ureg.mm:.05f}")# esto veo el error
print(f"Esfuerzo axial: {(U_red.to(ureg.mm)*M_Elasticidad/L).to(ureg.MPa)}")
#esto es lo que hago para calcular los esfuerzos
num_elementos = n - 1
esfuerzos = np.zeros(num_elementos) * ureg.MPa
for i in range(num_elementos):
    N1 = conectividad[i, 0]
    N2 = conectividad[i, 1]
    L_e = nodos[N2] - nodos[N1]
    du = U[N2] - U[N1]
    
    # sigma = E * epsilon = E * (du / L_e)
    sigma_e = (M_Elasticidad * (du / L_e)).to(ureg.MPa)
    esfuerzos[i] = sigma_e