import pint
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra la proyeccion 3d)

ureg = pint.UnitRegistry()



## Generadora de la matriz
def gen_matriz_global(xyz, conectividad, Area, M_Elasticidad=206*ureg.GPa):
    num_nodos = len(xyz)
    num_elementos = len(conectividad)
    dim = xyz.shape[1]          
    n_lib = num_nodos * dim     

    K_global = np.zeros((n_lib, n_lib)) * (ureg.newton / ureg.meter)# creo aqui la matriz general

    for e in range(num_elementos):
        n1, n2 = conectividad[e]
        p1 = xyz[n1] #en vez de trabajar con cada uno separado trabajo todos como 1 punto
        p2 = xyz[n2]

        delta = (p2 - p1).to(ureg.meter)        #Esos son los deltas de los componentes
        L_e = np.sqrt(np.sum(delta**2))
        n_dir = (delta / L_e).magnitude         # vector de cosenos directores [cx,cy] o [cx,cy,cz]

        k_const = (M_Elasticidad * Area / L_e).to(ureg.newton / ureg.meter)

        # Para este caso lo que tenemos que hacer es la proyeccion ortogonal, por ende matricialmente applicamos T= U(U^T U)^-1U^T
        
        T = np.outer(n_dir, n_dir)              #Eso se puede hacer con el producto externo que me genera la matriz de proyeccion pero eso es para 1 nodo, y necesito para 2
        ke_sub = np.block([[ T, -T],
                            [-T,  T]])           # Aqui se lo aplico a cada uno de los nodos
        ke_global = k_const * ke_sub

        # Lo mismo que en 2D, pero ahora tengo en cuenta 3  o 2 o 1 dimensiones.
        dofs = [dim*n1 + k for k in range(dim)] + [dim*n2 + k for k in range(dim)] #suma de dos listas, la primera es para el nodo 1 y la segunda para el nodo 2, y cada una tiene los grados de libertad de cada nodo

        for i in range(2*dim):
            for j in range(2*dim):
                K_global[dofs[i], dofs[j]] += ke_global[i, j]

    return K_global


#el vector de fuerzas 
def gen_f_global(xyz, f_nodales):
    dim = xyz.shape[1]
    n_lib = len(xyz) * dim
    F_global = np.zeros(n_lib) * ureg.newton
    for entrada in f_nodales:
        nodo = entrada[0]
        fuerzas = entrada[1:]            # (fx,fy) o (fx,fy,fz)
        for k, f in enumerate(fuerzas):
            F_global[dim*nodo + k] += f
    return F_global


##Aqui hago loa apoyos
def Restriciones(apoyos):
    dim = apoyos.shape[1] - 1         # Resto 1 por el numero del nodo en cual esta
    dofs_fijos = []
    for entrada in apoyos:
        nodo = entrada[0]
        restricciones = entrada[1:]
        for k, r in enumerate(restricciones):
            if r == 1:
                dofs_fijos.append(dim*nodo + k)
    return sorted(dofs_fijos)


#no hay cambias como es redicr las que tienen restricciones y resolver
def Sis_reducido(K_global, F_global, dofs_fijos):
    K_reducida = np.delete(np.delete(K_global, dofs_fijos, axis=0), dofs_fijos, axis=1)
    F_reducida = np.delete(F_global, dofs_fijos, axis=0)
    U_red = np.linalg.solve(K_reducida, F_reducida)
    return U_red, K_reducida, F_reducida

#aqui lleno el resto de las filas
def U_completa(dofs_fijos, U_red):
    n_lib = len(U_red) + len(dofs_fijos)
    gl_libres = [d for d in range(n_lib) if d not in dofs_fijos]
    U = np.zeros(n_lib) * U_red.units
    U[gl_libres] = U_red
    return U


#Fuerza axial y esfuerzo en cada elemento, misma logica que gen_matriz_global
#pero ahora en vez de armar la matriz, proyecto el desplazamiento relativo
#(u2-u1) sobre la direccion de la barra para sacar el alargamiento
def calc_esfuerzos(xyz, conectividad, U, Area, M_Elasticidad=206*ureg.GPa):
    dim = xyz.shape[1]
    num_elementos = len(conectividad)
    fuerzas = np.zeros(num_elementos) * ureg.newton
    esfuerzos = np.zeros(num_elementos) * ureg.MPa

    for e in range(num_elementos):
        n1, n2 = conectividad[e]
        p1, p2 = xyz[n1], xyz[n2]

        delta = (p2 - p1).to(ureg.meter)
        L_e = np.sqrt(np.sum(delta**2))
        n_dir = (delta / L_e).magnitude          # mismo vector de cosenos directores

        u1 = U[dim*n1:dim*n1+dim].to(ureg.meter).magnitude
        u2 = U[dim*n2:dim*n2+dim].to(ureg.meter).magnitude
        elongacion = np.dot(n_dir, u2 - u1) * ureg.meter   # positivo = se alarga (traccion)

        k_const = (M_Elasticidad * Area / L_e).to(ureg.newton / ureg.meter)
        N = (k_const * elongacion).to(ureg.newton)

        fuerzas[e] = N
        esfuerzos[e] = (N / Area).to(ureg.MPa)

    return fuerzas, esfuerzos


#Aqui le pedi ayuda a Claude para poder graficar los puntos
def _igualar_ejes_3d(ax, puntos):
    """Los ejes 3D de matplotlib no respetan aspect='equal' solos;
    hay que fijar los 3 rangos a mano con el mismo radio para que
    el reticulado no se vea estirado/deformado."""
    centro = puntos.mean(axis=0)
    radio = max(np.ptp(puntos, axis=0).max() / 2 * 1.2, 1e-6)
    ax.set_xlim(centro[0]-radio, centro[0]+radio)
    ax.set_ylim(centro[1]-radio, centro[1]+radio)
    ax.set_zlim(centro[2]-radio, centro[2]+radio)


def graficar_reticulado(xyz, conectividad, U=None, apoyos=None, escala=None, titulo="Reticulado"):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, dtype=float)
    num_nodos, dim = xyz_m.shape
    es_3d = (dim == 3)

    xyz_def = None
    if U is not None:
        U_m = U.magnitude if hasattr(U, "magnitude") else np.asarray(U, dtype=float)
        despl = U_m.reshape(num_nodos, dim)
        if escala is None:
            tam = np.ptp(xyz_m, axis=0).max()
            max_despl = np.abs(despl).max()
            escala = 0.1 * tam / max_despl if max_despl > 1e-12 else 1.0
        xyz_def = xyz_m + despl * escala

    n_paneles = 2 if U is not None else 1
    subplot_kw = {'projection': '3d'} if es_3d else {}
    fig, axes = plt.subplots(1, n_paneles, figsize=(6.5*n_paneles, 6), subplot_kw=subplot_kw)
    axes = np.atleast_1d(axes)

    def _dibujar_estructura(ax, coords, color, estilo='-', numerar=True):
        for n1, n2 in conectividad:
            if es_3d:
                xs = [coords[n1, 0], coords[n2, 0]]
                ys = [coords[n1, 1], coords[n2, 1]]
                zs = [coords[n1, 2], coords[n2, 2]]
                ax.plot(xs, ys, zs, marker='o', linestyle=estilo, color=color,
                        linewidth=2, markersize=6)
            else:
                ax.plot([coords[n1, 0], coords[n2, 0]], [coords[n1, 1], coords[n2, 1]],
                        marker='o', linestyle=estilo, color=color, linewidth=2,
                        markersize=6, zorder=2)
        if numerar:
            for i, p in enumerate(coords):
                if es_3d:
                    ax.text(p[0], p[1], p[2], str(i), fontsize=9, color=color, weight='bold')
                else:
                    ax.annotate(str(i), (p[0], p[1]), textcoords="offset points",
                                xyztext=(6, 6), fontsize=9, color=color, weight='bold')

    def _dibujar_apoyos(ax, coords):
        if apoyos is None:
            return
        for entrada in apoyos:
            nodo = entrada[0]
            restringido_total = all(r == 1 for r in entrada[1:])
            p = coords[nodo]
            estilo = dict(marker='^' if restringido_total else 'o',
                           color='black' if restringido_total else 'gray',
                           markersize=13 if restringido_total else 11, zorder=4)
            if not restringido_total:
                estilo.update(fillstyle='none', markeredgewidth=2)
            if es_3d:
                ax.plot([p[0]], [p[1]], [p[2]], **estilo)
            else:
                ax.plot(p[0], p[1], **estilo)

    ax0 = axes[0]
    _dibujar_estructura(ax0, xyz_m, 'steelblue')
    _dibujar_apoyos(ax0, xyz_m)
    ax0.set_title("Antes (geometria original)")
    ax0.set_xlabel('x [m]'); ax0.set_ylabel('y [m]')
    if es_3d:
        ax0.set_zlabel('z [m]')
        _igualar_ejes_3d(ax0, xyz_m)
    else:
        ax0.set_aspect('equal', adjustable='datalim')
        ax0.grid(alpha=0.3)

    if U is not None:
        ax1 = axes[1]
        _dibujar_estructura(ax1, xyz_m, 'lightsteelblue', numerar=False)
        _dibujar_estructura(ax1, xyz_def, 'indianred', estilo='--')
        _dibujar_apoyos(ax1, xyz_m)
        ax1.set_title(f"Despues (deformada, escala x{escala:.0f})")
        ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
        if es_3d:
            ax1.set_zlabel('z [m]')
            _igualar_ejes_3d(ax1, np.vstack([xyz_m, xyz_def]))
        else:
            ax1.set_aspect('equal', adjustable='datalim')
            ax1.grid(alpha=0.3)
        ax1.legend(handles=[
            Line2D([0], [0], color='lightsteelblue', marker='o', linewidth=2, label='Original'),
            Line2D([0], [0], color='indianred', marker='o', linestyle='--', linewidth=2, label='Deformada'),
        ], loc='best', fontsize=8)

    fig.suptitle(titulo)
    plt.tight_layout()
    return fig, axes


#Reticulado coloreado segun la fuerza axial de cada elemento
#(rojo=traccion, azul=compresion), con barra de colores
def graficar_esfuerzos(xyz, conectividad, fuerzas, titulo="Fuerzas axiales por elemento"):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, dtype=float)
    fuerzas_kN = fuerzas.to(ureg.kN).magnitude if hasattr(fuerzas, "magnitude") else np.asarray(fuerzas, dtype=float)
    dim = xyz_m.shape[1]
    es_3d = (dim == 3)

    vmax = max(np.abs(fuerzas_kN).max(), 1e-9)   # evita vmax=0 si todo esta descargado
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)
    cmap = plt.cm.coolwarm

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d' if es_3d else None)

    for e, (n1, n2) in enumerate(conectividad):
        color = cmap(norm(fuerzas_kN[e]))
        if es_3d:
            ax.plot([xyz_m[n1, 0], xyz_m[n2, 0]],
                    [xyz_m[n1, 1], xyz_m[n2, 1]],
                    [xyz_m[n1, 2], xyz_m[n2, 2]], color=color, linewidth=3)
        else:
            ax.plot([xyz_m[n1, 0], xyz_m[n2, 0]],
                    [xyz_m[n1, 1], xyz_m[n2, 1]], color=color, linewidth=3)

    if es_3d:
        ax.scatter(xyz_m[:, 0], xyz_m[:, 1], xyz_m[:, 2], color='black', s=15, zorder=3)
        ax.set_zlabel('z [m]')
        _igualar_ejes_3d(ax, xyz_m)
    else:
        ax.scatter(xyz_m[:, 0], xyz_m[:, 1], color='black', s=15, zorder=3)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Fuerza axial [kN]  (+ traccion / - compresion)")

    ax.set_title(titulo)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    plt.tight_layout()
    return fig, ax


#Deformacion vertical (eje y por defecto) de cada nodo, como grafico de barras
def graficar_deformacion_vertical(U, num_nodos, dim, eje_vertical=1,
                                    titulo="Deformacion vertical por nodo"):
    U_mm = U.to(ureg.mm).magnitude if hasattr(U, "magnitude") else np.asarray(U, dtype=float)
    despl_vert = U_mm.reshape(num_nodos, dim)[:, eje_vertical]

    fig, ax = plt.subplots(figsize=(9, 4))
    colores = ['indianred' if v < 0 else 'steelblue' for v in despl_vert]
    ax.bar(range(num_nodos), despl_vert, color=colores)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('nodo')
    ax.set_ylabel('desplazamiento vertical [mm]')
    ax.set_title(titulo)
    ax.set_xticks(range(num_nodos))
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    return fig, ax


# ==================================================================
#Coordenadas [x, y] en metros
Area = 0.01 * ureg.meter**2
M_Elasticidad = 206 * ureg.GPa
xyz = np.array([
    [0.0, 0.0,0.0],
    [2.0, 0.0,0.0],
    [2.0, 1.0,0.0],
    [4.0, 0.0,0.0],
    [4.0, 2.0,0.0],
    [6.0, 0.0,0.0],
    [6.0, 2.75,0.0],
    [8.0, 0.0,0.0],
    [8.0, 3.25,0.0],
    [10.0, 0.0,0.0],
    [10.0, 3.5,0.0],
    [12.0, 0.0,0.0],
    [12.0, 3.25,0.0],
    [14.0, 0.0,0.0],
    [14.0, 2.75,0.0],
    [16.0, 0.0,0.0],
    [16.0, 2.0,0.0],
    [18.0, 0.0,0.0],
    [18.0, 1.0,0.0],
    [20.0, 0.0,0.0],
    [0.0, 0.0,3.0],
    [2.0, 0.0,3.0],
    [2.0, 1.0,3.0],
    [4.0, 0.0,3.0],
    [4.0, 2.0,3.0],
    [6.0, 0.0,3.0],
    [6.0, 2.75,3.0],
    [8.0, 0.0,3.0],
    [8.0, 3.25,3.0],
    [10.0, 0.0,3.0],
    [10.0, 3.5,3.0],
    [12.0, 0.0,3.0],
    [12.0, 3.25,3.0],
    [14.0, 0.0,3.0],
    [14.0, 2.75,3.0],
    [16.0, 0.0,3.0],
    [16.0, 2.0,3.0],
    [18.0, 0.0,3.0],
    [18.0, 1.0,3.0],
    [20.0, 0.0,3.0],
]) * ureg.meter

conectividad = np.array([
    [0, 1],
    [0, 2],
    [1, 2],
    [1, 3],
    [2, 3],
    [2, 4],
    [3, 4],
    [3, 5],
    [3, 6],
    [4, 6],
    [5, 6],
    [5, 7],
    [6, 7],
    [6, 8],
    [7, 8],
    [7, 9],
    [7, 10],
    [8, 10],
    [9, 10],
    [9, 11],
    [10, 11],
    [10, 12],
    [11, 12],
    [11, 13],
    [11, 14],
    [12, 14],
    [13, 14],
    [13, 15],
    [14, 15],
    [14, 16],
    [15, 16],
    [15, 17],
    [15, 18],
    [16, 18],
    [17, 18],
    [17, 19],
    [18, 19],#primer lado
    [20, 21],
    [20, 22],
    [21, 22],
    [21, 23],
    [22, 23],
    [22, 24],
    [23, 24],
    [23, 25],
    [23, 26],
    [24, 26],
    [25, 26],
    [25, 27],
    [26, 27],
    [26, 28],
    [27, 28],
    [27, 29],
    [27, 30],
    [28, 30],
    [29, 30],
    [29, 31],
    [30, 31],
    [30, 32],
    [31, 32],
    [31, 33],
    [31, 34],
    [32, 34],
    [33, 34],
    [33, 35],
    [34, 35],
    [34, 36],
    [35, 36],
    [35, 37],
    [35, 38],
    [36, 38],
    [37, 38],
    [37, 39],
    [38, 39],#segundo lado
    [0, 20],#de aqui en adelante, postes rectos (cada nodo con su par al otro lado)
    [1, 21],
    [2, 22],
    [3, 23],
    [4, 24],
    [5, 25],
    [6, 26],
    [7, 27],
    [8, 28],
    [9, 29],
    [10, 30],
    [11, 31],
    [12, 32],
    [13, 33],
    [14, 34],
    [15, 35],
    [16, 36],
    [17, 37],
    [18, 38],
    [19, 39],
    [0, 21],#de aqui en adelante, cruzadas entre vecinos de la cuerda inferior (y=0)
    [1, 20],
    [1, 23],
    [3, 21],
    [3, 25],
    [5, 23],
    [5, 27],
    [7, 25],
    [7, 29],
    [9, 27],
    [9, 31],
    [11, 29],
    [11, 33],
    [13, 31],
    [13, 35],
    [15, 33],
    [15, 37],
    [17, 35],
    [17, 39],
    [19, 37],
    [2, 24],#de aqui en adelante, cruzadas entre vecinos de la arcada (cuerda superior)
    [4, 22],
    [4, 26],
    [6, 24],
    [6, 28],
    [8, 26],
    [8, 30],
    [10, 28],
    [10, 32],
    [12, 30],
    [12, 34],
    [14, 32],
    [14, 36],
    [16, 34],
    [16, 38],
    [18, 36],
    [32, 8],#AQUI ES PARA QUE NO SEA MECANISMO
    [28,12],
    
], dtype=int)
 #print(xy)


#COndiciones de borde y fuerzas

# [Nodo, rx, ry, rz] -> 1: Fijo/Restringido, 0: Libre
apoyos = np.array([
    [0, 1, 1,1],  # 0 fijo en x e y
    [19, 1, 1,1],  
    [20, 1, 1,1],  # 0 fijo en x e y
    [39, 1, 1,1],  
], dtype=int)

# [Nodo, Fx, Fy] 
fuerzas_nodales = [
    (9, 0.0 * ureg.kN,- 250.0 * ureg.kN, 0* ureg.kN),
    (29, 0.0 * ureg.kN,- 250.0 * ureg.kN, 0* ureg.kN),
    (29, 0 * ureg.kN,0 * ureg.kN, 100* ureg.kN),
]



K_global = gen_matriz_global(xyz, conectividad, Area, M_Elasticidad)
F_global = gen_f_global(xyz, fuerzas_nodales)
dofs_fijos = Restriciones(apoyos)
U_red, K_red, F_red = Sis_reducido(K_global, F_global, dofs_fijos)
U = U_completa(dofs_fijos, U_red)

print("dofs_fijos:", dofs_fijos)
print("rank(K_reducida):", np.linalg.matrix_rank(K_red.magnitude), "de", K_red.shape[0])
print("\nDesplazamientos (mm):")
U_mm = U.to(ureg.mm)

for i in range(len(xyz)):
    print(f"  nodo {i}: ux={U_mm[3*i]: .4f}  uy={U_mm[3*i+1]: .4f}  uz={U_mm[3*i+2]: .4f}")

fuerzas, esfuerzos = calc_esfuerzos(xyz, conectividad, U, Area, M_Elasticidad)

print("\nFuerza axial por elemento:")
fuerzas_kN = fuerzas.to(ureg.kN)
esfuerzos_MPa = esfuerzos.to(ureg.MPa)
for e, (n1, n2) in enumerate(conectividad):
    tipo = "traccion  " if fuerzas_kN[e].magnitude >= 0 else "compresion"
    print(f"  elemento {e:2d} ({n1:2d}-{n2:2d}): N={fuerzas_kN[e]: 8.3f}  "
          f"sigma={esfuerzos_MPa[e]: 8.3f}  ({tipo})")

fig1, axes1 = graficar_reticulado(xyz, conectividad, U=U, apoyos=apoyos,
                                    escala=100, titulo="Puente 3D")
fig2, ax2 = graficar_esfuerzos(xyz, conectividad, fuerzas, titulo="Fuerzas axiales - Puente 3D")
fig3, ax3 = graficar_deformacion_vertical(U, num_nodos=len(xyz), dim=xyz.shape[1],
                                            eje_vertical=1, titulo="Deformacion vertical (eje y) por nodo")

plt.show()
#plt.savefig("puente_3d.png", dpi=150)
#print("\nguardado")