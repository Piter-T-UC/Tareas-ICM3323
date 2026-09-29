import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registra la proyeccion 3d)

from igualar_ejes_3d import _igualar_ejes_3d


def _curva_elemento(xyz, despl, rot, n1, n2, escala, n_pts=21):
    # deformada del elemento con interpolacion cubica de Hermite (usa desplazamientos y giros)
    x1, x2 = xyz[n1], xyz[n2]
    u1, u2 = despl[n1], despl[n2]
    L = np.linalg.norm(x2 - x1)
    e = (x2 - x1) / L
    xi = np.linspace(0, 1, n_pts)[:, None]
    base = x1 + xi * (x2 - x1)
    if rot is None:   # sin giros -> recta entre nodos
        return base + escala * ((1 - xi) * u1 + xi * u2)

    dim = len(e)
    e3 = np.pad(e, (0, 3 - dim))
    a1, a2 = (u1 @ e) * e, (u2 @ e) * e          # parte axial (lineal)
    t1, t2 = u1 - a1, u2 - a2                    # parte transversal
    r1 = np.cross(rot[n1], e3)[:dim]             # pendiente transversal: d(delta)/ds = theta x e
    r2 = np.cross(rot[n2], e3)[:dim]

    H1 = 1 - 3*xi**2 + 2*xi**3
    H2 = xi - 2*xi**2 + xi**3
    H3 = 3*xi**2 - 2*xi**3
    H4 = -xi**2 + xi**3
    delta = (1 - xi)*a1 + xi*a2 + H1*t1 + H2*L*r1 + H3*t2 + H4*L*r2
    return base + escala * delta


def graficar_reticulado(xyz, conectividad, U=None, apoyos=None, escala=None, titulo="Reticulado"):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, dtype=float)
    num_nodos, dim = xyz_m.shape
    es_3d = (dim == 3)

    xyz_def = None
    curvas = None
    if U is not None:
        U_m = U.magnitude if hasattr(U, "magnitude") else np.asarray(U, dtype=float)
        dof_por_nodo = U_m.size // num_nodos
        U_nodos = U_m.reshape(num_nodos, dof_por_nodo)
        despl = U_nodos[:, :dim]   # traslaciones [U,V,W]
        # giros como vector 3D (en 2D solo theta_z); None si no hay giros (reticulado)
        rot = None
        if es_3d and dof_por_nodo == 6:
            rot = U_nodos[:, 3:6]
        elif not es_3d and dof_por_nodo == 3:
            rot = np.column_stack([np.zeros(num_nodos), np.zeros(num_nodos), U_nodos[:, 2]])
        if escala is None:
            tam = np.ptp(xyz_m, axis=0).max()
            max_despl = np.abs(despl).max()
            escala = 0.1 * tam / max_despl if max_despl > 1e-12 else 1.0
        xyz_def = xyz_m + despl * escala
        curvas = [_curva_elemento(xyz_m, despl, rot, n1, n2, escala) for n1, n2 in conectividad]

    subplot_kw = {'projection': '3d'} if es_3d else {}
    fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=subplot_kw)
    axes = np.atleast_1d(ax)

    def _dibujar_estructura(ax, coords, color, estilo='-', numerar=True, curvas=None):
        if curvas is not None:
            # deformada cubica: curva por elemento + marcadores solo en los nodos
            for c in curvas:
                ax.plot(*c.T, linestyle=estilo, color=color, linewidth=2)
            ax.plot(*coords.T, marker='o', linestyle='none', color=color, markersize=6)
        for n1, n2 in (conectividad if curvas is None else []):
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
                                xytext=(6, 6), fontsize=9, color=color, weight='bold')

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

    # un solo grafico: sin U muestra la geometria; con U, la original tenue + la deformada
    if U is None:
        _dibujar_estructura(ax, xyz_m, 'steelblue')
        coords_lim = xyz_m
    else:
        _dibujar_estructura(ax, xyz_m, 'lightsteelblue', numerar=False)
        _dibujar_estructura(ax, xyz_def, 'indianred', estilo='--', curvas=curvas)
        ax.set_title(f"Deformada (escala x{escala:.0f})")
        ax.legend(handles=[
            Line2D([0], [0], color='lightsteelblue', marker='o', linewidth=2, label='Original'),
            Line2D([0], [0], color='indianred', marker='o', linestyle='--', linewidth=2, label='Deformada'),
        ], loc='best', fontsize=8)
        coords_lim = np.vstack([xyz_m, xyz_def] + curvas)
    _dibujar_apoyos(ax, xyz_m)
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    if es_3d:
        ax.set_zlabel('z [m]')
        _igualar_ejes_3d(ax, coords_lim)
    else:
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.3)

    fig.suptitle(titulo)
    plt.tight_layout()
    return fig, axes
