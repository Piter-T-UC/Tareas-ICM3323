import numpy as np


# ------------------------------------------------------------------
# 6. GRAFICAR RETICULADO -- generalizado a 2D o 3D
# ------------------------------------------------------------------
def _igualar_ejes_3d(ax, puntos):
    """Los ejes 3D de matplotlib no respetan aspect='equal' solos;
    hay que fijar los 3 rangos a mano con el mismo radio para que
    el reticulado no se vea estirado/deformado."""
    centro = puntos.mean(axis=0)
    radio = max(np.ptp(puntos, axis=0).max() / 2 * 1.2, 1e-6)
    ax.set_xlim(centro[0]-radio, centro[0]+radio)
    ax.set_ylim(centro[1]-radio, centro[1]+radio)
    ax.set_zlim(centro[2]-radio, centro[2]+radio)
