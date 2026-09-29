import numpy as np


def matriz_rotacion_viga(n_dir, v_ref):
    #aqui lo que hago es darme que lado va para cual en la matriz
    z_L = np.asarray(n_dir, dtype=float) #ya es unitario
    x_L = np.cross(v_ref, z_L) #aqui saco la direccion x si es 2D es el que sale
    norma = np.linalg.norm(x_L)
    if norma < 1e-8:
        raise ValueError(f"vec_ref es paralelo al eje de la barra (n_dir={z_L})")
    x_L = x_L / norma #normalizopor si vec ref no es unitario o otra cosa

    y_L = np.cross(z_L, x_L)

    return np.vstack([x_L, y_L, z_L])
