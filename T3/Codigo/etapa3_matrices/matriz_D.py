import numpy as np

from unidades import ureg


## Matriz constitutiva D (3x3) en Pa: sigma = D eps, con
## sigma = [sxx, syy, txy] y eps = [exx, eyy, gxy] (gxy = 2 exy, deformacion ingenieril)
## estado: "tension_plana" o "deformacion_plana"
def matriz_D(E, nu, estado):
    E = E.to(ureg.Pa).magnitude
    if estado == "tension_plana":
        return E/(1 - nu**2) * np.array([[1,  nu, 0],
                                         [nu, 1,  0],
                                         [0,  0,  (1 - nu)/2]])
    if estado == "deformacion_plana":
        return E/((1 + nu)*(1 - 2*nu)) * np.array([[1 - nu, nu,     0],
                                                   [nu,     1 - nu, 0],
                                                   [0,      0,      (1 - 2*nu)/2]])
    raise ValueError(f"estado '{estado}' no existe (opciones: 'tension_plana', 'deformacion_plana')")
