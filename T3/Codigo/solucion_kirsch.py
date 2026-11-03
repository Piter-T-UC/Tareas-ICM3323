import numpy as np


## Solucion de Kirsch (Timoshenko ec. 61): placa infinita con agujero de radio a,
## traccion S lejos en la direccion x. Todo en floats SI (Pa, m).


## Tensiones cartesianas (sxx, syy, txy) en (x, y)
def kirsch_tensiones(x, y, S, a):
    r, th = np.hypot(x, y), np.arctan2(y, x)
    q2, q4 = (a/r)**2, (a/r)**4
    c2, s2 = np.cos(2*th), np.sin(2*th)
    srr = S/2*(1 - q2) + S/2*(1 + 3*q4 - 4*q2)*c2
    stt = S/2*(1 + q2) - S/2*(1 + 3*q4)*c2
    trt = -S/2*(1 - 3*q4 + 2*q2)*s2
    c, s = np.cos(th), np.sin(th)
    sxx = srr*c**2 + stt*s**2 - 2*trt*s*c
    syy = srr*s**2 + stt*c**2 + 2*trt*s*c
    txy = (srr - stt)*s*c + trt*(c**2 - s**2)
    return sxx, syy, txy


## Traccion t = sigma n [Pa] en un punto de borde con normal exterior (nx, ny)
def kirsch_tracciones(x, y, nx, ny, S, a):
    sxx, syy, txy = kirsch_tensiones(x, y, S, a)
    return np.array([sxx*nx + txy*ny, txy*nx + syy*ny])
