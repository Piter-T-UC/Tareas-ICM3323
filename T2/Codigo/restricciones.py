# ------------------------------------------------------------------
# 3. RESTRICCIONES -- generalizado (infiere dim de la forma de "apoyos")
# ------------------------------------------------------------------
def Restriciones(apoyos):
    dim = apoyos.shape[1] - 1         # cada fila es [nodo, r1, r2, ...]
    dofs_fijos = []
    for entrada in apoyos:
        nodo = entrada[0]
        restricciones = entrada[1:]
        for k, r in enumerate(restricciones):
            if r == 1:
                dofs_fijos.append(dim*nodo + k)
    return sorted(dofs_fijos)
