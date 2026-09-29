import numpy as np
from datetime import date


# ==================================================================
# EXPORTACION A LATEX: tablas, figuras TikZ e informe completo
# ==================================================================
# Todas las funciones devuelven strings (para pegar sueltas en otro .tex);
# informe_latex() arma el documento completo y lo escribe a disco.
# "res" es la salida de calc_esfuerzos() y "R" la de calc_reacciones().

PREAMBULO = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-noshorthands]{babel}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{caption}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\setlength{\tabcolsep}{4.5pt}

\definecolor{baja}{RGB}{42,120,214}
\definecolor{alta}{RGB}{227,73,72}
\definecolor{neutro}{RGB}{137,135,129}
\definecolor{fantasma}{RGB}{200,199,193}
\definecolor{apoyocol}{RGB}{74,58,167}
\definecolor{cargacol}{RGB}{237,161,0}
\definecolor{momcol}{RGB}{0,150,120}
\definecolor{diagcol}{RGB}{214,95,40}
"""

# (azimut, elevacion) en grados, misma idea que matplotlib
VISTAS = {"iso": (-60, 30), "XY": (-90, 90), "XZ": (-90, 0), "YZ": (0, 0)}
_NOMBRE_VISTA = {"iso": "isométrica", "XY": "en planta ($xy$)",
                 "XZ": "en elevación ($xz$)", "YZ": "en elevación ($yz$)"}


# ------------------------------------------------------------------
# utilidades
# ------------------------------------------------------------------
def _n(v, d=3):
    """Numero en modo matematico (asi el signo menos sale bien); NaN -> --."""
    if not np.isfinite(v):
        return "--"
    return f"${round(v, d) + 0.0:.{d}f}$"          # + 0.0 evita el "-0.000"


def _sci(v):
    m, x = f"{v:.1e}".split("e")
    return rf"{m}\times 10^{{{int(x)}}}"


def _c(p):
    return f"({p[0]:.4f},{p[1]:.4f},{p[2]:.4f})"


def _proyector(vista):
    az, el = np.deg2rad(VISTAS[vista] if isinstance(vista, str) else vista)
    sx = [-np.sin(az), np.cos(az), 0.0]
    sy = [-np.cos(az)*np.sin(el), -np.sin(az)*np.sin(el), np.cos(el)]
    return np.array([sx, sy])                      # (2,3): pantalla = P @ X


def _abrir_tikz(puntos, vista, ancho, alto):
    """Abre un tikzpicture con ejes x,y,z proyectados y escalados para que
    'puntos' (m) quepa en ancho x alto (cm). Las coordenadas se escriben en 3D
    y en metros; lo que lleva cm explicitos (apoyos, leyenda) va en el lienzo."""
    P = _proyector(vista)
    pant = puntos @ P.T
    rango = np.ptp(pant, axis=0)
    k = min(ancho / max(rango[0], 1e-9), alto / max(rango[1], 1e-9))
    ejes = ", ".join(f"{a}={{({k*P[0, i]:.4f}cm,{k*P[1, i]:.4f}cm)}}"
                     for i, a in enumerate("xyz"))
    y_min = k*pant[:, 1].min()
    x_med = k*(pant[:, 0].min() + pant[:, 0].max())/2
    return rf"\begin{{tikzpicture}}[{ejes}]", (x_med, y_min, k)


def _figura(tikz, caption, label):
    return "\n".join([r"\begin{figure}[!tbp]", r"\centering", tikz,
                      rf"\caption{{{caption}}}", rf"\label{{fig:{label}}}", r"\end{figure}"])


def _tabla(caption, label, columnas, encabezados, filas):
    """Tabla no flotante; longtable solo si es larga (longtable + figuras
    flotantes en la misma pagina desbordan el \\output)."""
    n = len(encabezados)
    cab = " & ".join(encabezados) + r" \\"
    if len(filas) <= 30:
        # minipage: el titulo y la tabla no se separan entre paginas
        return "\n".join([r"\begin{center}", r"\begin{minipage}{\linewidth}", r"\centering",
                          rf"\captionof{{table}}{{{caption}}}\label{{tab:{label}}}",
                          rf"\begin{{tabular}}{{@{{}}{columnas}@{{}}}}",
                          r"\toprule", cab, r"\midrule"]
                         + [" & ".join(f) + r" \\" for f in filas]
                         + [r"\bottomrule", r"\end{tabular}", r"\end{minipage}", r"\end{center}"])
    lineas = [rf"\begin{{longtable}}{{@{{}}{columnas}@{{}}}}",
              rf"\caption{{{caption}}}\label{{tab:{label}}}\\",
              r"\toprule", cab, r"\midrule", r"\endfirsthead",
              r"\toprule", cab, r"\midrule", r"\endhead",
              rf"\midrule \multicolumn{{{n}}}{{r@{{}}}}{{\footnotesize\itshape continúa}}\\",
              r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    lineas += [" & ".join(f) + r" \\" for f in filas]
    lineas.append(r"\end{longtable}")
    return "\n".join(lineas)


def _metrica(res):
    """Tension usada para colorear: von Mises si hay r_t y c, si no sigma_max,
    si no solo la axial. Devuelve (valor por elemento en MPa, rotulo)."""
    for clave, rotulo in (("sigma_vm", r"\sigma_{\mathrm{VM}}"),
                          ("sigma_max", r"\sigma_{\max}"),
                          ("sigma_N", r"|\sigma_N|")):
        v = np.abs(res[clave]).max(axis=1)
        if np.all(np.isfinite(v)):
            return v/1e6, rotulo


def escala_sugerida(xyz_m, U):
    tam = np.ptp(xyz_m, axis=0).max()
    d = np.abs(U.reshape(-1, 6)[:, :3]).max()
    if d < 1e-15:
        return 1.0
    bruta = 0.12*tam/d
    base = 10**np.floor(np.log10(bruta))
    return float(max(b*base for b in (1, 2, 5) if b*base <= bruta))


def curva_deformada(xyz_m, n1, n2, res, e, escala, n=17):
    """Deformada de un elemento con las funciones de Hermite (flexion) y lineales
    (axial), en coordenadas globales [m]. Misma interpolacion que la rigidez."""
    L = res["largos"][e]
    U1, V1, W1, Tx1, Ty1, _, U2, V2, W2, Tx2, Ty2, _ = res["u_local"][e]
    xi = np.linspace(0, 1, n)
    f1, f2 = 1 - 3*xi**2 + 2*xi**3, 3*xi**2 - 2*xi**3
    p1, p2 = L*(xi - 2*xi**2 + xi**3), L*(-xi**2 + xi**3)
    u = f1*U1 + p1*Ty1 + f2*U2 + p2*Ty2            # plano x-z: Thy = du/dz
    v = f1*V1 - p1*Tx1 + f2*V2 - p2*Tx2            # plano y-z: Thx = -dv/dz
    w = (1 - xi)*W1 + xi*W2
    base = xyz_m[n1] + xi[:, None]*(xyz_m[n2] - xyz_m[n1])
    return base + escala*(np.column_stack([u, v, w]) @ res["rot"][e])


# ------------------------------------------------------------------
# figura principal: geometria + apoyos + cargas + deformada + tensiones
# ------------------------------------------------------------------
def figura_estructura(xyz, conectividad, res, apoyos=None, F_global=None, U=None,
                      vista="iso", escala=None, ancho=11.0, alto=7.0,
                      caption=None, label="viga3d-estructura"):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, float)
    tam = np.ptp(xyz_m, axis=0).max()
    if escala is None and U is not None:
        escala = escala_sugerida(xyz_m, U)

    curvas = []
    if U is not None:
        curvas = [curva_deformada(xyz_m, n1, n2, res, e, escala)
                  for e, (n1, n2) in enumerate(conectividad)]
    pts = np.vstack([xyz_m] + curvas)
    abrir, (x_med, y_min, k) = _abrir_tikz(pts, vista, ancho, alto)
    L = [abrir]

    # 1) elementos coloreados por tension
    sig, rot_sig = _metrica(res)
    s_max = sig.max() if sig.max() > 0 else 1.0
    for e, (n1, n2) in enumerate(conectividad):
        p = round(100*sig[e]/s_max)
        L.append(rf"\draw[alta!{p}!baja, line width=2.2pt] {_c(xyz_m[n1])} -- {_c(xyz_m[n2])};")

    # 2) deformada (Hermite), rayada encima
    for cur in curvas:
        L.append(r"\draw[black!80, line width=0.7pt] plot[smooth] coordinates {"
                 + " ".join(_c(q) for q in cur) + "};")

    # 3) apoyos: cuadrado = empotrado (6 GDL), triangulo = parcial
    if apoyos is not None:
        for fila in apoyos:
            q = _c(xyz_m[fila[0]])
            if all(r == 1 for r in fila[1:]):
                L.append(rf"\draw[apoyocol, fill=apoyocol!30, line width=0.6pt] "
                         rf"{q} ++(-0.16cm,-0.16cm) rectangle ++(0.32cm,0.32cm);")
            else:
                L.append(rf"\draw[apoyocol, fill=apoyocol!15, line width=0.6pt] "
                         rf"{q} -- ++(-0.17cm,-0.3cm) -- ++(0.34cm,0) -- cycle;")

    # 4) cargas: fuerzas (flecha simple) y momentos (flecha doble)
    if F_global is not None:
        Fn = F_global.reshape(-1, 6)
        for bloque, color, punta, unidad in ((slice(0, 3), "cargacol", "-{Latex}", "kN"),
                                             (slice(3, 6), "momcol", "-{Latex}{Latex}", r"kN\,m")):
            vec = Fn[:, bloque]
            mags = np.linalg.norm(vec, axis=1)
            if mags.max() < 1e-12:
                continue
            for i in np.nonzero(mags > 1e-12)[0]:
                d = vec[i]/mags[i]*0.18*tam*(0.5 + 0.5*mags[i]/mags.max())
                ini = xyz_m[i] - d if color == "cargacol" else xyz_m[i]
                fin = xyz_m[i] if color == "cargacol" else xyz_m[i] + d
                lab = ini if color == "cargacol" else fin
                L.append(rf"\draw[{color}, line width=1pt, {punta}] {_c(ini)} -- {_c(fin)};")
                L.append(rf"\node[{color}, font=\scriptsize, fill=white, inner sep=1pt] at "
                         rf"{_c(lab)} {{{mags[i]/1e3:g}~{unidad}}};")

    # 5) numeracion de nodos y elementos
    for e, (n1, n2) in enumerate(conectividad):
        L.append(rf"\node[draw, circle, font=\tiny, fill=white, inner sep=0.8pt] at "
                 rf"{_c((xyz_m[n1] + xyz_m[n2])/2)} {{{e}}};")
    for i, q in enumerate(xyz_m):
        L.append(rf"\fill {_c(q)} circle (1.4pt);")
        L.append(rf"\node[font=\scriptsize, above left=0pt] at {_c(q)} {{{i}}};")

    # 6) leyenda de colores (en el lienzo, bajo la figura)
    y0 = y_min - 1.1
    L.append(rf"\shade[left color=baja, right color=alta] ({x_med - 2:.3f}cm,{y0:.3f}cm) "
             rf"rectangle ++(4cm,0.22cm);")
    L.append(rf"\node[font=\scriptsize, anchor=east] at ({x_med - 2.1:.3f}cm,{y0 + 0.11:.3f}cm) {{0}};")
    L.append(rf"\node[font=\scriptsize, anchor=west] at ({x_med + 2.1:.3f}cm,{y0 + 0.11:.3f}cm) "
             rf"{{{s_max:.1f} MPa}};")
    L.append(rf"\node[font=\scriptsize, anchor=north] at ({x_med:.3f}cm,{y0 - 0.05:.3f}cm) "
             rf"{{${rot_sig}$ máxima por elemento}};")
    # triada de ejes globales (0.8 cm en el lienzo)
    L.append(rf"\begin{{scope}}[shift={{({x_med - 5.5:.3f}cm,{y0:.3f}cm)}}, "
             r"-{Latex[length=3pt]}, font=\scriptsize]")
    for i, a in enumerate("xyz"):
        d = np.zeros(3)
        d[i] = 0.8/k
        L.append(rf"\draw (0,0,0) -- {_c(d)} node[pos=1.3] {{${a}$}};")
    L.append(r"\end{scope}")
    L.append(r"\end{tikzpicture}")

    if caption is None:
        caption = (f"Estructura, vista {_NOMBRE_VISTA.get(vista, '')}. Elementos coloreados "
                   f"según ${rot_sig}$; cuadrados: empotramientos, triángulos: apoyos parciales; "
                   r"flechas naranjas: fuerzas, flechas dobles verdes: momentos.")
        if U is not None:
            caption += f" En negro fino, la deformada amplificada {escala:g} veces."
    return _figura("\n".join(L), caption, label)


# ------------------------------------------------------------------
# diagrama de esfuerzo interno (N, V_x, V_y, M_x, M_y o T) sobre la estructura
# ------------------------------------------------------------------
# Se dibuja perpendicular a cada barra: M_y y V_x hacia +x_L, M_x y V_y hacia +y_L,
# N y T hacia +x_L. Sin cargas distribuidas, N, V y T son constantes y M lineal.
_DIAG = {"N": (0, "N", "kN", 1e3), "V_x": (0, "V_x", "kN", 1e3), "V_y": (1, "V_y", "kN", 1e3),
         "M_x": (1, "M_x", r"kN\,m", 1e3), "M_y": (0, "M_y", r"kN\,m", 1e3),
         "T": (0, "T", r"kN\,m", 1e3)}


def figura_diagrama(xyz, conectividad, res, componente="M_y", vista="iso",
                    ancho=11.0, alto=7.0, label=None):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, float)
    eje, simbolo, unidad, div = _DIAG[componente]
    val = res[componente]
    v_max = np.abs(val).max()
    s = 0.15*np.ptp(xyz_m, axis=0).max()/v_max if v_max > 1e-12 else 0.0

    polis = []
    for e, (n1, n2) in enumerate(conectividad):
        d = res["rot"][e][eje]                         # direccion local donde se dibuja
        a, b = xyz_m[n1], xyz_m[n2]
        polis.append([a, a + d*val[e, 0]*s, b + d*val[e, 1]*s, b])
    abrir, *_ = _abrir_tikz(np.vstack([xyz_m] + [np.array(p) for p in polis]), vista, ancho, alto)

    L = [abrir]
    for e, (n1, n2) in enumerate(conectividad):
        a, pa, pb, b = polis[e]
        L.append(rf"\filldraw[diagcol, fill=diagcol!20, line width=0.6pt] "
                 rf"{_c(a)} -- {_c(pa)} -- {_c(pb)} -- {_c(b)} -- cycle;")
        L.append(rf"\draw[black, line width=1.2pt] {_c(a)} -- {_c(b)};")
        for q, v in ((pa, val[e, 0]), (pb, val[e, 1])):
            if abs(v) > 1e-9*max(v_max, 1):
                L.append(rf"\node[font=\tiny, fill=white, inner sep=0.8pt] at {_c(q)} "
                         rf"{{${v/div:.2f}$}};")
    for i, q in enumerate(xyz_m):
        L.append(rf"\fill {_c(q)} circle (1.3pt);")
        L.append(rf"\node[font=\scriptsize, above left=0pt] at {_c(q)} {{{i}}};")
    L.append(r"\end{tikzpicture}")

    lado = "x_L" if eje == 0 else "y_L"
    return _figura("\n".join(L),
                   rf"Diagrama de ${simbolo}$ [{unidad}], dibujado hacia ${lado}$ positivo "
                   rf"para valores positivos.", label or f"viga3d-diag-{componente}")


# ------------------------------------------------------------------
# tablas
# ------------------------------------------------------------------
def tabla_nodos(xyz):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, float)
    return _tabla("Coordenadas de los nodos.", "viga3d-nodos", "cccc",
                  ["Nodo", "$x$ [m]", "$y$ [m]", "$z$ [m]"],
                  [[str(i)] + [_n(c) for c in p] for i, p in enumerate(xyz_m)])


def tabla_elementos(conectividad, res):
    """Conectividad y propiedades; area en cm^2 e inercias en cm^4 como 'inicio / fin'."""
    def _par(a, s=1e8, d=1):
        return _n(a[0]*s, d) if np.isclose(a[0], a[1]) else f"{_n(a[0]*s, d)} / {_n(a[1]*s, d)}"
    return _tabla(r"Elementos: conectividad y propiedades (inercias en cm$^4$, "
                  r"``inicio / fin'' si el área o las inercias varían linealmente).", "viga3d-elementos", "cccccccc",
                  ["Elem.", "$i$", "$j$", "$L$ [m]", "$A$ [cm$^2$]", "$I_x$", "$I_y$", "$J_p$"],
                  [[str(e), str(n1), str(n2), _n(res["largos"][e]), _par(res["A"][e], 1e4, 2),
                    _par(res["I_x"][e]), _par(res["I_y"][e]), _par(res["J_p"][e])]
                   for e, (n1, n2) in enumerate(conectividad)])


def tabla_desplazamientos(U):
    Un = U.reshape(-1, 6)
    return _tabla("Desplazamientos y giros nodales (ejes globales).", "viga3d-desplazamientos",
                  "ccccccc",
                  ["Nodo", "$u_x$ [mm]", "$u_y$ [mm]", "$u_z$ [mm]",
                   r"$\theta_x$ [mrad]", r"$\theta_y$ [mrad]", r"$\theta_z$ [mrad]"],
                  [[str(i)] + [_n(v*1e3, 4) for v in fila] for i, fila in enumerate(Un)])


def tabla_reacciones(R, apoyos):
    return _tabla("Reacciones en los apoyos (ejes globales).", "viga3d-reacciones", "ccccccc",
                  ["Nodo", "$R_x$ [kN]", "$R_y$ [kN]", "$R_z$ [kN]",
                   r"$M_x$ [kN\,m]", r"$M_y$ [kN\,m]", r"$M_z$ [kN\,m]"],
                  [[str(f[0])] + [_n(v/1e3) if r else "--" for v, r in zip(R[f[0]], f[1:])]
                   for f in apoyos])


def _filas_por_extremo(conectividad, columnas):
    filas = []
    for e, (n1, n2) in enumerate(conectividad):
        for k, nodo in enumerate((n1, n2)):
            filas.append([str(e) if k == 0 else "", str(nodo)] + [c(e, k) for c in columnas])
    return filas


def tabla_fuerzas_internas(conectividad, res):
    cols = [lambda e, k, c=c: _n(res[c][e, k]/1e3) for c in ("N", "V_x", "V_y", "T", "M_x", "M_y")]
    return _tabla(r"Esfuerzos internos en los extremos de cada elemento (ejes locales; "
                  r"$N>0$ tracción).", "viga3d-internas", "cccccccc",
                  ["Elem.", "Nodo", "$N$ [kN]", "$V_x$ [kN]", "$V_y$ [kN]",
                   r"$T$ [kN\,m]", r"$M_x$ [kN\,m]", r"$M_y$ [kN\,m]"],
                  _filas_por_extremo(conectividad, cols))


def tabla_deformaciones(conectividad, res):
    cols = [lambda e, k: _n(res["eps"][e, k]*1e6, 2),
            lambda e, k: _n(res["kappa_x"][e, k]*1e3, 4),
            lambda e, k: _n(res["kappa_y"][e, k]*1e3, 4),
            lambda e, k: _n(res["giro_unitario"][e, k]*1e3, 4)]
    return _tabla(r"Deformaciones generalizadas de la sección: $\varepsilon=N/EA$, "
                  r"$\kappa=M/EI$, $\theta'=T/GJ$.", "viga3d-deformaciones", "cccccc",
                  ["Elem.", "Nodo", r"$\varepsilon$ [$\mu\varepsilon$]",
                   r"$\kappa_x$ [$10^{-3}$/m]", r"$\kappa_y$ [$10^{-3}$/m]",
                   r"$\theta'$ [mrad/m]"],
                  _filas_por_extremo(conectividad, cols))


def tabla_tensiones(conectividad, res):
    cols = [lambda e, k, c=c: _n(res[c][e, k]/1e6, 2)
            for c in ("sigma_N", "sigma_max", "tau_T", "sigma_vm")]
    return _tabla(r"Tensiones en los extremos: $\sigma_N=N/A$, "
                  r"$\sigma_{\max}=|N|/A+|M_x|c_y/I_x+|M_y|c_x/I_y$, $\tau_T=|T|r/J$, "
                  r"$\sigma_{\mathrm{VM}}=\sqrt{\sigma_{\max}^2+3\tau_T^2}$.",
                  "viga3d-tensiones", "cccccc",
                  ["Elem.", "Nodo", r"$\sigma_N$ [MPa]", r"$\sigma_{\max}$ [MPa]",
                   r"$\tau_T$ [MPa]", r"$\sigma_{\mathrm{VM}}$ [MPa]"],
                  _filas_por_extremo(conectividad, cols))


# ------------------------------------------------------------------
# informe completo
# ------------------------------------------------------------------
def informe_latex(ruta, xyz, conectividad, apoyos, F_global, U, res, R,
                  titulo="Análisis de pórtico espacial", vista="iso", escala=None,
                  diagramas=("N", "M_x", "M_y", "T")):
    xyz_m = xyz.magnitude if hasattr(xyz, "magnitude") else np.asarray(xyz, float)
    if escala is None:
        escala = escala_sugerida(xyz_m, U)
    Un = U.reshape(-1, 6)
    d_norma = np.linalg.norm(Un[:, :3], axis=1)
    i_max = int(np.argmax(d_norma))
    sig, rot_sig = _metrica(res)
    e_max = int(np.argmax(sig))

    Fn = F_global.reshape(-1, 6)
    suma_F, suma_R = Fn[:, :3].sum(0)/1e3, R[:, :3].sum(0)/1e3
    residuo = np.abs(suma_F + suma_R).max()
    energia = 0.5*F_global @ U

    def _vec(v):
        return r",\;".join(f"{x:.3f}" for x in v)

    partes = [PREAMBULO,
              rf"\title{{{titulo}}}",
              r"\author{Método de rigidez directa --- viga de Euler-Bernoulli 3D}",
              rf"\date{{{date.today().strftime('%d/%m/%Y')}}}",
              r"\begin{document}", r"\maketitle",
              r"\section{Modelo}",
              rf"La estructura tiene {len(xyz_m)} nodos y {len(conectividad)} elementos de viga "
              rf"con 6 grados de libertad por nodo ({6*len(xyz_m)} en total, "
              rf"{int(np.sum(np.asarray(apoyos)[:, 1:]))} restringidos). Las inercias pueden variar "
              r"linealmente dentro de cada elemento; la rigidez a flexión se integra de forma exacta "
              r"con $I(\xi)=I_1(1-\xi)+I_2\,\xi$.",
              tabla_nodos(xyz), tabla_elementos(conectividad, res),
              figura_estructura(xyz, conectividad, res, apoyos, F_global, U, vista, escala),
              r"\section{Desplazamientos}",
              rf"El mayor desplazamiento ocurre en el nodo \textbf{{{i_max}}}, con "
              rf"$|u|={d_norma[i_max]*1e3:.4f}$~mm. La deformada de la "
              r"figura~\ref{fig:viga3d-estructura} se interpola con las mismas funciones de "
              r"Hermite de la formulación, así que muestra la curvatura real de cada elemento.",
              tabla_desplazamientos(U),
              r"\section{Esfuerzos internos}",
              tabla_fuerzas_internas(conectividad, res)]
    partes += [figura_diagrama(xyz, conectividad, res, c, vista) for c in diagramas]
    partes += [r"\section{Deformaciones y tensiones}",
               tabla_deformaciones(conectividad, res)]
    if np.all(np.isfinite(res["sigma_max"])):
        partes.append(tabla_tensiones(conectividad, res))
    partes += [rf"El elemento más solicitado es el \textbf{{{e_max}}} "
               rf"(nodos {conectividad[e_max][0]}--{conectividad[e_max][1]}), con "
               rf"${rot_sig}={sig[e_max]:.2f}$~MPa.",
               r"\section{Reacciones y equilibrio}",
               tabla_reacciones(R, apoyos),
               r"\begin{equation*}" "\n"
               rf"\textstyle\sum F=({_vec(suma_F)})~\text{{kN}},\qquad "
               rf"\sum R=({_vec(suma_R)})~\text{{kN}}" "\n"
               r"\end{equation*}",
               rf"El residuo máximo es ${_sci(residuo)}$~kN, de modo que se cumple el equilibrio "
               rf"global. La energía de deformación es "
               rf"$\tfrac12\mathbf F^T\mathbf u={energia:.4f}$~J.",
               r"\end{document}"]

    texto = "\n\n".join(partes)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(texto)
    return texto
