import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

S  = 100e6# tensión aplicada lejos [Pa]
a  = 0.01 # radio del agujero [m]
E  = 200e9  # módulo de Young [Pa]
nu = 0.3  # Poisson
PLANO = "tension"   # "tension" o "deformacion"

G = E / (2 * (1 + nu))
kappa = (3 - nu) / (1 + nu) if PLANO == "tension" else 3 - 4 * nu


def tensiones_polares(r, th):
    """Ec. (61) de Timoshenko."""
    q2, q4 = (a / r) ** 2, (a / r) ** 4
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    srr = S / 2 * (1 - q2) + S / 2 * (1 + 3 * q4 - 4 * q2) * c2
    stt = S / 2 * (1 + q2) - S / 2 * (1 + 3 * q4) * c2
    trt = -S / 2 * (1 - 3 * q4 + 2 * q2) * s2
    return srr, stt, trt


def desplazamientos_polares(r, th):
    c2, s2 = np.cos(2 * th), np.sin(2 * th)
    ur = S / (8 * G * r) * ((kappa - 1) * r**2 + 2 * a**2
                            + 2 * (r**2 + (kappa + 1) * a**2 - a**4 / r**2) * c2)
    ut = -S / (8 * G * r) * 2 * (r**2 + (kappa - 1) * a**2 + a**4 / r**2) * s2
    return ur, ut


def kirsch_tensiones(x, y):
    r, th = np.hypot(x, y), np.arctan2(y, x)
    srr, stt, trt = tensiones_polares(r, th)
    c, s = np.cos(th), np.sin(th)
    sxx = srr * c**2 + stt * s**2 - 2 * trt * s * c
    syy = srr * s**2 + stt * c**2 + 2 * trt * s * c
    txy = (srr - stt) * s * c + trt * (c**2 - s**2)
    return sxx, syy, txy


def kirsch_desplazamientos(x, y):
    r, th = np.hypot(x, y), np.arctan2(y, x)
    ur, ut = desplazamientos_polares(r, th)
    c, s = np.cos(th), np.sin(th)
    return ur * c - ut * s, ur * s + ut * c


def kirsch_tracciones(x, y, nx, ny):

    sxx, syy, txy = kirsch_tensiones(x, y)
    return sxx * nx + txy * ny, txy * nx + syy * ny


def von_mises(sxx, syy, txy):
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3 * txy**2)

def verificar(n=7, h=1e-7):
    rng = np.random.default_rng(0)
    x = rng.uniform(-5 * a, 5 * a, n); y = rng.uniform(-5 * a, 5 * a, n)
    ok = np.hypot(x, y) > 1.2 * a
    x, y = x[ok], y[ok]
    ux_px = (kirsch_desplazamientos(x + h, y)[0] - kirsch_desplazamientos(x - h, y)[0]) / (2 * h)
    uy_py = (kirsch_desplazamientos(x, y + h)[1] - kirsch_desplazamientos(x, y - h)[1]) / (2 * h)
    gxy = ((kirsch_desplazamientos(x, y + h)[0] - kirsch_desplazamientos(x, y - h)[0])
           + (kirsch_desplazamientos(x + h, y)[1] - kirsch_desplazamientos(x - h, y)[1])) / (2 * h)
    sxx, syy, txy = kirsch_tensiones(x, y)
    if PLANO == "tension":
        exx = (sxx - nu * syy) / E; eyy = (syy - nu * sxx) / E
    else:
        exx = (1 + nu) / E * ((1 - nu) * sxx - nu * syy)
        eyy = (1 + nu) / E * ((1 - nu) * syy - nu * sxx)
    err = max(np.max(abs(ux_px - exx)), np.max(abs(uy_py - eyy)),
              np.max(abs(gxy - txy / G))) / (S / E)
    print(f"Error relativo deformaciones (u) vs Hooke(sigma): {err:.1e}")



def malla_polar(L=5 * a, nr=200, nt=360):
    r = a * np.logspace(0, np.log10(np.sqrt(2) * L / a), nr)
    th = np.linspace(-np.pi, np.pi, nt)
    R, T = np.meshgrid(r, th)
    return R * np.cos(T), R * np.sin(T)

## Aqui abajo me ayudo mucho la IA
def graficar(L=5 * a, escala=None):
    X, Y = malla_polar(L)
    sxx, syy, txy = kirsch_tensiones(X, Y)
    ux, uy = kirsch_desplazamientos(X, Y)
    campos = {
        r"$\sigma_{xx}/S$": sxx / S, r"$\sigma_{yy}/S$": syy / S,
        r"$\tau_{xy}/S$": txy / S, r"$\sigma_{VM}/S$": von_mises(sxx, syy, txy) / S,
        r"$u_x$ [µm]": ux * 1e6, r"$u_y$ [µm]": uy * 1e6,
    }

    fig, axs = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for ax, (nom, F) in zip(axs.flat, campos.items()):
        cmap = "coolwarm" if np.nanmin(F) < 0 < np.nanmax(F) else "viridis"
        lim = np.nanmax(abs(F)) if cmap == "coolwarm" else None
        cf = ax.contourf(X / a, Y / a, F, 40, cmap=cmap,
                         vmin=-lim if lim else None, vmax=lim)
        ax.add_patch(Circle((0, 0), 1, color="w", zorder=3))
        ax.add_patch(Circle((0, 0), 1, fill=False, lw=1.2, zorder=4))
        ax.set_xlim(-L / a, L / a); ax.set_ylim(-L / a, L / a)
        ax.set_aspect("equal"); ax.set_title(nom); ax.set_xlabel("x/a"); ax.set_ylabel("y/a")
        fig.colorbar(cf, ax=ax, shrink=0.85)
    fig.suptitle(f"Kirsch: placa infinita, agujero r=a, tracción S en x  (plano de {PLANO})")
    fig.savefig("kirsch_campos.png", dpi=150)

    # --- Cortes clásicos -------------------------------------------------
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    th = np.linspace(0, 2 * np.pi, 400)
    axs[0].plot(np.degrees(th), tensiones_polares(a, th)[1] / S)
    axs[0].axhline(3, ls="--", c="k", lw=.8); axs[0].axhline(-1, ls="--", c="k", lw=.8)
    axs[0].set(xlabel=r"$\theta$ [°]", ylabel=r"$\sigma_{\theta\theta}/S$",
               title=r"Borde del agujero: $1-2\cos 2\theta$  (máx 3 en $\theta=90°$)")
    axs[0].set_xticks(range(0, 361, 45)); axs[0].grid(alpha=.3)

    y = np.linspace(a, 8 * a, 300)
    axs[1].plot(y / a, kirsch_tensiones(0 * y, y)[0] / S, label=r"$\sigma_{xx}$ en $x=0$")
    axs[1].plot(y / a, 1 + 0.5 * (a / y)**2 + 1.5 * (a / y)**4, "k:", label="Timoshenko p.92")
    axs[1].axhline(1, c="gray", lw=.8)
    axs[1].set(xlabel="y/a", ylabel=r"$\sigma_{xx}/S$", title="Sección por el centro ⊥ a la carga")
    axs[1].legend(); axs[1].grid(alpha=.3)

    x = np.linspace(a, 8 * a, 300)
    sx, sy, _ = kirsch_tensiones(x, 0 * x)
    axs[2].plot(x / a, sx / S, label=r"$\sigma_{xx}$"); axs[2].plot(x / a, sy / S, label=r"$\sigma_{yy}$")
    axs[2].set(xlabel="x/a", ylabel=r"$\sigma/S$", title="Eje de la carga (y=0)")
    axs[2].legend(); axs[2].grid(alpha=.3)
    fig.savefig("kirsch_cortes.png", dpi=150)

    # --- Deformada -------------------------------------------------------
    th = np.linspace(0, 2 * np.pi, 361)
    if escala is None:
        escala = 0.15 * L / np.max(np.hypot(ux, uy))
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for rr in a * np.array([1, 1.5, 2, 3, 4]):
        xx, yy = rr * np.cos(th), rr * np.sin(th)
        dx, dy = kirsch_desplazamientos(xx, yy)
        ax.plot(xx / a, yy / a, c="0.75", lw=.8)
        ax.plot((xx + escala * dx) / a, (yy + escala * dy) / a, c="C3", lw=1.4 if rr == a else .9)
    for t in np.radians(np.arange(0, 360, 22.5)):
        rr = np.linspace(a, 4 * a, 50); xx, yy = rr * np.cos(t), rr * np.sin(t)
        dx, dy = kirsch_desplazamientos(xx, yy)
        ax.plot(xx / a, yy / a, c="0.75", lw=.8)
        ax.plot((xx + escala * dx) / a, (yy + escala * dy) / a, c="C3", lw=.9)
    ax.set_aspect("equal"); ax.set_xlabel("x/a"); ax.set_ylabel("y/a")
    ax.set_title(f"Deformada (gris: original, rojo: ×{escala:.0f})")
    fig.savefig("kirsch_deformada.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    verificar()
    sxx, syy, _ = kirsch_tensiones(np.array([0.0]), np.array([a]))
    print(f"Kt en (0, a): sigma_xx/S = {sxx[0]/S:.3f}   (esperado 3)")
    sxx, _, _ = kirsch_tensiones(np.array([a]), np.array([0.0]))
    print(f"En (a, 0):    sigma_yy/S = {kirsch_tensiones(np.array([a]), np.array([0.0]))[1][0]/S:.3f}   (esperado -1)")
    graficar()
