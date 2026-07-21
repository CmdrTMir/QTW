import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# ---- H-Laser ----
def H_z(H_tb, z, sigma_z, k0, a):
    z0 = 3 * sigma_z
    O0 = 0.2 * t
    omega_0 = -2 * t * cos(k0 * a)

    f_z = np.exp(-(z - z0)**2 / (2 * sigma_z**2))
    Omega = O0 * f_z * np.exp(-1j * omega_0 * z)

    H_laser = np.zeros((N+1, N+1), dtype=complex)
    H_laser[0, 1] = Omega
    H_laser[1, 0] = np.conj(Omega)

    return H_tb + H_laser

# ---- solver-Funktion ----
def schrödinger(t, y, H_tb, hbar, sigma_z, k0, a):
    psi = y
    H = H_z(H_tb, t, sigma_z, k0, a)
    psi_dot = -1j/ hbar * (H @ psi)
    return psi_dot

def wave_omega(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 2)
    t = 0.5 #params.get("t", 1.0)
    a = 1 # damit ist die Brillouin-Zone -pi < k0 < pi
    k0 = params.get("k0", 1.5708) # np.pi / 2
    hbar = 1

    write_to_output(f'The parameters are set to:    t: {t:.2f}')

    # definition of laser with omega
    # z = Zeit!
    v_g = ((2 * t * a) / hbar) * math.sin(k0 * a)
    if v_g == 0:
        tau = ((2 * t * a**2) / hbar) * math.cos(k0 * a)
    else:
        tau = (((N+1) * a) / v_g)

    M = 5  # Anzahl der Sites, die das Paket überdeckt
    sigma_x = (M * a) / 2.0
    sigma_z = (2 * sigma_x) / v_g
    zf = 3 * sigma_z + 5 * sigma_z + tau # 5 = Faktor

    # New calculation for H and c's
    vecs = []
    H_tb = []

    for i in range(N+1):
        vec = np.zeros(N+1)
        vec[i] = 1
        vecs.append(vec)

    for j in range(N+1):
        col = np.zeros(N+1)
        if j == 0:
            pass
        else:
            if j-1 >= 1:
                col += -t * vecs[j-1]
            if j+1 <= N:
                col += -t * vecs[j+1]
        H_tb.append(col)

    H_tb = np.array(H_tb)

    # Anfangszustand (Grundzustand)
    psi0 = np.zeros(N+1, dtype=complex)
    psi0[0] = 1.0

    start_solve = time.perf_counter()
    # Solver
    ###############################################################

    write_to_output(f'v_g: {v_g:.4f} \t tau:{tau:.4f} \t tf:{zf:.4f}')

    #packet_out = False

    dt = 0.5
    t0 = 0
    t_all = []

    y = psi0
    ew_listen = [[] for _ in range(N+1)]

    count = 0
    while t0 < zf:
        loesung = si.solve_ivp(
            fun=schrödinger,
            t_span=[t0, t0 + dt],
            y0=y,
            method='DOP853',
            t_eval=np.linspace(t0, t0+dt, 10),
            args=(H_tb, hbar, sigma_z, k0, a),
            rtol=1e-5,
            atol=1e-8
        )

        y = loesung.y[:, -1]
        t0 = loesung.t[-1]

        for site in range(N+1):
            for i in range(len(loesung.t)):
                rho_i = loesung.y[:, i].reshape(N+1, N+1)
                wert = rho_i[site, site]
                ew_listen[site].append(np.real(wert))

        t_all.extend(loesung.t)

        # # Gesamtwahrscheinlichkeit im Gitter (ohne Vakuum)
        # prob = np.sum(np.abs(ew_listen[1:])**2)
        # # Abbruch, wenn das Paket das Gitter verlassen hat (z.B. 99% verschwunden)
        # if prob < 0.01:
        #     write_to_output("---> breaking because of probability!")
        #     packet_out = True
        #     break
    ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Computing took {(end_solve - start_solve):.4f} s', "#228B22")
    # if not packet_out:
    #     write_to_output(f"WARNUNG: tf={tf} erreicht, aber das Wellenpaket ist noch nicht aus dem System herausgewandert.", "#CD2626")

    # Plot:
    max_val = max(max(ew_listen[1]), max(ew_listen[-1]))
    if max_val < 0.01:
        y_max = 0.01
    else:
        y_max = max_val * 1.1

    ax.plot(t_all, ew_listen[1], "b-")
    ax.plot(t_all, ew_listen[-1], "r--")
    ax.grid(True)
    ax.set_ylim(0, y_max)
    ax.set_xlim(0, t_all[-1] + 1)

    ax.set_xlabel('Zeit')
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.set_title(f'Besetzungszahlen für N={N} Sites')
    ax.legend(['Site 1', 'Site N'])


## TODO: in GUI
## TODO: testen
## TODO: siehe Waveinit und checke abbruchbedingung
## TODO: Tiefes Denken bei KI an und dann Zusammenfassung Text Notizen, Formeln.
















