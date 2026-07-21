import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# global variable
ss_values = None

# ---- Lindblad-Terme ----
def fun_rho_dot(t, y, H, c, c_dag, gamma, N):
    rho = y.reshape(N+1, N+1)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = 0
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()

def wave_init(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 2)
    t = 0.5 #params.get("t", 1.0)
    gamma = 0.1 #params.get("gamma", 1.0)
    a = 1 # damit ist die Brillouin-Zone -pi < k0 < pi
    k0 = params.get("k0", 1.5708) # np.pi / 2

    write_to_output(f'The parameters are set to:    t: {t:.2f};  in: wavepacket;  out: {gamma:.2f}')

    # New calculation for H and c's
    vecs = []
    H = []

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
        H.append(col)

    H = np.array(H)

    c0_dagger = np.zeros((N+1, N+1))
    c0_dagger[1,0] = 1
    c0 = c0_dagger.T
    cN = np.zeros((N+1, N+1))
    cN[0,N] = 1
    cN_dagger = cN.T

    c = []
    c.append(c0)
    c.append(cN)
    c_dag = []
    c_dag.append(c0_dagger)
    c_dag.append(cN_dagger)

    # Anfangszustand (Grundzustand) definieren mit Wellenpaket
    M = 5  # Anzahl der Sites, die das Paket überdeckt
    sigma_x = (M * a) / 2.0
    psi = np.zeros((N+1), dtype=complex)
    prefactor = 1.0 / (2 * np.pi * sigma_x**2)**(0.25)
    for j in range(1, N+1):
        x = (j - 1) * a
        psi[j] = prefactor * np.exp(-x**2 / (4 * sigma_x**2)) * np.exp(1j * k0 * x)
    norm = np.sqrt(np.sum(np.abs(psi[1:])**2))
    psi[1:] /= norm
    rho0 = np.outer(psi, psi.conj())   # rho = |psi><psi|
    rho0[0, 0] = 1.0

    start_solve = time.perf_counter()
    # Solver
    ###############################################################
    # Umrechnung
    hbar = 1
    v_g = ((2 * t * a) / hbar) * math.sin(k0 * a)
    if v_g == 0:
        tau = ((2 * t * a**2) / hbar) * math.cos(k0 * a)
    else:
        tau = (((N+1) * a) / v_g)
    tf = tau * 20 # 2 = Faktor

    write_to_output(f'v_g: {v_g:.4f} \t tau:{tau:.4f} \t tf:{tf:.4f}')

    packet_out = False
    deltas = False

    eps_delta = 1e-3
    dt = 0.5
    t0 = 0
    t_all = []

    y = rho0.flatten()
    ew_listen = [[] for _ in range(N+1)]

    count = 0
    while t0 < tf:
        loesung = si.solve_ivp(
            fun=fun_rho_dot,
            t_span=[t0, t0 + dt],
            y0=y,
            method='DOP853',
            t_eval=np.linspace(t0, t0+dt, 10),
            args=(H, c, c_dag, gamma, N),
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

        # Gesamtwahrscheinlichkeit im Gitter (ohne Vakuum)
        prob = np.sum(np.abs(ew_listen[1:])**2)
        # Abbruch, wenn das Paket das Gitter verlassen hat (z.B. 99% verschwunden)
        if prob < 0.01:
            write_to_output("---> breaking because of probability!")
            packet_out = True
            break
    ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Solving took {(end_solve - start_solve):.4f} s', "#228B22")
    if not packet_out:
        write_to_output(f"WARNUNG: tf={tf} erreicht, aber das Wellenpaket ist noch nicht aus dem System herausgewandert.", "#CD2626")

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























