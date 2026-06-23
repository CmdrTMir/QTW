import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time


# ---- Lindblad-Terme ----
def fun_rho_dot(t, y, H, c, c_dag, kappa, gamma, N):
    rho = y.reshape(N+1, N+1)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = kappa * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()

def dim_reduction(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 2)
    t = params.get("t", 1.0)
    gamma = params.get("gamma", 1.0)
    kappa = params.get("kappa", 4.0)
    mode = params.get("mode", "time")
    tf = params.get("tf", 100)

    # New calculation for H and c's
    vec = np.zeros(N+1)
    vecs = []
    H = []

    for i in range(N+1):
        if i == 0:
            vec[i] = 0
        else:
            vec[i] = 1
        vecs.append(vec)
        vec = np.zeros(N+1)

    col = np.zeros(N+1)

    for j in range(N+1):
        if j+1 <= N and j-1 >= 0:
            col = -t * (vecs[j+1] + vecs[j-1])
        if j == N:
            col = -t * (vecs[N-1])
        H.append(col)
        col = np.zeros(N+1)

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

    # Anfangszustand (Grundzustand)
    #plot_len = 1000 #int(max(10/kappa, 10/t))
    rho0 = np.zeros((N+1, N+1), dtype=complex)
    rho0[0, 0] = 1.0

    start_solve = time.perf_counter()
    # Solver
    ###############################################################
    #n_ss: kappa is IN
    # J = (4 * kappa * gamma * t**2) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    # n_1_theo = 1 - J / kappa
    # n_j_theo = (kappa * (gamma**2 + 4 * t**2)) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    # n_N_theo = J / gamma
    #n_ss: kappa is IN
    n_1_theo = (gamma**2 + 4 * t**2) / (gamma**2 + 4 * t**2 + 4 * kappa * t**2)
    n_j_theo = kappa / gamma
    n_N_theo = (4 * kappa * t**2) / (gamma**2 + 4 * t**2 + 4 * kappa * t**2)

    write_to_output(f'n_1 analytischer Wert: {n_1_theo:.6f} ')
    write_to_output(f'n_j analytischer Wert: {n_j_theo:.6f} ')
    write_to_output(f'n_N analytischer Wert: {n_N_theo:.6f} ')

    ss_reached = False

    eps = 1e-6
    dt = 0.1
    t0 = 0
    t_all = []

    y = rho0.flatten()
    ew_listen = [[] for _ in range(N)]

    while t0 < tf:
        loesung = si.solve_ivp(
            fun=fun_rho_dot,
            t_span=[t0, t0 + dt],
            y0=y,
            method='DOP853',
            t_eval=np.linspace(t0, t0+dt, 10),
            args=(H, c, c_dag, kappa, gamma, N)
        )

        y = loesung.y[:, -1]
        t0 = loesung.t[-1]

        for site in range(N):
            for i in range(len(loesung.t)):
                rho_i = loesung.y[:, i].reshape(N+1, N+1)
                wert = rho_i[site, site] #np.trace(rho_i @ c_dag[site] @ c[site])
                ew_listen[site].append(np.real(wert))

        t_all.extend(loesung.t)

        current_n_1 = ew_listen[0][-1]
        current_n_N = ew_listen[N-1][-1]
        # if N > 2:
        #     current_n_j = ew_listen[N//2][-1]
        # else:
        #     current_n_j = 0.5

        diff_1 = abs(current_n_1 - n_1_theo)
        diff_N = abs(current_n_N - n_N_theo)
        #diff_j = abs(current_n_j - n_j_theo)

        if diff_1 < eps and diff_N < eps:# and diff_j < eps:
            write_to_output(f"Steady State erreicht bei t={t0:.2f}")
            ss_reached = True
            break
    ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Solving took {(end_solve - start_solve):.4f} s')
    if not ss_reached:
        write_to_output(f"WARNUNG: tf={tf} erreicht, aber Steady State wurde nicht erreicht.", "#CD2626")
        write_to_output(f"Aktuelle Besetzungen weichen noch um" + "\n"
                        + f"diff_1: {diff_1:.6f}" + "\n"
                        #+ f"diff_j: {diff_j:.6f}" + "\n"
                        + f"diff_N: {diff_N:.6f}" + "\n"
                        + "von den analytischen Werten ab.", "#CD2626")
        write_to_output("Erhöhen Sie 'tf' in den Parametern.")

    # Plot:
    ax.plot(t_all, ew_listen[0], "b-")
    ax.plot(t_all, ew_listen[-1], "r--")
    ax.grid(True)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, t_all[-1] + 1)

    ax.set_xlabel('Zeit')
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.set_title(f'Besetzungszahlen für N={N} Sites')
    ax.legend(['Site 1', 'Site N'])
























