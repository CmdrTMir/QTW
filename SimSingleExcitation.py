import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time

from SimJWT import create_JWT_t

# global variable
ss_values = None

# ---- Lindblad-Terme ----
def fun_rho_dot(t, y, H, c, c_dag, kappa, gamma, N):
    rho = y.reshape(2**N, 2**N)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = kappa * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()


def single_excitation_time(ax, write_to_output, params):
# ---- Variablen ----
    N = params.get("N", 2)
    t = params.get("t", 1.0)
    gamma = params.get("gamma", 1.0)
    kappa = params.get("kappa", 4.0)
    mode = params.get("mode", "time")
    tf = params.get("tf", 100)

    c, c_dag, H = create_JWT_t(N, t)

    # Anfangszustand (Grundzustand)
    plot_len = 1000 #int(max(10/kappa, 10/t))
    rho0 = qt.fock_dm([2]*N, [0]*N).full()
    #t_span = (0, plot_len)

    start_solve = time.perf_counter()
    # Solver
    ###############################################################
    #n_ss: kappa is IN
    J = (4 * kappa * gamma * t**2) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_1_theo = 1 - J / kappa
    n_j_theo = (kappa * (gamma**2 + 4 * t**2)) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_N_theo = J / gamma

    ss_reached = False

    eps = 1e-6
    dt = 0.1
    t0 = 0
    #tf = t_span[1]
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
                rho_i = loesung.y[:, i].reshape(2**N, 2**N)
                wert = np.trace(rho_i @ c_dag[site] @ c[site])
                ew_listen[site].append(np.real(wert))

        t_all.extend(loesung.t)

        current_n_1 = ew_listen[0][-1]
        current_n_N = ew_listen[N-1][-1]
        if N > 2:
            current_n_j = ew_listen[N//2][-1]
        else:
            current_n_j = 0.5

        diff_1 = abs(current_n_1 - n_1_theo)
        diff_N = abs(current_n_N - n_N_theo)
        diff_j = abs(current_n_j - n_j_theo)

        if diff_1 < eps and diff_N < eps and diff_j < eps:
            write_to_output(f"Steady State erreicht bei t={t0:.2f}")
            ss_reached = True
            break
    ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Solving took {(end_solve - start_solve):.4f} s')
    if not ss_reached:
        write_to_output(f"WARNUNG: tf={tf} erreicht, aber Steady State wurde nicht erreicht.", "#CD2626")
        write_to_output(f"Aktuelle Besetzungen weichen noch um" + "\n"
                        + f"diff_1: {diff_1:.4f}" + "\n"
                        + f"diff_j: {diff_j:.4f}" + "\n"
                        + f"diff_N: {diff_N:.4f}" + "\n"
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
    #ax.text(6.1, 0.01, f'Solving took {(end_solve - start_solve):.4f} s')

    steady_state = [ew_listen[site][-1] for site in range(N)]
    global ss_values
    ss_values = steady_state

    if mode == "time":
        return None
    elif mode == "steady":
        return steady_state






def single_excitation_ss(ax, write_to_output, params):
    steady_values = []

    if ss_values is not None:
        steady_values = ss_values
    else:
        params["mode"] = "steady"
        steady_values = single_excitation_time(ax, write_to_output, params)

    N = params.get("N", 2)
    ax.clear()
    ax.set_ylim(0, 1.1)
    ax.bar(range(1, N+1), steady_values)
    ax.set_xticks(range(1, N+1))
    ax.set_xlabel("Site")
    ax.set_ylabel("Besetzungszahl")
    ax.set_title("Steady State")


def single_excitation(ax, write_to_output, params):
    mode = params.get("mode", "time")
    if mode == "time":
        single_excitation_time(ax, write_to_output, params)
    elif mode == "steady":
        single_excitation_ss(ax, write_to_output, params)
