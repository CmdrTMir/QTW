import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

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
    rho0 = qt.fock_dm([2]*N, [0]*N).full()

    start_solve = time.perf_counter()
    # Solver
    ###############################################################
    #n_ss: kappa is IN
    J = (4 * kappa * gamma * t**2) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_1_theo = 1 - J / kappa
    n_j_theo = (kappa * (gamma**2 + 4 * t**2)) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_N_theo = J / gamma

    tau = ((N + 1)**3) / (2 * math.pi**2 * (kappa + gamma))

    write_to_output(f'tau: {tau:.2f}')
    if tau > tf:
        write_to_output(f"WARNUNG: tf={tf} < tau={tau:.2f}, steady state cannot be reliably reached", "#CD2626")
    write_to_output(f'n_1 analytischer Wert: {n_1_theo:.6f} ')
    write_to_output(f'n_j analytischer Wert: {n_j_theo:.6f} ')
    write_to_output(f'n_N analytischer Wert: {n_N_theo:.6f} ')

    ss_reached = False
    ss_delta = False

    eps_delta = 1e-3
    eps_diff = 1e-4
    dt = 0.5
    t0 = 0
    t_all = []

    y = rho0.flatten()
    ew_listen = [[] for _ in range(N)]

    count = 0
    while t0 < tf:
        loesung = si.solve_ivp(
            fun=fun_rho_dot,
            t_span=[t0, t0 + dt],
            y0=y,
            method='DOP853',
            t_eval=np.linspace(t0, t0+dt, 10),
            args=(H, c, c_dag, kappa, gamma, N),
            rtol=1e-5,
            atol=1e-8
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
        current_n_j = 0
        current_n_N = ew_listen[N-1][-1]

        delta_1 = float('inf')
        delta_N = float('inf')
        delta_j = float('inf')
        diff_j = float('inf')

        if t0 > tau:
            last_n_1 = ew_listen[0][-2]
            last_n_j = 0
            last_n_N = ew_listen[N-1][-2]
            delta_1 = abs(last_n_1 - current_n_1) / max(abs(current_n_1), 1e-14)
            delta_N = abs(last_n_N - current_n_N) / max(abs(current_n_N), 1e-14)

        if N > 2:
            current_n_j = ew_listen[N//2][-1]
            diff_j = abs(current_n_j - n_j_theo)
            if t0 > tau:
                last_n_j = ew_listen[N//2][-2]
                delta_j = abs(last_n_j - current_n_j) / max(abs(current_n_j), 1e-14)
        else:
            diff_j = 0.0

        if delta_j < eps_delta and delta_1 < eps_delta and delta_N < eps_delta:
            count += 1
        else:
            count = 0

        diff_1 = abs(current_n_1 - n_1_theo)
        diff_N = abs(current_n_N - n_N_theo)

        if count == 5:
            write_to_output(f"Steady State erreicht als Delta bei t={t0:.2f}")
            write_to_output(f"deltas remaining: eps={eps_delta}" + "\n"
                        + f"delta_1: {delta_1:.10f}" + "\n"
                        + f"delta_j: {delta_j:.10f}" + "\n"
                        + f"delta_N: {delta_N:.10f}" + "\n")
            write_to_output(f"Aktuelle Besetzungen weichen noch um" + "\n"
                        + f"diff_1: {diff_1:.6f}" + "\n"
                        + f"diff_j: {diff_j:.6f}" + "\n"
                        + f"diff_N: {diff_N:.6f} ab." + "\n")
            ss_delta = True
            break

        if diff_1 < eps_diff and diff_N < eps_diff and diff_j < eps_diff:
            write_to_output(f"Steady State erreicht durch eps={eps_diff} bei t={t0:.2f}")
            ss_reached = True
            break
    ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Solving took {(end_solve - start_solve):.4f} s', "#228B22")
    if not ss_reached and not ss_delta:
        write_to_output(f"WARNUNG: tf={tf} erreicht, aber Steady State wurde nicht erreicht.", "#CD2626")
        write_to_output(f"Aktuelle Besetzungen weichen noch um" + "\n"
                        + f"diff_1: {diff_1:.6f}" + "\n"
                        + f"diff_j: {diff_j:.6f}" + "\n"
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
