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


def single_excitation_time(ax, params):
# ---- Variablen ----
    N = params.get("N", 2)
    t = params.get("t", 1.0)
    gamma = params.get("gamma", 1.0)
    kappa = params.get("kappa", 4.0)
    mode = params.get("mode", "time")

    c, c_dag, H = create_JWT_t(N, t)

    # Anfangszustand (Grundzustand)
    #dim = (2**N)**2
    plot_len = 50 #int(max(10/kappa, 10/t))
    rho0 = qt.fock_dm([2]*N, [0]*N).full()
    t_span = (0, plot_len)
    t_eval = np.linspace(0, plot_len, plot_len*4)

    start_solve = time.perf_counter()
    # Solver
    loesung = si.solve_ivp(
        fun=fun_rho_dot,
        t_span=t_span,
        y0=rho0.flatten(),
        method='DOP853',
        t_eval=t_eval,
        args=(H, c, c_dag, kappa, gamma, N)
    )

    ew_listen = []

    for site in range(N):
        ew_site = []
        for i in range(len(loesung.t)):
            rho_i = loesung.y[:, i].reshape(2**N, 2**N)
            wert = np.trace(rho_i @ c_dag[site] @ c[site])
            ew_site.append(np.real(wert))
        ew_listen.append(ew_site)


    end_solve = time.perf_counter()

    # Plot:
    ax.plot(loesung.t, ew_listen[0], "b-")
    ax.plot(loesung.t, ew_listen[-1], "r--")
    ax.grid(True)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, plot_len+1)

    ax.set_xlabel('Zeit')
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.set_title(f'Besetzungszahlen für N={N} Sites')
    ax.legend(['Site 1', 'Site N'])
    ax.text(6.1, 0.01, f'Solving took {(end_solve - start_solve):.4f} s')

    steady_state = [ew_listen[site][-1] for site in range(N)]
    ss_values = steady_state

    if mode == "time":
        return None
    elif mode == "steady":
        return steady_state






def single_excitation_ss(ax, params):
    global ss_values
    steady_values = []

    if ss_values is not None:
        steady_values = ss_values
    else:
        params["mode"] = "steady"
        steady_values = single_excitation_time(ax, params)

    N = params.get("N", 2)
    ax.clear()
    ax.set_ylim(0, 1.1)
    ax.bar(range(1, N+1), steady_values)
    ax.set_xticks(range(1, N+1))
    ax.set_xlabel("Site")
    ax.set_ylabel("Besetzungszahl")
    ax.set_title("Steady State")


def single_excitation(ax, params):
    mode = params.get("mode", "time")
    if mode == "time":
        single_excitation_time(ax, params)
    elif mode == "steady":
        single_excitation_ss(ax, params)
