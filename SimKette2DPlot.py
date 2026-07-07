import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# ---- Lindblad-Terme ----
def fun_rho_dot(t, y, H, c, c_dag, kappa, gamma, N):
    rho = y.reshape(N+1, N+1)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = kappa * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()

def chain_to_2D(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 9)
    t = 0.5 #params.get("t", 1.0)
    gamma = 0.1 #params.get("gamma", 1.0)
    kappa = 0.001 #params.get("kappa", 4.0)
    mode = params.get("mode", "time")

    write_to_output(f'The parameters are set to:    t: {t:.2f};  in: {kappa:.3f};  out: {gamma:.2f}')

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

    # Anfangszustand (Grundzustand)
    rho0 = np.zeros((N+1, N+1), dtype=complex)
    rho0[0, 0] = 1.0

    start_solve = time.perf_counter()
    # Solver
    ###############################################################
    #n_ss: kappa is IN
    J = (4 * kappa * gamma * t**2) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_1_theo = 1 - J / kappa
    n_j_theo = (kappa * (gamma**2 + 4 * t**2)) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
    n_N_theo = J / gamma

    tau = ((N + 1)**3) / (2 * math.pi**2 * (kappa + gamma))
    tf = 2 * tau

    write_to_output(f'tau: {tau:.2f}')
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
    ew_listen = [[] for _ in range(N+1)]

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

        for site in range(N+1):
            for i in range(len(loesung.t)):
                rho_i = loesung.y[:, i].reshape(N+1, N+1)
                wert = rho_i[site, site]
                ew_listen[site].append(np.real(wert))

        t_all.extend(loesung.t)

        current_n_1 = ew_listen[1][-1]
        current_n_j = 0
        current_n_N = ew_listen[N][-1]

        delta_1 = float('inf')
        delta_N = float('inf')
        delta_j = float('inf')
        diff_j = float('inf')

        if t0 > tau:
            last_n_1 = ew_listen[1][-2]
            last_n_j = 0
            last_n_N = ew_listen[N][-2]
            delta_1 = abs(last_n_1 - current_n_1) / max(abs(current_n_1), 1e-14)
            delta_N = abs(last_n_N - current_n_N) / max(abs(current_n_N), 1e-14)

        if N > 2:
            current_n_j = ew_listen[N//2][-1]
            diff_j = abs(current_n_j - n_j_theo)
            if t0 > tau:
                last_n_j = ew_listen[N//2][-2]
                delta_j = delta_j = abs(last_n_j - current_n_j) / max(abs(current_n_j), 1e-14)
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
        write_to_output(f"Aktuelle Besetzungen weichen numerisch um" + "\n"
                        + f"diff_1: {diff_1:.6f}" + "\n"
                        + f"diff_j: {diff_j:.6f}" + "\n"
                        + f"diff_N: {diff_N:.6f}" + "\n"
                        + "von den analytischen Werten ab.", "#CD2626")

    steady_state = [ew_listen[site][-1] for site in range(1, N+1)]
    # Plot:
    if int(np.sqrt(N))**2 != N:
        raise ValueError(f"N = {N} ist keine Quadratzahl. Es muss n^2 = N gelten.")
    n = int(math.sqrt(N))
    grid = np.zeros((n, n))

    for site in range(N):               # site = 0, 1, 2, ..., N-1
        r = site // n                   # Zeile: 0, 0, 0, 1, 1, 1, 2, 2, 2
        if r % 2 == 0:                  # gerade Zeile: links → rechts
            c = site % n                # Spalte: 0, 1, 2, 0, 1, 2, ...
        else:                           # ungerade Zeile: rechts → links
            c = n - 1 - (site % n)      # Spalte: 2, 1, 0, 2, 1, 0, ...
        grid[r, c] = steady_state[site]

    # --- Scatter-Plot ---
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    # Flachmachen für scatter
    x_flat = X.flatten()
    y_flat = Y.flatten()
    densities = grid.flatten()

    # Achtung: density könnte 0 sein, dann size=10 (Mindestgröße)
    sizes = 100 * densities + 10

    # Scatter-Plot
    scatter = ax.scatter(
        x_flat, y_flat,
        s=sizes,
        c=densities,
        cmap='hot',
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    # --- Site-Nummern (Schlangen-Mapping) ---
    offset = 0.15
    for r in range(n):
        for c in range(n):
            if r % 2 == 0:
                site = r * n + c +1
            else:
                site = r * n + (n - 1 - c) +1
            y_pos = n - 1 - r
            ax.text(c + offset, y_pos + offset, str(site), ha='left', va='bottom', color='black', fontsize=7)

    # Colorbar
    fig = ax.figure
    fig.colorbar(scatter, ax=ax, label='Dichte (steady state)')

    ax.set_title('Steady-State Dichteverteilung (Schlangen-Mapping)')
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(0, n))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(np.arange(n-1, -1, -1))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
























