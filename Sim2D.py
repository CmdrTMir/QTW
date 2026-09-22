import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# globale Konfiguration:
plt.rcParams.update({
    'font.size': 14,            # allgemeine Schriftgröße
    'axes.titlesize': 14,       # Titel
    'axes.labelsize': 14,       # Achsenbeschriftungen
    'xtick.labelsize': 12,      # x-Achsenticks
    'ytick.labelsize': 12,      # y-Achsenticks
    'legend.fontsize': 12,      # Legende
    'figure.titlesize': 16,     # Figurentitel
})

def plot_time_evolution_with_slider(write_to_output, ax, ew_listen, t_all, n, N):
    write_to_output(f"Offene Figuren vor dem Plot: {plt.get_fignums()}", "#fbcf2b")
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        if fig is not ax.figure:  # GUI-Figure nicht schließen
            plt.close(fig)

    num_frames = min(700, max(400, len(t_all) // 4))
    step = max(1, len(t_all) // num_frames)
    indices = list(range(0, len(t_all), step))
    if indices[-1] != len(t_all) - 1:
        indices.append(len(t_all) - 1)
    t_selected = [t_all[i] for i in indices]
    ew_selected = []
    for site in range(1, N+1):
        ew_site_selected = [ew_listen[site][i] for i in indices]
        ew_selected.append(ew_site_selected)

    write_to_output(f"Time evolution: {len(t_all)} frames -> {len(indices)} frames (subsampled)")
    # vmin/vmax über alle Frames
    all_densities = []
    for site_data in ew_selected:
        all_densities.extend(site_data)
    vmin = min(all_densities)
    vmax = max(all_densities)

    # Container
    class PlotContainer:
        pass
    container = PlotContainer()
    container.t_selected = t_selected
    container.ew_selected = ew_selected
    container.n = n
    container.N = N
    container.ax = ax
    container.current_idx = len(indices) - 1
    container.vmin = vmin
    container.vmax = vmax

    # --- INITIAL PLOT erstellen (NICHT in einer Funktion!) ---
    frame_idx = container.current_idx
    steady_state = [ew_selected[site][frame_idx] for site in range(N)]
    grid = np.zeros((n, n))
    for site in range(N):
        r = site // n
        c = site % n
        grid[r, c] = steady_state[site]
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    densities = grid.flatten()
    # Scatter erstellen und speichern
    container.scatter = ax.scatter(
        x_flat, y_flat,
        s=400,
        c=densities,
        cmap='hot',
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5,
        vmin=vmin,
        vmax=vmax
    )
    # Site-Nummern (bleiben gleich)
    offset = 0.15
    for r in range(n):
        for c in range(n):
            site = r * n + c + 1
            y_pos = n - 1 - r
            ax.text(c + offset, y_pos + offset, str(site),
                   ha='left', va='bottom', color='black', fontsize=12)
    fig = ax.figure
    container.cbar = fig.colorbar(container.scatter, ax=ax)
    container.cbar.set_label('Expectation value (density)', fontsize=14)
    container.cbar.ax.tick_params(labelsize=12)

    container.title = ax.set_title(f'Time evolution (t = {t_selected[frame_idx]:.2f})', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(0, n))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(np.arange(n-1, -1, -1))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)

    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
    container.slider = Slider(
        ax=ax_slider,
        label='Time Step',
        valmin=0,
        valmax=len(indices) - 1,
        valinit=container.current_idx,
        valstep=1
    )
    def update(val):
        frame_idx = int(val)
        steady_state = [ew_selected[site][frame_idx] for site in range(N)]
        grid = np.zeros((n, n))
        for site in range(N):
            r = site // n
            c = site % n
            grid[r, c] = steady_state[site]
        densities = grid.flatten()
        container.scatter.set_array(densities)
        container.title.set_text(f'Time evolution (t = {t_selected[frame_idx]:.2f})')
        fig.canvas.draw_idle()

    container.slider.on_changed(update)
    container.slider.set_val(0)
    return container

# ---- Lindblad-Terme ----
def fun_rho_dot(t, y, H, c, c_dag, kappa, gamma, N):
    rho = y.reshape(N+1, N+1)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = kappa * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()

def dimRed_2D(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 9)
    t_h = params.get("t", 0.1)
    t_v = params.get("t_v", 0.1)
    gamma = params.get("gamma", 0.1)
    kappa = params.get("kappa", 0.005)
    tf = params.get("tf", 100)

    # New calculation for H and c's
    if int(np.sqrt(N))**2 != N:
        write_to_output(f"ERROR: N = {N} isn't a square number. Make sure that n^2 = N.", "#CD2626")
        #raise ValueError(f"N = {N} ist keine Quadratzahl. Es muss n^2 = N gelten.")
    n = int(math.sqrt(N))

    write_to_output(f'The parameters are set to:    in: {kappa:.3f};  out: {gamma:.2f}')

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
            # horizontal tunneling
            if (j-1) % n != 0:
                col += -t_h * vecs[j-1]
            if j % n != 0:
                col += -t_h * vecs[j+1]
            # vertical tunneling
            if j+n <= N:
                col += -t_v * vecs[j+n]
            if j-n > 0:
                col += -t_v * vecs[j-n]
        H.append(col)

    H = np.array(H)
    #print(H)

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
    #########
    ######### Stimmt nicht mehr ?????????????????
    #########
    ###############################################################
    #n_ss: kappa is IN
#     J = (4 * kappa * gamma * t**2) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
#     n_1_theo = 1 - J / kappa
#     n_j_theo = (kappa * (gamma**2 + 4 * t**2)) / ((kappa + gamma)*(4 * t**2 + kappa * gamma))
#     n_N_theo = J / gamma
# ###############################################################
#     write_to_output(f'n_1 analytischer Wert: {n_1_theo:.6f} ')
#     write_to_output(f'n_j analytischer Wert: {n_j_theo:.6f} ')
#     write_to_output(f'n_N analytischer Wert: {n_N_theo:.6f} ')
#
#     ss_reached = False
    ss_delta = False

    eps_delta = 1e-3
    eps_diff = 1e-4
    dt = 0.5
    t0 = 0
    t_all = []

    y = rho0.flatten()
    ew_listen = [[] for _ in range(N+1)]

    count = 0
    min_time = 100.0
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
        #diff_j = float('inf')

        if t0 > min_time:
            last_n_1 = ew_listen[1][-2]
            last_n_j = 0
            last_n_N = ew_listen[N][-2]
            delta_1 = abs(last_n_1 - current_n_1)
            delta_N = abs(last_n_N - current_n_N)

        if N > 2:
            current_n_j = ew_listen[N//2][-1]
            #diff_j = abs(current_n_j - n_j_theo)
            if t0 > min_time:
                last_n_j = ew_listen[N//2][-2]
                delta_j = abs(last_n_j - current_n_j)
        #else:
            #diff_j = 0.0

        if delta_j < eps_delta and delta_1 < eps_delta and delta_N < eps_delta:
            count += 1
        else:
            count = 0

         #diff_1 = abs(current_n_1 - n_1_theo)
         #diff_N = abs(current_n_N - n_N_theo)

        if count == 5:
            write_to_output(f"Steady State reached as delta at t={t0:.2f}")
            write_to_output(f"deltas remaining: eps={eps_delta}" + "\n"
                        + f"delta_1: {delta_1:.10f}" + "\n"
                        + f"delta_j: {delta_j:.10f}" + "\n"
                        + f"delta_N: {delta_N:.10f}")
            ss_delta = True
            break

         #if diff_1 < eps_diff and diff_N < eps_diff and diff_j < eps_diff:
         #    write_to_output(f"Steady State erreicht durch eps={eps_diff} bei t={t0:.2f}")
         #    ss_reached = True
         #    break
     ###############################################################


    end_solve = time.perf_counter()
    write_to_output(f'Solving took {(end_solve - start_solve):.4f} s')
    if not ss_delta:
        write_to_output(f"WARNING: tf={tf} reached, but Steady State couldn't be reached.", "#CD2626")
        write_to_output("Increase 'tf' parameter.")

    steady_state = [ew_listen[site][-1] for site in range(1, N+1)]
    # Plot:
    grid = np.zeros((n, n))

    for site in range(N):           # site = 0, 1, 2, ..., N-1
        r = site // n               # Zeile: 0, 0, 0, 1, 1, 1, 2, 2, 2
        c = site % n                # Spalte: 0, 1, 2, 0, 1, 2, 0, 1, 2
        grid[r, c] = steady_state[site]

    # --- Scatter-Plot ---
    x = np.arange(n)
    y = np.arange(n)
    X, Y = np.meshgrid(x, y)

    # Flachmachen für scatter
    x_flat = X.flatten()
    y_flat = Y.flatten()
    densities = grid.flatten()

    # Scatter-Plot
    scatter = ax.scatter(
        x_flat, y_flat,
        s=400,
        c=densities,
        cmap='hot',
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    # --- Site-Nummern ---
    offset = 0.15
    for r in range(n):
        for c in range(n):
            site = r * n + c + 1
            y_pos = n - 1 - r
            ax.text(c + offset, y_pos + offset, str(site), ha='left', va='bottom', color='black', fontsize=12)

    # Colorbar
    fig = ax.figure
    #fig.colorbar(scatter, ax=ax, label='Expectation value (density)')
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Expectation value (density)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.set_title('Steady state density distribution', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(np.arange(0, n))
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(np.arange(n-1, -1, -1))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)

    ax.figure.savefig("plot2D.png", dpi=400, bbox_inches="tight")

    # 2. Zeitentwicklung mit Slider (NEU)
    fig2, ax2 = plt.subplots()
    container = plot_time_evolution_with_slider(write_to_output, ax2, ew_listen, t_all, n, N)
    plt.show()
























