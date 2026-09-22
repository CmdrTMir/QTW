import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.widgets as widgets
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# ---- H-Laser ----
def H_z(H_tb, z, sigma_z, k0, a, J):
    z0 = 2 * sigma_z
    O0 = 0.2 * J                                    # O0 - Kopplungsstärke
    omega_0 = -2 * J * math.cos(k0 * a)             # Trägerfrequenz des Lasers?

    f_z = np.exp(-(z - z0)**2 / (2 * sigma_z**2))   # Gauß-Teil
    Omega = O0 * f_z * np.exp(-1j * omega_0 * z)    # Zusammensetzen (abhängig von Zeit z)

    H_laser = np.zeros(H_tb.shape, dtype=complex)
    H_laser[0, 1] = Omega
    H_laser[1, 0] = np.conj(Omega)

    return H_tb + H_laser

# ---- solver-Funktion ----
def schrödinger(t, y, H_tb, hbar, sigma_z, k0, a, J):
    psi = y
    H = H_z(H_tb, t, sigma_z, k0, a, J)
    psi_dot = -1j / hbar * (H @ psi)
    return psi_dot

def wave_omega(ax, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 2)
    J = 0.5 #params.get("t", 1.0)
    a = 1   # damit ist die Brillouin-Zone -pi < k0 < pi
    k0 = params.get("k0", 1.5708) # np.pi / 2
    hbar = 1

    write_to_output(f'The parameters are set to:    t: {J:.2f}')

    # definition of laser with omega
    # z = Zeit!
    M = 5                                                   # Anzahl der Sites, die das Paket überdeckt
    sigma_x = (M * a) / 2.0                                 # örtliche Breite des Pulses
    v_g = ((2 * J * a) / hbar) * math.sin(k0 * a)
    if v_g == 0:
        tau = ((2 * J * a**2) / hbar) * math.cos(k0 * a)    # Dissipation (2. Derivation)
        sigma_z = 2.5 / J                                   # zeitliche Breite des Pulses
    else:
        tau = (((N+1) * a) / v_g)
        sigma_z = (2 * sigma_x) / v_g

    z_puls_on = 2 * sigma_z
    z_puls_off = 2 * sigma_z * 2 # definitiv aus
    # tau = wandert durchs Gitter (oder bei v_g=0 Dissipation)
    zf = z_puls_on + z_puls_off * sigma_z + tau

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
                col += -J * vecs[j-1]
            if j+1 <= N:
                col += -J * vecs[j+1]
        H_tb.append(col)

    H_tb = np.array(H_tb)

    # Anfangszustand (Grundzustand)
    psi0 = np.zeros(N+1, dtype=complex)
    psi0[0] = 1.0

    start_solve = time.perf_counter()
    # Solver
    ###############################################################

    write_to_output(f'v_g: {v_g:.4f} \t tau:{tau:.4f} \t tf:{zf:.4f}')

    dt = 0.5
    t0 = 0
    t_all = []

    y = psi0
    ew_listen = [[] for _ in range(N+1)]

    while t0 < zf:
        loesung = si.solve_ivp(
            fun=schrödinger,
            t_span=[t0, t0 + dt],
            y0=y,
            method='DOP853',
            t_eval=np.linspace(t0, t0+dt, 10),
            args=(H_tb, hbar, sigma_z, k0, a, J),
            rtol=1e-5,
            atol=1e-8
        )

        y = loesung.y[:, -1]
        t0 = loesung.t[-1]

        for site in range(N+1):
            for i in range(len(loesung.t)):
                psi_i = loesung.y[site, i]
                wert = np.abs(psi_i)**2
                ew_listen[site].append(np.real(wert))

        t_all.extend(loesung.t)

    ###############################################################

    end_solve = time.perf_counter()
    write_to_output(f'Computing took {(end_solve - start_solve):.4f} s', "#228B22")
                    # z0
    t_reflection = 2 * sigma_z + tau
    write_to_output(f'First reflection at site N occurs at t = {t_reflection:.3f}')
    t_first_interference = 2 * sigma_z + tau - (2 * sigma_x) / v_g
    write_to_output(f'After {t_first_interference:.3f}, interference patterns start to form.')

    # Plot:
    max_val = max(max(ew_listen[1]), max(ew_listen[-1]))
    if max_val < 0.01:
        y_max = 0.01
    else:
        y_max = max_val * 1.1

    write_to_output(f'Time steps in t_all: {len(t_all)}')
    num_frames = min(700, max(400, len(t_all) // 4))
    step = max(1, len(t_all) // num_frames)
    indices = list(range(0, len(t_all), step))
    if indices[-1] != len(t_all) - 1:
        indices.append(len(t_all) - 1)
    # t_min = t_all[0]
    # t_max = t_all[-1]
    # if t_min == 0:
    #     t_min = 1e-6
    # log_times = np.logspace(np.log10(t_min), np.log10(t_max), num_frames)
    # indices = [0] + [np.argmin(np.abs(np.array(t_all) - t)) for t in log_times] + [len(t_all) - 1]
    # indices = np.unique(indices)
    #indices = range(0, len(t_all), step)
    t_selected = [t_all[i] for i in indices]

    write_to_output(f'Time steps selected: {len(t_selected)}')
    ew_selected = [] # ohne Vakuum! 0=>1
    for site in range(1, N+1):
        ew_site_selected = [ew_listen[site][i] for i in indices]
        ew_selected.append(ew_site_selected)

    fig = ax.figure
    if hasattr(fig, '_slider'):
        fig._slider.disconnect_events()
        del fig._slider
    fig.clear()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2)
    slider_ax = fig.add_axes([0.2, 0.02, 0.6, 0.04])
    slider = widgets.Slider(
        ax=slider_ax, label='Steps:  ', valmin=0, valmax=len(t_selected)-1,
        valinit=0, valstep=1
    )
    fig._slider = slider

    def get_occupations(idx):
        return [ew_selected[site][idx] for site in range(N)]

    # Initialer Plot
    idx = 0
    occupations = get_occupations(idx)
    bars = ax.bar(range(1, N+1), occupations, color='#fbb32b')
    ax.set_xlabel('Site')
    ax.set_ylabel('expectation value')
    ax.set_title(f'time = {t_selected[idx]:.3f}')
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(1, N+1))

    # Update-Funktion für Slider-Bewegung
    def update(val):
        idx = int(slider.val)
        occupations = get_occupations(idx)
        ax.clear()
        ax.bar(range(1, N+1), occupations, color='#fbb32b')
        ax.set_xlabel('Site')
        ax.set_ylabel('expectation value')
        ax.set_title(f'time = {t_selected[idx]:.3f}')
        ax.set_ylim(0, y_max)
        ax.set_xticks(range(1, N+1))
        fig.canvas.draw()

    slider.on_changed(update)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # ---------------------------------------------------------
    # Zusätzlicher Plot: Zeitcollage mit ausgewählten Zeitpunkten
    # ---------------------------------------------------------
    desired_times = [2.5, 9.0, 15.0, 17.5, 20.0, 24.0, 29.1, 45.0]
    selected_indices = [np.argmin(np.abs(np.asarray(t_all) - t)) for t in desired_times]
    fig_collage, axes = plt.subplots(
        2, 4,
        figsize=(12, 6),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()
    for ax_c, idx in zip(axes, selected_indices):
        occupations = [ew_listen[site][idx] for site in range(1, N + 1)]
        ax_c.bar(
            range(1, N + 1),
            occupations,
            color='#fbb32b'
        )
        ax_c.set_title(f'time = {t_all[idx]:.3f}')
        ax_c.set_ylim(0, y_max)
        tick_step = 2
        ticks = list(range(1, N + 1, tick_step))
        if ticks[-1] != N:
            ticks.append(N)
        ax_c.set_xticks(ticks)

    # Gemeinsame Achsenbeschriftungen
    fig_collage.supxlabel('Site')
    fig_collage.supylabel('expectation value')
    fig_collage.tight_layout()
    fig_collage.savefig(
        'wave_omega_collage.pdf',
        bbox_inches='tight'
    )
    plt.close(fig_collage)



    #Alter Plot:
    # max_val = max(max(ew_listen[1]), max(ew_listen[-1]))
    # if max_val < 0.01:
    #     y_max = 0.01
    # else:
    #     y_max = max_val * 1.1
    #
    # ax.plot(t_all, ew_listen[1], "b-")
    # ax.plot(t_all, ew_listen[-1], "r--")
    # ax.axvline(x=z_puls_on, color='orange', linestyle='-', linewidth=1.0, alpha=1.0)
    # ax.axvline(x=z_puls_off, color='orange', linestyle='-', linewidth=1.0, alpha=1.0)
    # ax.axvline(x=tau, color='orange', linestyle='-', linewidth=1.0, alpha=1.0)
    # ax.grid(True)
    # ax.set_ylim(0, y_max)
    # ax.set_xlim(0, t_all[-1] + 1)
    #
    # ax.set_xlabel('Zeit')
    # ax.set_ylabel(r'$\langle n_j \rangle$')
    # ax.set_title(f'Besetzungszahlen für N={N} Sites')
    # ax.legend(['Site 1', 'Site N'])








