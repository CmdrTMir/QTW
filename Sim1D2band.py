import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math
import pdb

##### --- New Visualisations: #####
def plot_d_vector_path(dx_list, dy_list, dz_list):
    """Plot the path of d(k) in 3D space."""
    fig = plt.figure(num="d-vector path", figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(dx_list, dy_list, dz_list, 'b-', linewidth=2, label='$\\mathbf{d}(k)$ path')
    # Mark the origin (Dirac point location)
    ax.scatter([0], [0], [0], color='red', s=50, label='Origin')
    ax.set_xlabel('$d_x$')
    ax.set_ylabel('$d_y$')
    ax.set_zlabel('$d_z$')
    ax.set_title('Path of $\\mathbf{d}(k)$ in 3D Space')
    ax.legend()
    fig.canvas.manager.set_window_title('d-vector path')
    plt.draw()
    plt.pause(0.001)

def bloch_sphere_coordinates(state):
    """
    Convert a 2-component spinor to Bloch sphere coordinates.
    Returns (theta, phi) in radians.
    """
    alpha, beta = state[0], state[1]
    # Normalize (just in case)
    norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
    alpha = alpha / norm
    beta = beta / norm
    theta = 2 * np.arccos(np.abs(alpha))
    phi = np.angle(beta) - np.angle(alpha)
    # Ensure phi is in [-pi, pi]
    phi = np.arctan2(np.sin(phi), np.cos(phi))
    return theta, phi

def plot_bloch_sphere_states(k_points, u_plus_list, u_minus_list):
    """Plot both bands' states on the Bloch sphere."""
    fig = plt.figure(num="Bloch sphere states", figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Draw the Bloch sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.3, edgecolor='none')
    # Plot states
    for u_state in u_plus_list:
        theta, phi = bloch_sphere_coordinates(u_state)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        ax.scatter(x, y, z, color='red', s=10, alpha=0.5)

    for u_state in u_minus_list:
        theta, phi = bloch_sphere_coordinates(u_state)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        ax.scatter(x, y, z, color='blue', s=10, alpha=0.5)

    ax.set_xlabel('$\\langle \\sigma_x \\rangle$')
    ax.set_ylabel('$\\langle \\sigma_y \\rangle$')
    ax.set_zlabel('$\\langle \\sigma_z \\rangle$')
    ax.set_title('Bloch States on the Bloch Sphere\n(Red = Upper band, Blue = Lower band)')
    fig.canvas.manager.set_window_title('Bloch sphere states')
    plt.draw()
    plt.pause(0.001)

def plot_d_hat_on_bloch_sphere(d_hat_list):
    """Plot the path of d_hat on the Bloch sphere."""
    fig = plt.figure(num="d-hat on Bloch sphere", figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # Draw the sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color='lightgray', alpha=0.2, edgecolor='none')

    if d_hat_list:
        d_hat_list = np.array(d_hat_list)
        ax.plot(d_hat_list[:, 0], d_hat_list[:, 1], d_hat_list[:, 2],
                'b-', linewidth=2, label='$\\hat{\\mathbf{d}}(k)$ path')

    # Mark the north/south poles
    ax.scatter([0], [0], [1], color='gold', s=30, label='+z (pure $s$)')
    ax.scatter([0], [0], [-1], color='purple', s=30, label='-z (pure $p$)')
    ax.set_xlabel('$\\hat{d}_x$')
    ax.set_ylabel('$\\hat{d}_y$')
    ax.set_zlabel('$\\hat{d}_z$')
    ax.set_title('Path of $\\hat{\\mathbf{d}}(k)$ on the Bloch Sphere')
    ax.legend()
    fig.canvas.manager.set_window_title('d-hat on Bloch sphere')
    plt.draw()
    plt.pause(0.001)

#####################################################################################################################
def twoband_1D(ax, write_to_output, params):
    write_to_output(f"Offene Figuren vor dem Plot: {plt.get_fignums()}")
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        if fig is not ax.figure:  # GUI-Figure nicht schließen
            plt.close(fig)
    write_to_output(f"Offene Figuren nach dem Löschen: {plt.get_fignums()}")
    # ---- variables ----
    t_ss = params.get("t_ss", 1.0)
    t_pp = params.get("t_pp", 1.0)
    r_tsp = params.get("r_tsp", 0.0)
    t_tsp = params.get("t_tsp", 0.0)
    r_tps = params.get("r_tps", 0.0)
    t_tps = params.get("t_tps", 0.0)
    e_s = params.get("e_s", 0.0)
    e_p = params.get("e_p", 0.0)
    ks = 200
    a = 1.0

    write_to_output(f'r_tsp: {r_tsp}; t_tsp: {t_tsp}; r_tps: {r_tps}; t_tps: {t_tps}')
    t_sp = r_tsp * math.e**(-1j*t_tsp)
    t_ps = r_tps * math.e**(-1j*t_tps)
    write_to_output(f't_sp: {t_sp}')
    write_to_output(f't_ps: {t_ps}')

    k_werte = np.linspace(-np.pi/a, np.pi/a, ks)

    s_band = []
    p_band = []
    bandlücke_k = []
    V_k_abs = []
    evs_minus = []
    evs_plus = []
    T = np.array([[t_ss, t_sp],
                  [t_ps, t_pp]], dtype=complex)

    ##### variables for visualisations:
    dx_list = []
    dy_list = []
    dz_list = []
    d_hat_list = []

    for k in k_werte:
        H_k = -T * np.exp(1j * k * a) - T.conj().T * np.exp(-1j * k * a)
        # --- check hermiticity ---
        if not np.allclose(H_k, H_k.conj().T, atol=1e-12):
            write_to_output(f"Error: H(k) is NOT Hermitian at k={k:.3f}!\n"
                                 f"H = {H_k}\nH^dag = {H_k.conj().T}")
        # --- check off-diagonal matches the expected h_sp(k) ---
        # H_01(k) = -(t_sp * e^{ika} + conj(t_ps) * e^{-ika})
        h_sp = -(t_sp * np.exp(1j * k * a) + np.conj(t_ps) * np.exp(-1j * k * a))
        if not np.allclose(H_k[0, 1], h_sp, atol=1e-12):
            write_to_output(f"Error: off-diagonal mismatch at k={k:.3f}!\n"
                                 f"Computed: {H_k[0, 1]}\nExpected: {h_sp}")

        d0 = (e_s + e_p) / 2 - (t_ss + t_pp) * math.cos(k*a)
        dz = (e_s - e_p) / 2 - (t_ss - t_pp) * math.cos(k*a)
        E_plus = d0 + math.sqrt(dz**2 + abs(h_sp)**2)
        E_minus = d0 - math.sqrt(dz**2 + abs(h_sp)**2)
        p_band.append(E_plus)
        s_band.append(E_minus)
        bandlücke_k.append(E_plus - E_minus)
        V_k_abs.append(abs(h_sp))
        dx_list.append(np.real(h_sp))
        dy_list.append(-np.imag(h_sp))
        dz_list.append(dz)
        # --- Eigenvectors
        d_mag = np.sqrt(dz**2 + np.abs(h_sp)**2)
        # Handle the gapless case (bands touch)
        if d_mag < 1e-12:
            d_mag = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        else:
            d_hat = np.array([np.real(h_sp), -np.imag(h_sp), dz]) / d_mag
            d_hat_list.append(d_hat)
        # Strategy: Use the formula that DOES NOT divide by zero
        # Formula A works unless dz ≈ -|d|
        # Formula B works unless dz ≈ +|d|
        if abs(d_mag + dz) > 1e-12:  # Safe to use Formula A
            denom = np.sqrt(2 * d_mag * (d_mag + dz))
            u_plus = np.array([d_mag + dz, np.conj(h_sp)]) / denom
            u_minus = np.array([-h_sp, d_mag + dz]) / denom
            evs_plus.append(u_plus)
            evs_minus.append(u_minus)
        else:  # Use Formula B (dz ≈ -|d|)
            denom = np.sqrt(2 * d_mag * (d_mag - dz))
            u_plus = np.array([np.conj(h_sp), d_mag - dz]) / denom
            u_minus = np.array([d_mag - dz, -np.conj(h_sp)]) / denom
            evs_plus.append(u_plus)
            evs_minus.append(u_minus)

    min_gap = np.min(bandlücke_k)
    idx_min = np.argmin(bandlücke_k)
    k_min = k_werte[idx_min]

    ### ref-band
    p_band_ref = []
    s_band_ref = []
    t_sp_ref = 0
    t_ps_ref = 0
    t_ss_ref = 1
    t_pp_ref = 0.5
    for k in k_werte:
        h_sp = -(t_sp_ref * np.exp(1j * k * a) + np.conj(t_ps_ref) * np.exp(-1j * k * a))
        d0 = (e_s + e_p) / 2 - (t_ss_ref + t_pp_ref) * math.cos(k*a)
        dz = (e_s - e_p) / 2 - (t_ss_ref - t_pp_ref) * math.cos(k*a)
        E_plus = d0 + math.sqrt(dz**2 + abs(h_sp)**2)
        E_minus = d0 - math.sqrt(dz**2 + abs(h_sp)**2)
        p_band_ref.append(E_plus)
        s_band_ref.append(E_minus)

    ### --- Plot ---
    fig = ax.figure
    ax.plot(k_werte, p_band, label='(p)-band', color='red')
    ax.plot(k_werte, s_band, label='(s)-band', color='blue')
    ax.plot(k_werte, V_k_abs, label=r'$|V(k)|$', color='#42994B')
    #ax.axhline(y=e_s, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)
    #ax.axhline(y=e_p, color='orange', linestyle=':', linewidth=1.0, alpha=0.7)
    ax.plot(k_werte, p_band_ref, color='red', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.plot(k_werte, s_band_ref, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0)
    ax.axvline(x=0.0, color='gray', linestyle='-', linewidth=1.0)

    if t_sp != 0:
        ax.fill_between(k_werte, s_band, p_band, color='gray', alpha=0.2, label='band gap')
        #ax.vlines(x=k_min, ymin=s_band[idx_min], ymax=p_band[idx_min], color='black', linestyle='--', linewidth=1)
        #ax.vlines(x=-k_min, ymin=s_band[idx_min], ymax=p_band[idx_min], color='black', linestyle='--', linewidth=1)
        write_to_output(f"Minimale Bandlücke: {min_gap:.6f} bei k = {k_min:.4f}")

    ax.set_xlabel(r'$k$ (wave number)', fontsize=12)
    ax.set_ylabel(r'$E(k)$', fontsize=12)
    #ax.lengend()
    ax.legend(loc='center left', bbox_to_anchor=(1.045, 0.5))
    param_text = (
        f"t_ss = {t_ss:.2f}\n"
        f"t_pp = {t_pp:.2f}\n"
        f"t_sp: r={r_tsp:.2f}   θ:{t_tsp:.2f} \n"
        f"t_ps: r={r_tps:.2f}   θ:{t_tps:.2f} \n"
        f"e_s = {e_s:.2f}\n"
        f"e_p = {e_p:.2f}"
    )
    for t in fig.texts:
        t.remove()
    #        0.75, 0.79
    fig.text(0.95, 0.14, param_text, fontsize=12, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    #fig.savefig("plot.png", dpi=400, bbox_inches="tight")

    # execute new plots:
    plot_d_vector_path(dx_list, dy_list, dz_list)
    plot_bloch_sphere_states(k_werte, evs_plus, evs_minus)
    plot_d_hat_on_bloch_sphere(d_hat_list)
    plt.pause(0.1)





