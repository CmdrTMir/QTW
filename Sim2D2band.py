import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

ks = 200

##### --- New Visualisations: #####
def plot_combined_visualisations(k_x_werte, k_y_werte, s_band, p_band, d_vecs):
    fig = plt.figure(num="2D Combined visualisations", figsize=(14, 12))
    KX, KY = np.meshgrid(k_x_werte, k_y_werte, indexing='ij')

    # ---- 1. Bulk energy bands  (oben links) ----
    E_minus = np.array(s_band).reshape(ks, ks)
    E_plus  = np.array(p_band).reshape(ks, ks)
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.plot_surface(
        KX, KY, E_minus,
        color='#00AEB3',
        alpha=1.0,
    )
    ax.plot_surface(
        KX, KY, E_plus,
        color='#8C368C',
        alpha=1.0,
    )
    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_zlabel(r'$E$')
    ax.set_title('Bulk energy bands')
    ax.view_init(elev=15, azim=-50)

    # ---- 2. d(k) im 3D d-Raum  (oben rechts) ----
    d_array = np.array(d_vecs).reshape(ks, ks, 3)
    DX = d_array[:, :, 0]
    DY = d_array[:, :, 1]
    DZ = d_array[:, :, 2]

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.scatter(
        DX.flatten(),
        DY.flatten(),
        DZ.flatten(),
        c=KX.flatten(),
        cmap='Blues',
        s=3,
        alpha=0.2
    )
    ax2.scatter(
        [0], [0], [0],
        color='red',
        s=50,
        label='Origin'
    )
    ax2.set_xlabel(r'$d_x$')
    ax2.set_ylabel(r'$d_y$')
    ax2.set_zlabel(r'$d_z$')
    ax2.set_title(r'$\mathbf{d}(k_x,k_y)$')
    ax2.view_init(elev=15, azim=-50)
    ax2.legend()

    # ---- 3. normierter Vektor d_hat(k) (unten links) ----
    norm = np.sqrt(
        DX**2 +
        DY**2 +
        DZ**2
    )
    DHX = DX / norm
    DHY = DY / norm
    DHZ = DZ / norm

    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    ax3.scatter(
        DHX.flatten(),
        DHY.flatten(),
        DHZ.flatten(),
        c=KX.flatten(),
        cmap='Oranges',
        s=4,
        alpha=0.2
    )
    ax3.set_xlabel(r'$\hat d_x$')
    ax3.set_ylabel(r'$\hat d_y$')
    ax3.set_zlabel(r'$\hat d_z$')
    ax3.set_title(r'$\hat{\mathbf{d}}(k_x,k_y)$ — Bloch sphere')
    ax3.view_init(elev=15, azim=-50)
    # gleiche Skalierung der Achsen
    ax3.set_xlim(-1, 1)
    ax3.set_ylim(-1, 1)
    ax3.set_zlim(-1, 1)
    ticks = [-1, -0.5, 0, 0.5, 1]
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_zticks(ticks)

    # ---- 4. Brillouin zone (unten rechats) ----
    ax4 = fig.add_axes([0.65, 0.10, 0.30, 0.30])
    ax4.pcolormesh(
        KX,
        KY,
        E_plus - E_minus,
        shading='gouraud',
        cmap='OrRd_r'
    )
    ax4.set_xlabel(r'$k_x$')
    ax4.set_ylabel(r'$k_y$')
    ax4.set_title(r'Band gap $E_+ - E_-$')
    ax4.axhline(0, color='gray', linewidth=0.8)
    ax4.axvline(0, color='gray', linewidth=0.8)


    fig.subplots_adjust(
        left=0.08,    # Abstand vom linken Rand
        right=0.99,   # Abstand vom rechten Rand (Platz für Legende)
        top=0.92,     # Abstand vom oberen Rand
        bottom=0.08,  # Abstand vom unteren Rand
        wspace=0.2,   # Abstand zwischen Subplots (horizontal)
        hspace=0.3    # Abstand zwischen Subplots (vertikal)
    )
    fig.canvas.manager.set_window_title('2D Combined visualisations')
    plt.draw()
    plt.pause(0.001)



def twoband_2D(ax, write_to_output, params):
    write_to_output(f"Offene Figuren vor dem Plot: {plt.get_fignums()}")
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        if fig is not ax.figure:  # GUI-Figure nicht schließen
            plt.close(fig)
    write_to_output(f"Offene Figuren nach dem Löschen: {plt.get_fignums()}")
    # ---- Variablen ----
    N = params.get("N", 9)
    t = params.get("t", 2)
    #tf = params.get("tf", 100)
    a = 1.0

    # New calculation for H and c's
    if int(np.sqrt(N))**2 != N:
        write_to_output(f"ERROR: N = {N} isn't a square number. Make sure that n^2 = N.", "#CD2626")
        #raise ValueError(f"N = {N} ist keine Quadratzahl. Es muss n^2 = N gelten.")
    n = int(math.sqrt(N))

    # Energie
    k_x_werte = np.linspace(-np.pi/a, np.pi/a, ks)
    k_y_werte = np.linspace(-np.pi/a, np.pi/a, ks)

    p_band = []
    s_band = []
    d_vecs = []

    for k_x in k_x_werte:
        for k_y in k_y_werte:
            ev_minus = 2 * t - 2 * t * math.sqrt(math.sin(k_x)**2 + math.sin(k_y)**2 + (1 + math.cos(k_x) + math.cos(k_y))**2)
            ev_plus = 2 * t + 2 * t * math.sqrt(math.sin(k_x)**2 + math.sin(k_y)**2 + (1 + math.cos(k_x) + math.cos(k_y))**2 )
            s_band.append(ev_minus)
            p_band.append(ev_plus)
            dx = math.sin(k_x)
            dy = math.sin(k_y)
            dz = (1 + math.cos(k_x) + math.cos(k_y))
            d_vec = -2 * t * np.array([[dx], [dy], [dz]])
            d_vecs.append(d_vec)


    ### --- Plot --- stimmt irgendwas nicht...
    fig = ax.figure
    ax_dummy = fig.add_subplot(2, 2, 4, projection='3d')
    KX, KY = np.meshgrid(k_x_werte, k_y_werte, indexing='ij')
    Z = np.sin(KX) * np.cos(KY)
    ax_dummy.plot_surface(
        KX,
        KY,
        Z,
        color='#00AEB3',
        alpha=1.0
    )
    ax_dummy.set_xlabel(r'$k_x$')
    ax_dummy.set_ylabel(r'$k_y$')
    ax_dummy.set_zlabel(r'$E$')
    ax_dummy.set_title('Dummy Plot')
    ax_dummy.view_init(elev=15, azim=-50)

    plot_combined_visualisations(k_x_werte, k_y_werte, s_band, p_band, d_vecs)
























