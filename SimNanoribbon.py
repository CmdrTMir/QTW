import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

def chern_from_m(m):
    if np.isclose(abs(m), 2.0, atol=1e-9) or np.isclose(m, 0.0, atol=1e-9):
        return 0
    if -2.0 < m < 0.0:
        return 1
    if 0.0 < m < 2.0:
        return -1
    return 0

def Nanoribbon(ax, write_to_output, params):
    # ---- Variablen ----
    t = 1 #params.get("t", 2)
    m_string = params.get("m", 1.0)
    m = float(m_string)
    Ny = 20  # Breite des Nanoribbons
    dim = 2 * Ny
    C = chern_from_m(m)
    collage_mode = False

    write_to_output(f"The parameters are: t = {t} \t Ny = {Ny}.")
    write_to_output(f"Collage Mode is: {collage_mode}")

    M  = np.array([[2*t*(1-m), 0], [0, 2*t*(1+m)]])
    Tx = np.array([[t, 1j*t], [1j*t, -t]])
    Ty = np.array([[t, t], [-t, -t]])

    # --- Berechnung der Bänder ---
    kx_vals = np.linspace(-np.pi, np.pi, 100)
    kx_bulk, energy_bulk = [], []
    kx_edge, energy_edge = [], []

    for kx in kx_vals:
        # 1D-Hamiltonian für festes kx
        H_kx = np.zeros((dim, dim), dtype=complex)
        for ny in range(Ny):
            # On-site + kx-Hopping
            H_kx[2*ny:2*ny+2, 2*ny:2*ny+2] = M - Tx * np.exp(1j*kx) - Tx.conj().T * np.exp(-1j*kx)
            # y-Hopping (Realraum)
            if ny + 1 < Ny:
                H_kx[2*(ny+1):2*(ny+1)+2, 2*ny:2*ny+2] = -Ty
                H_kx[2*ny:2*ny+2, 2*(ny+1):2*(ny+1)+2] = -Ty.conj().T

        evals, evecs = eigh(H_kx)

        for i in range(dim):
            energy = evals[i]
            vec = evecs[:, i]
            # Wahrscheinlichkeit an den Rändern (y=0 und y=Ny-1)
            prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
            if prob_edges > 0.3:
                kx_edge.append(kx)
                energy_edge.append(energy)
            else:
                kx_bulk.append(kx)
                energy_bulk.append(energy)



    if collage_mode:
        m_values = [-2.1, -1.8, 0.1, 0.9, 1.8, 2.1]
        collage_data = []
        for m_val in m_values:
            m = float(m_val)
            C = chern_from_m(m)
            M  = np.array([[2*t*(1-m), 0], [0, 2*t*(1+m)]])

            # --- Berechnung der Bänder ---
            kx_vals = np.linspace(-np.pi, np.pi, 100)
            kx_bulk, energy_bulk = [], []
            kx_edge, energy_edge = [], []

            for kx in kx_vals:
                # 1D-Hamiltonian für festes kx
                H_kx = np.zeros((dim, dim), dtype=complex)
                for ny in range(Ny):
                    # On-site + kx-Hopping
                    H_kx[2*ny:2*ny+2, 2*ny:2*ny+2] = M - Tx * np.exp(1j*kx) - Tx.conj().T * np.exp(-1j*kx)
                    # y-Hopping (Realraum)
                    if ny + 1 < Ny:
                        H_kx[2*(ny+1):2*(ny+1)+2, 2*ny:2*ny+2] = -Ty
                        H_kx[2*ny:2*ny+2, 2*(ny+1):2*(ny+1)+2] = -Ty.conj().T

                evals, evecs = eigh(H_kx)

                for i in range(dim):
                    energy = evals[i]
                    vec = evecs[:, i]
                    # Wahrscheinlichkeit an den Rändern (y=0 und y=Ny-1)
                    prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
                    if prob_edges > 0.3:
                        kx_edge.append(kx)
                        energy_edge.append(energy)
                    else:
                        kx_bulk.append(kx)
                        energy_bulk.append(energy)

            # Daten speichern
            collage_data.append({
                'm': m,
                'C': C,
                'kx_bulk': kx_bulk.copy(),
                'energy_bulk': energy_bulk.copy(),
                'kx_edge': kx_edge.copy(),
                'energy_edge': energy_edge.copy()
            })


        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, data in enumerate(collage_data):
            ax_c = axes[idx]
            # Bulk
            ax_c.scatter(data['kx_bulk'], data['energy_bulk'],
                        s=10, c='#888888', alpha=0.4, edgecolors='none', zorder=1)
            # Edge
            ax_c.scatter(data['kx_edge'], data['energy_edge'],
                        s=25, c='deeppink', alpha=1.0, edgecolors='none', zorder=2)

            ax_c.set_title(f'm = {data["m"]:.1f} | C = {data["C"]}', fontsize=20)
            ax_c.set_xlim(-np.pi, np.pi)
            ax_c.set_xticks([-np.pi, 0, np.pi])
            ax_c.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'], fontsize=20)
            if idx >= 3:  # Nur untere Reihe
                ax_c.set_xlabel(r'$k_x$', fontsize=20)
            if idx % 3 == 0:  # Nur linke Spalte
                ax_c.set_ylabel('Energy', fontsize=20)

        fig.suptitle('Nanoribbon: illustrating the band gap', fontsize=24, y=0.98)
        plt.tight_layout()
        plt.savefig('nanoribbon_collage.pdf', bbox_inches='tight')
        plt.close(fig)

        write_to_output(f'Collage saved with {len(collage_data)} configurations')
        return


    # --- Plot (Vektorisiert) ---
    if not collage_mode:
        # Bulk-Zustände: Klein, grau, halbtransparent
        ax.scatter(kx_bulk, energy_bulk,
                s=15,               # Fläche des Punktes
                c='#888888',
                alpha=0.4,          # Ersetzt fill_opacity
                edgecolors='none',  # Entfernt den hässlichen schwarzen Rand
                zorder=1)           # Zeichne sie im Hintergrund

        # Edge-Zustände: Größer, pink, voll deckend
        ax.scatter(kx_edge, energy_edge,
                s=40,               # Deutlich größer, um sie hervorzuheben
                c='deeppink',
                alpha=1.0,
                edgecolors='none',
                zorder=2)

        ax.set_title(f'Nanoribbon | m = {m:.1f} | C = {C}')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_xlabel(r'$k_x$', fontsize=14)
        ax.set_ylabel("Energy", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)






