import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

# helper function: calculates Chern number
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
    Ny = 20  # width of Nanoribbons
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
        # 1D-Hamiltonian for fixed kx
        H_kx = np.zeros((dim, dim), dtype=complex)
        for ny in range(Ny):
            # On-site + kx-Hopping
            H_kx[2*ny:2*ny+2, 2*ny:2*ny+2] = M - Tx * np.exp(1j*kx) - Tx.conj().T * np.exp(-1j*kx)
            # y-Hopping (Real space)
            if ny + 1 < Ny:
                H_kx[2*(ny+1):2*(ny+1)+2, 2*ny:2*ny+2] = -Ty
                H_kx[2*ny:2*ny+2, 2*(ny+1):2*(ny+1)+2] = -Ty.conj().T

        evals, evecs = eigh(H_kx)

        for i in range(dim):
            energy = evals[i]
            vec = evecs[:, i]
            # probability at the edges (y=0 und y=Ny-1)
            prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
            if prob_edges > 0.2:
                kx_edge.append(kx)
                energy_edge.append(energy)
            else:
                kx_bulk.append(kx)
                energy_bulk.append(energy)



    if collage_mode:
        m_groups = [-2.5, -1.8, 1.9, 2.0]
        #overall_idx = 1
        #for group_idx, m_group in enumerate(m_groups):
        collage_data = []
        for m_val in m_groups:
            m = float(m_val)
            C = chern_from_m(m)
            M  = np.array([[2*t*(1-m), 0], [0, 2*t*(1+m)]])

            # --- Berechnung der Bänder ---
            kx_vals = np.linspace(-np.pi, np.pi, 100)
            kx_bulk, energy_bulk = [], []
            kx_edge, energy_edge = [], []

            for kx in kx_vals:
                # 1D-Hamiltonian for fixed kx
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
                    # probability at the edges (y=0 und y=Ny-1)
                    prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
                    if prob_edges > 0.2:
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

            fig, axes = plt.subplots(4, 2, figsize=(10, 16))
            fig.text(0.32, 0.92, 'all states', ha='center', fontsize=18)
            fig.text(0.77, 0.92, 'edge states highlighted', ha='center', fontsize=18)

            for i, data in enumerate(collage_data):
                ax_left = axes[i, 0]
                ax_right = axes[i, 1]
                title_str = f'm = {data["m"]:.1f} | C = {data["C"]}'
                ax_left.set_title(f" {i+1}a)    " + title_str, fontsize=18, loc='center')
                ax_right.set_title(f" {i+1}b)    " + title_str, fontsize=18, loc='center')
                #overall_idx += 1
                # --- Linke Spalte: Alles, aber monochrom ---
                kx_all = np.concatenate([data['kx_bulk'], data['kx_edge']])
                E_all  = np.concatenate([data['energy_bulk'], data['energy_edge']])
                ax_left.scatter(kx_all, E_all, s=12, c='#888888', alpha=0.5, edgecolors='none', zorder=1)
                # --- Rechte Spalte: (Bulk + Pink) ---
                ax_right.scatter(data['kx_bulk'], data['energy_bulk'], s=12, c='#888888', alpha=0.5, edgecolors='none', zorder=1)
                ax_right.scatter(data['kx_edge'], data['energy_edge'], s=12, c='deeppink', alpha=1.0, edgecolors='none', zorder=2)

                for ax in [ax_left, ax_right]:
                    ax.set_xlim(-np.pi, np.pi)
                    ax.set_xticks([-np.pi, 0, np.pi])
                    ax.set_xticklabels([r'$k_x: -\pi$', r'$0$', r'$\pi$'], fontsize=14)
                # Y-Achsen Labels nur in der linken Spalte
                ax_left.set_ylabel('Energy', fontsize=14)

            fig.subplots_adjust(hspace=0.4)
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            plt.savefig(f'nanoribbon_collage_new.pdf', bbox_inches='tight')
            plt.close(fig)
        write_to_output(f'Collage saved with {len(collage_data)} configurations')
        return


    # --- Plot (Vektorisiert) ---
    if not collage_mode:
        ax.scatter(kx_bulk, energy_bulk, s=15, c='#888888', alpha=0.5, edgecolors='none', zorder=1)
        ax.scatter(kx_edge, energy_edge, s=15, c='deeppink', alpha=1.0, edgecolors='none', zorder=2)

        ax.set_title(f'Nanoribbon | m = {m:.1f} | C = {C}')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
        #ax.set_xlabel(r'$k_x$', fontsize=14, loc='right')
        ax.text(0.5, 0.05, r'$k_x$', transform=ax.transAxes, ha='center',  fontsize=14)
        ax.set_ylabel("Energy", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.3)
        #ax.figure.savefig("Nano1.png", dpi=400, bbox_inches="tight")




# TODO: re-write to delete redundant code






