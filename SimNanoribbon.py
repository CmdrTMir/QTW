import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time
import math

def Nanoribbon(ax, write_to_output, params):
    # ---- Variablen ----
    t = 1 #params.get("t", 2)
    m_string = params.get("m", 2)
    m = int(m_string)
    Ny = 20  # Breite des Nanoribbons
    dim = 2 * Ny

    write_to_output(f"The parameters are: t = {t} \t Ny = {Ny}.")

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
            # Wahrscheinlichkeit an den Rändern (m=0 und m=Ny-1)
            prob_edges = np.sum(np.abs(vec[0:2])**2) + np.sum(np.abs(vec[-2:])**2)
            if prob_edges > 0.3:
                kx_edge.append(kx)
                energy_edge.append(energy)
            else:
                kx_bulk.append(kx)
                energy_bulk.append(energy)

    # --- Plot (Vektorisiert) ---

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

    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_xlabel(r'$k_x$', fontsize=14)
    ax.set_ylabel(r'$E$', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)









