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

def eigenvals_2D(ax1, ax2, write_to_output, params):
    # ---- Variablen ----
    N = params.get("N", 9)
    t_h = params.get("t", 0.1)
    t_v = params.get("t_v", 0.1)
    gamma = 0.1 #params.get("gamma", 1.0)
    kappa = 0.001 #params.get("kappa", 4.0)

    # New calculation for H and c's
    if int(np.sqrt(N))**2 != N:
        write_to_output(f"FEHLER: N = {N} ist keine Quadratzahl. Es muss n^2 = N gelten.", "#CD2626")
        raise ValueError(f"N = {N} ist keine Quadratzahl. Es muss n^2 = N gelten.")
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

    # ========== Numerische Eigenwerte ==========
    eig_num = np.linalg.eigvalsh(H)
    eig_num_sorted = np.sort(eig_num)

    # ========== Analytische Eigenwerte (Sinustransformation) ==========
    eig_ana = [0.0]
    for p in range(1, n + 1):
        for q in range(1, n + 1):
            E = -2 * t_h * np.cos(np.pi * p / (n + 1)) - 2 * t_v * np.cos(np.pi * q / (n + 1))
            eig_ana.append(E)
    eig_ana_sorted = np.sort(eig_ana)

    # ========== Plot ==========
    indices = np.arange(len(eig_num_sorted))

    # ax1: Numerisch
    ax1.scatter(indices, eig_num_sorted, s=20, color='blue', label='Numerisch (H)')
    ax1.set_xlabel('Index (sortiert)')
    ax1.set_ylabel('Energie')
    ax1.set_title('Numerische Eigenwerte')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ax2: Analytisch
    ax2.scatter(indices, eig_ana_sorted, s=20, marker='x', color='red', label='Analytisch (Sinus)')
    ax2.set_xlabel('Index (sortiert)')
    ax2.set_ylabel('Energie')
    ax2.set_title('Analytische Eigenwerte')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ========== Differenz als Text ausgeben ==========
    diff = eig_num_sorted - eig_ana_sorted
    write_to_output("Differenz (Numerisch - Analytisch) für jeden Index:")
    for i, d in enumerate(diff):
        write_to_output(f"Index {i:3d}: {d:.2e}")
    # Zusätzlich max. Abweichung
    max_diff = np.max(np.abs(diff))
    write_to_output(f"\nMaximale absolute Abweichung: {max_diff:.2e}")
















