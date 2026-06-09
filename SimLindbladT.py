import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time

# ===============================================================================
# Hilfsfunktionen:
#def inspect_operator(op, label):
#    print("\n---[ %s ]---" % label)
#    #print("Dims: %s" % str(op.dims))
#    print("Shape: %s x %s" % (op.shape[0], op.shape[1]))
#    print("Matrix:")
#    real_part = np.real(op)
#    print(real_part)
#    print("-" * 80)

def fun_rho_dot(t, y, H, c, c_dag, gamma, N):
    rho = y.reshape(2**N, 2**N)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = gamma * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()
# ===============================================================================

# global variable
ss_values = None

def lindblad_time(ax, N_val, t_val, gamma_val, d_val, mode):
# ---- Variablen ----
    sigma = qt.destroy(2)
    sig_dag = qt.create(2)
    sigz = qt.sigmaz()
    eye = qt.qeye(2)

    N = N_val
    t = t_val
    gamma = gamma_val

    H = 0
    c = []
    c_dag = []

    # ---- berechne H ----
    for j in range(N):
        ops = []
        for k in range(N):
            if k < j:
                ops.append(sigz)
            elif k == j:
                ops.append(sigma)
            else:
                ops.append(eye)
        c_j = qt.tensor(ops)
        c.append(c_j)

    for op in c:
        c_dag.append(op.dag())


# create T matrix for more than nearest neighbour tunneling
    d = d_val
    T = np.zeros((N, N))
    for j in range(N):
        for jp in range(N):
            if j != jp:
                val = t / abs(j - jp)**d
                T[j, jp] = round(val,2) if val > 1e-3 else 0.0

# berechne H
    for j in range(N-1):
        for jp1 in range(j+1, N):
            H += T[j,jp1] * (c_dag[j] * c[jp1] + c_dag[jp1] * c[j])
    H = -H

    ### Um c's in np arrays umzuwandeln
    np_c = []
    np_c_dag = []

    for cop in c:
        np_c.append(cop.full())

    for cdop in c_dag:
        np_c_dag.append(cdop.full())

    c = np_c
    c_dag = np_c_dag

    H = H.full()

    # ---- Lindblad-Terme ----
    #dim = (2**N)**2
    plot_len = int(max(10/gamma, 10/t))
    rho0 = qt.fock_dm([2]*N, [0]*N).full()
    t_span = (0, plot_len)
    t_eval = np.linspace(0, plot_len, plot_len*4)

    start_solve = time.perf_counter()

    loesung = si.solve_ivp(
        fun=fun_rho_dot,
        t_span=t_span,
        y0=rho0.flatten(),
        method='DOP853',
        t_eval=t_eval,
        args=(H, c, c_dag, gamma, N)
    )

    ew_listen = []

    for site in range(N):
        ew_site = []
        for i in range(len(loesung.t)):
            rho_i = loesung.y[:, i].reshape(2**N, 2**N)
            wert = np.trace(rho_i @ c_dag[site] @ c[site])
            ew_site.append(wert)
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






def lindblad_ss(ax, N_val, t_val, gamma_val, d_val):
    global ss_values
    steady_values = []

    if ss_values is not None:
        steady_values = ss_values
    else:
        steady_values = lindblad_time(ax, N_val, t_val, gamma_val, d_val, "steady")

    ax.clear()
    ax.set_ylim(0, 1.1)
    ax.bar(range(1, N_val+1), steady_values)
    ax.set_xticks(range(1, N_val+1))
    ax.set_xlabel("Site")
    ax.set_ylabel("Besetzungszahl")
    ax.set_title("Steady State")



def T_lindblad(ax, N_val, t_val, gamma_val, d_val, mode):
    if mode == "time":
        lindblad_time(ax, N_val, t_val, gamma_val, d_val, mode)
    elif mode == "steady":
        lindblad_ss(ax, N_val, t_val, gamma_val, d_val)




















