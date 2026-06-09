import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

def fun_rho_dot(t, y, H, c, c_dag, gamma, N):
    rho = y.reshape(2**N, 2**N)
    kommutator = -1j * (H @ rho - rho @ H)
    L_in = gamma * (c_dag[0] @ rho @ c[0] - 1/2 * (c[0] @ c_dag[0] @ rho + rho @ c[0] @ c_dag[0]))
    L_out = gamma * (c[-1] @ rho @ c_dag[-1] - 1/2 * (c_dag[-1] @ c[-1] @ rho + rho @ c_dag[-1] @ c[-1]))

    rho_dot = kommutator + L_in + L_out
    return rho_dot.flatten()

# ignoring d and mode here, not needed
def analytical(ax, N_val, t_val, gamma_val, d, mode):
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

    for j in range(N-1):
        jp1 = (j + 1) % N
        H += c_dag[j] * c[jp1] + c_dag[jp1] * c[j]
    H = -t * H

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

    # ============== analytische Lösung ===============================

    dim = 4**N
    M = np.zeros((dim, dim), dtype=complex)

    for i in range(dim-1):
        e_i = np.zeros(dim)
        e_i[i] = 1
        e_i = e_i.reshape(2**N, 2**N)
        result = fun_rho_dot(0, e_i, H, c, c_dag, gamma, N)
        M[:, i] = result

    #U, s, Vh = np.linalg.svd(M, full_matrices=False)
    #rho_stat_vec = Vh[-1, :]
    #rho_stat = rho_stat_vec.reshape(2**N, 2**N)
    #spur = np.trace(rho_stat)
    #rho_stat /= spur

    ew_0 = []
    ew_n = []
    rho0 = qt.fock_dm([2]*N, [0]*N).full()
    plot_len = 10
    t_list = np.linspace(0, plot_len, plot_len*4)

    eigenwerte, V = np.linalg.eig(M)
    V_inv = np.linalg.inv(V)
    c0 = V_inv @ rho0.flatten()

    for t in t_list:
        c_t = c0 * np.exp(eigenwerte * t)
        rho_vec_t = V @ c_t
        rho_t = rho_vec_t.reshape(2**N, 2**N)
        wert_0 = np.trace(rho_t @ c_dag[0] @ c[0])
        wert_n = np.trace(rho_t @ c_dag[-1] @ c[-1])
        ew_0.append(wert_0)
        ew_n.append(wert_n)

    # Plot:

    ax.plot(t_list, ew_0, "b-")
    ax.plot(t_list, ew_n, "r--")
    ax.grid(True)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, plot_len+1)

    ax.set_xlabel('Zeit')
    ax.set_ylabel(r'$\langle n_j \rangle$')
    ax.set_title(f'Besetzungszahlen für N={N} Sites')
    ax.legend(['Site 1', 'Site N'])







