import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.linalg import eigh
import scipy.integrate as si
import qutip as qt
import time

def cs(N):
    # ---- Variablen ----
    sigma = qt.destroy(2)
    sig_dag = qt.create(2)
    sigz = qt.sigmaz()
    eye = qt.qeye(2)

    c = []
    c_dag = []

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

    return c, c_dag

# normal t
def create_JWT_t(N, t):
    H = 0
    c, c_dag = cs(N)
    # ---- berechne H ----
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
    H = H.full() #.to("csr_matrix").data

    return c, c_dag, H

# with T matrix
def create_JWT_T(N, t, d_val):
    H = 0
    c, c_dag = cs(N)

    # create T matrix for more than nearest neighbour tunneling
    ## neue Berechnung für t
    #  t_all = t / (j-jp)**d

    d = d_val
    T = np.zeros((N, N))
    for j in range(N):
        for jp in range(N):
            if j != jp:
                val = t / abs(j - jp)**d
                T[j, jp] = round(val,2) if val > 1e-3 else 0.0
    # indices = np.arange(N)
    # distances = np.abs(indices[:, None] - indices)
    # T_raw = t / (distances ** d)
    # T_raw[distances == 0] = 0.0
    # T = np.round(T_raw, 2)
    # T[T < 1e-3] = 0.0
    # print(T)


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

    return c, c_dag, H


